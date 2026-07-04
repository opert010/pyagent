"""
LangGraph 多 Agent 编排骨架。

流程：Planner 分解任务 → 按序调度 Researcher / Simulation / Analyst / Lab
     → Reviewer 汇总审核 → 输出最终答案。

用法：
    from langgraph.checkpoint.memory import MemorySaver
    from orchestrator import get_orchestrator_graph

    graph = get_orchestrator_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [("user", "设计一种低介电常数封装材料并评估热稳定性")]},
        config={"configurable": {"thread_id": "session-001"}},
    )
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from tools.registry import AGENT_PROMPTS, get_tools_for_agent

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

AgentRole = Literal["researcher", "simulation", "analyst", "lab"]
SUPPORTED_AGENTS: set[str] = {"researcher", "simulation", "analyst", "lab"}


class SubTask(TypedDict):
    id: str
    agent: AgentRole
    description: str
    status: Literal["pending", "running", "done"]


class AgentState(TypedDict):
    """LangGraph 全局状态。"""

    messages: Annotated[list, add_messages]
    task_plan: list[SubTask]
    current_task_index: int
    current_agent: str
    tool_results: dict[str, dict]
    rag_context: list[str]
    final_answer: str


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=ZHIPU_API_KEY,
    )


def _extract_json(text: str) -> dict:
    """从 LLM 输出中解析 JSON，兼容 markdown 代码块。"""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


def _get_user_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, tuple) and msg[0] == "user":
            return msg[1]
    return ""


def _normalize_task_plan(raw_tasks: list[dict]) -> list[SubTask]:
    """校验并规范化 Planner 输出的子任务列表。"""
    normalized: list[SubTask] = []
    for index, task in enumerate(raw_tasks, start=1):
        agent = task.get("agent", "researcher")
        if agent not in SUPPORTED_AGENTS:
            agent = "researcher"
        normalized.append(
            {
                "id": str(task.get("id", index)),
                "agent": agent,
                "description": task.get("description", "完成用户请求的相关分析"),
                "status": "pending",
            }
        )
    return normalized


def _default_task_plan(user_query: str) -> list[SubTask]:
    """Planner 解析失败时的兜底计划。"""
    return [
        {
            "id": "1",
            "agent": "researcher",
            "description": f"检索与问题相关的材料与工艺知识：{user_query}",
            "status": "pending",
        },
        {
            "id": "2",
            "agent": "analyst",
            "description": "综合检索结果，给出结构化分析与建议",
            "status": "pending",
        },
    ]


# ---------------------------------------------------------------------------
# 节点实现
# ---------------------------------------------------------------------------


def planner_node(state: AgentState) -> dict:
    """Planner：将用户请求分解为可执行的子任务 DAG（线性骨架版）。"""
    llm = _get_llm()
    user_query = _get_user_query(state)

    system_prompt = """你是材料科学研发任务规划专家。
请将用户需求分解为 2-5 个子任务，分配给以下 Agent 之一：
- researcher：文献/知识库检索、材料与工艺调研
- simulation：第一性原理、分子动力学等模拟计算（可先给出计算方案）
- analyst：数据统计、物性对比、实验结果解读
- lab：实验方案、合成路线、表征 SOP

只输出 JSON，格式如下：
{
  "tasks": [
    {"id": "1", "agent": "researcher", "description": "..."},
    {"id": "2", "agent": "simulation", "description": "..."}
  ]
}"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]
    )

    try:
        payload = _extract_json(response.content)
        task_plan = _normalize_task_plan(payload.get("tasks", []))
    except (json.JSONDecodeError, TypeError, ValueError):
        task_plan = _default_task_plan(user_query)

    if not task_plan:
        task_plan = _default_task_plan(user_query)

    plan_summary = "\n".join(
        f"- [{t['agent']}] {t['description']}" for t in task_plan
    )
    return {
        "task_plan": task_plan,
        "current_task_index": 0,
        "current_agent": "planner",
        "tool_results": {},
        "rag_context": [],
        "final_answer": "",
        "messages": [
            AIMessage(content=f"任务规划完成，共 {len(task_plan)} 个子任务：\n{plan_summary}")
        ],
    }


def _make_react_agent_node(agent_name: str, react_agent):
    """将 create_react_agent 封装为 LangGraph 节点。"""

    def node(state: AgentState, config) -> dict:
        idx = state["current_task_index"]
        task = state["task_plan"][idx]
        task["status"] = "running"

        task_msg = HumanMessage(
            content=(
                f"【子任务 {task['id']} | {agent_name}】\n"
                f"{task['description']}\n\n"
                f"原始用户问题：{_get_user_query(state)}"
            )
        )

        result = react_agent.invoke(
            {"messages": state["messages"] + [task_msg]},
            config,
        )

        new_messages = result["messages"][len(state["messages"]) + 1 :]
        summary = new_messages[-1].content if new_messages else ""

        tool_results = dict(state.get("tool_results") or {})
        tool_results[task["id"]] = {
            "agent": agent_name,
            "description": task["description"],
            "summary": summary,
        }

        task["status"] = "done"
        updated_plan = list(state["task_plan"])
        updated_plan[idx] = task

        return {
            "messages": new_messages,
            "task_plan": updated_plan,
            "tool_results": tool_results,
            "current_agent": agent_name,
        }

    return node


def _make_llm_agent_node(agent_name: str, system_prompt: str):
    """无工具的占位 Agent 节点（Simulation / Analyst / Lab 骨架）。"""

    def node(state: AgentState) -> dict:
        llm = _get_llm()
        idx = state["current_task_index"]
        task = state["task_plan"][idx]
        task["status"] = "running"

        prior_results = state.get("tool_results") or {}
        context_block = json.dumps(prior_results, ensure_ascii=False, indent=2)

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"子任务：{task['description']}\n"
                        f"用户问题：{_get_user_query(state)}\n"
                        f"前序 Agent 结果：\n{context_block}"
                    )
                ),
            ]
        )

        tool_results = dict(prior_results)
        tool_results[task["id"]] = {
            "agent": agent_name,
            "description": task["description"],
            "summary": response.content,
        }

        task["status"] = "done"
        updated_plan = list(state["task_plan"])
        updated_plan[idx] = task

        return {
            "messages": [AIMessage(content=f"【{agent_name}】\n{response.content}")],
            "task_plan": updated_plan,
            "tool_results": tool_results,
            "current_agent": agent_name,
        }

    return node


def advance_task_node(state: AgentState) -> dict:
    """推进到下一个子任务。"""
    return {"current_task_index": state["current_task_index"] + 1}


def reviewer_node(state: AgentState) -> dict:
    """Reviewer：汇总各 Agent 输出，生成带溯源的最终答案。"""
    llm = _get_llm()
    tool_results = state.get("tool_results") or {}

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是材料科学研发审核专家。"
                    "请综合各 Agent 的输出，给出结构化最终报告，包含："
                    "1) 结论摘要 2) 关键依据 3) 风险与待验证项 4) 建议下一步。"
                    "若前序结果存在冲突，必须明确指出。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：{_get_user_query(state)}\n\n"
                    f"各 Agent 结果：\n"
                    f"{json.dumps(tool_results, ensure_ascii=False, indent=2)}"
                )
            ),
        ]
    )

    return {
        "final_answer": response.content,
        "current_agent": "reviewer",
        "messages": [AIMessage(content=response.content)],
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def route_to_worker(state: AgentState) -> str:
    """根据 current_task_index 路由到对应 Agent 或 Reviewer。"""
    idx = state["current_task_index"]
    plan = state.get("task_plan") or []

    if idx >= len(plan):
        return "reviewer"

    agent = plan[idx]["agent"]
    if agent not in SUPPORTED_AGENTS:
        return "researcher"
    return agent


def route_after_worker(state: AgentState) -> str:
    """Worker 完成后：若还有任务则继续，否则进入 Reviewer。"""
    next_index = state["current_task_index"] + 1
    plan = state.get("task_plan") or []
    if next_index >= len(plan):
        return "reviewer"
    return "dispatch"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------


def get_orchestrator_graph(checkpointer=None):
    """
    构建并编译多 Agent 编排图。

    节点：
        planner → dispatch → {researcher|simulation|analyst|lab}
               → advance → dispatch → ... → reviewer → END
    """
    llm = _get_llm()

    researcher_agent = create_react_agent(
        model=llm,
        tools=get_tools_for_agent("researcher"),
        prompt=AGENT_PROMPTS["researcher"],
    )
    simulation_agent = create_react_agent(
        model=llm,
        tools=get_tools_for_agent("simulation"),
        prompt=AGENT_PROMPTS["simulation"],
    )
    analyst_agent = create_react_agent(
        model=llm,
        tools=get_tools_for_agent("analyst"),
        prompt=AGENT_PROMPTS["analyst"],
    )
    lab_agent = create_react_agent(
        model=llm,
        tools=get_tools_for_agent("lab"),
        prompt=AGENT_PROMPTS["lab"],
    )

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", _make_react_agent_node("researcher", researcher_agent))
    graph.add_node("simulation", _make_react_agent_node("simulation", simulation_agent))
    graph.add_node("analyst", _make_react_agent_node("analyst", analyst_agent))
    graph.add_node("lab", _make_react_agent_node("lab", lab_agent))
    graph.add_node("advance", advance_task_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_to_worker,
        {
            "researcher": "researcher",
            "simulation": "simulation",
            "analyst": "analyst",
            "lab": "lab",
            "reviewer": "reviewer",
        },
    )

    # dispatch 节点：advance 后重新路由（用 advance 的 conditional 实现）
    for worker in ("researcher", "simulation", "analyst", "lab"):
        graph.add_conditional_edges(
            worker,
            route_after_worker,
            {
                "reviewer": "reviewer",
                "dispatch": "advance",
            },
        )

    graph.add_conditional_edges(
        "advance",
        route_to_worker,
        {
            "researcher": "researcher",
            "simulation": "simulation",
            "analyst": "analyst",
            "lab": "lab",
            "reviewer": "reviewer",
        },
    )

    graph.add_edge("reviewer", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver

    graph = get_orchestrator_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-session"}}

    demo_query = "设计一种低介电常数封装材料，并评估其热稳定性与工艺可行性"
    print(f"用户问题：{demo_query}\n")
    print("=" * 60)

    result = graph.invoke(
        {"messages": [HumanMessage(content=demo_query)]},
        config=config,
    )

    print("\n【任务计划】")
    for task in result.get("task_plan", []):
        print(f"  [{task['status']}] {task['agent']}: {task['description']}")

    print("\n【最终答案】")
    print(result.get("final_answer") or result["messages"][-1].content)
