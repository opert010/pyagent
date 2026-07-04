"""Tool Registry：集中注册工具并按 Agent 分配白名单。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from tools.analyst_tools import analyze_numeric_data, compare_material_properties
from tools.knowledge_tools import search_knowledge_base
from tools.lab_tools import search_process_sop, write_experiment_record
from tools.simulation_tools import (
    get_simulation_job_status,
    get_vasp_incar_template,
    submit_vasp_job,
)

AgentName = Literal["researcher", "simulation", "analyst", "lab"]
ToolCategory = Literal["knowledge", "simulation", "analysis", "lab"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    agents: tuple[AgentName, ...]
    handler: Callable
    description: str


def _spec(
    name: str,
    category: ToolCategory,
    agents: tuple[AgentName, ...],
    handler: Callable,
    description: str,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        category=category,
        agents=agents,
        handler=handler,
        description=description,
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "search_knowledge_base": _spec(
        "search_knowledge_base",
        "knowledge",
        ("researcher",),
        search_knowledge_base,
        "检索材料研发知识库，返回带引用来源的文本片段",
    ),
    "submit_vasp_job": _spec(
        "submit_vasp_job",
        "simulation",
        ("simulation",),
        submit_vasp_job,
        "提交 VASP 模拟任务（Mock），返回 job_id 和建议 INCAR",
    ),
    "get_vasp_incar_template": _spec(
        "get_vasp_incar_template",
        "simulation",
        ("simulation",),
        get_vasp_incar_template,
        "获取 VASP INCAR 参数模板（dielectric/relax/static）",
    ),
    "get_simulation_job_status": _spec(
        "get_simulation_job_status",
        "simulation",
        ("simulation",),
        get_simulation_job_status,
        "查询 VASP 模拟任务状态（Mock）",
    ),
    "compare_material_properties": _spec(
        "compare_material_properties",
        "analysis",
        ("analyst",),
        compare_material_properties,
        "对比多种封装材料的物性参数（k、Td 等）",
    ),
    "analyze_numeric_data": _spec(
        "analyze_numeric_data",
        "analysis",
        ("analyst",),
        analyze_numeric_data,
        "对数值序列做统计分析（均值、标准差等）",
    ),
    "search_process_sop": _spec(
        "search_process_sop",
        "lab",
        ("lab",),
        search_process_sop,
        "检索实验/封装工艺 SOP 与表征方法",
    ),
    "write_experiment_record": _spec(
        "write_experiment_record",
        "lab",
        ("lab",),
        write_experiment_record,
        "写入实验记录到 ELN（Mock）",
    ),
}

AGENT_TOOL_NAMES: dict[AgentName, tuple[str, ...]] = {
    "researcher": ("search_knowledge_base",),
    "simulation": (
        "submit_vasp_job",
        "get_vasp_incar_template",
        "get_simulation_job_status",
    ),
    "analyst": ("compare_material_properties", "analyze_numeric_data"),
    "lab": ("search_process_sop", "write_experiment_record"),
}

AGENT_PROMPTS: dict[AgentName, str] = {
    "researcher": (
        "你是材料科学知识检索专家，负责封装材料、工艺、模拟与表征知识检索。"
        "回答前必须调用 search_knowledge_base，并在回答中引用来源编号（如 [1]）。"
        "若知识库无相关内容，明确说明未检索到，不得编造参数。"
    ),
    "simulation": (
        "你是 computational materials 模拟专家，擅长 VASP 第一性原理计算。"
        "设计模拟方案时，先调用 get_vasp_incar_template 获取参数模板，"
        "需要提交任务时使用 submit_vasp_job，并用 get_simulation_job_status 跟踪状态。"
        "当前为 Mock 环境，需在结论中说明后续应对接真实 HPC。"
    ),
    "analyst": (
        "你是材料数据分析师，负责物性对比与数值统计。"
        "对比材料时使用 compare_material_properties，分析数值序列时使用 analyze_numeric_data。"
        "输出需包含表格化对比、统计摘要与不确定性说明。"
    ),
    "lab": (
        "你是实验室工艺专家，负责合成路线、表征方法与实验记录。"
        "制定实验方案前必须调用 search_process_sop 检索工艺与 SOP；"
        "方案确定后可调用 write_experiment_record 写入 ELN 记录（Mock）。"
        "输出可执行步骤、关键参数与安全注意事项。"
    ),
}


def get_tool(name: str) -> Callable:
    """按名称获取工具 handler。"""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"未知工具: {name}")
    return TOOL_REGISTRY[name].handler


def get_tools_for_agent(agent: AgentName) -> list[Callable]:
    """获取指定 Agent 的工具白名单（handler 列表）。"""
    names = AGENT_TOOL_NAMES.get(agent, ())
    missing = [name for name in names if name not in TOOL_REGISTRY]
    if missing:
        raise KeyError(f"Agent '{agent}' 引用了未注册工具: {missing}")
    return [TOOL_REGISTRY[name].handler for name in names]


def list_tool_catalog() -> list[dict[str, str]]:
    """列出工具目录，供调试或 API 展示。"""
    catalog: list[dict[str, str]] = []
    for spec in TOOL_REGISTRY.values():
        catalog.append(
            {
                "name": spec.name,
                "category": spec.category,
                "agents": ",".join(spec.agents),
                "description": spec.description,
            }
        )
    return catalog
