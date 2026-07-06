"""项目基线验证脚本（不调用 LLM，适合快速自检）。

用法:
    python scripts/verify_baseline.py
    python scripts/verify_baseline.py --with-llm   # 额外测试 LLM 连通性
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

BENCHMARK_FILE = ROOT / "benchmarks" / "material_queries.json"


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name} -> {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="pyagent 基线验证")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="额外验证智谱 LLM API 连通性",
    )
    args = parser.parse_args()

    results: list[bool] = []

    def t_imports():
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langgraph.prebuilt import create_react_agent  # noqa: F401
        from rag import search_knowledge_base  # noqa: F401
        from orchestrator import get_orchestrator_graph  # noqa: F401
        from api_server import app  # noqa: F401

    def t_knowledge_files():
        knowledge_dir = ROOT / "knowledge"
        files = list(knowledge_dir.rglob("*"))
        supported = [f for f in files if f.suffix.lower() in {".md", ".txt", ".pdf"}]
        assert supported, "knowledge/ 下无可用文档"

    def t_rag_retrieval():
        from rag import search_knowledge_base

        result = search_knowledge_base("BCB 固化温度")
        assert "来源:" in result
        assert "BCB" in result or "350" in result

    def t_orchestrator_graph():
        from orchestrator import get_orchestrator_graph

        graph = get_orchestrator_graph()
        nodes = set(graph.get_graph().nodes.keys())
        assert "planner" in nodes and "reviewer" in nodes

    def t_llm():
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        load_dotenv()
        api_key = os.getenv("ZHIPU_API_KEY")
        assert api_key, "ZHIPU_API_KEY 未配置"

        llm = ChatOpenAI(
            model="glm-4",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=api_key,
            max_tokens=10,
        )
        resp = llm.invoke([HumanMessage(content="回复 OK")])
        assert resp.content.strip()

    def t_tools():
        from tools import compare_material_properties, submit_vasp_job, write_experiment_record
        from tools.registry import AGENT_TOOL_NAMES, get_tools_for_agent, list_tool_catalog

        result = compare_material_properties("SiO2,BCB", "k,Td_C")
        assert "SiO2" in result and "BCB" in result
        job = submit_vasp_job("SiO2", "dielectric")
        assert "job_id" in job
        eln = write_experiment_record("BCB 固化测试", "110C 软烘 -> 350C 硬固化")
        assert "record_id" in eln

        researcher_tools = get_tools_for_agent("researcher")
        assert len(researcher_tools) == len(AGENT_TOOL_NAMES["researcher"])
        assert len(list_tool_catalog()) >= 8

    def t_benchmark_file():
        data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        queries = data.get("queries", [])
        assert len(queries) >= 5, "benchmark 问题数量不足"
        for item in queries:
            assert item.get("id") and item.get("query") and item.get("rag_keywords")

    def t_rag_benchmark_eval():
        from rag import search_knowledge_base

        data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        passed = 0
        for item in data["queries"]:
            result = search_knowledge_base(item["query"])
            has_source = "来源:" in result
            keywords = item["rag_keywords"]
            matched = sum(1 for kw in keywords if kw.lower() in result.lower())
            coverage = matched / len(keywords) if keywords else 1.0
            if has_source and coverage >= 0.5:
                passed += 1
        assert passed >= len(data["queries"]) * 0.75, (
            f"RAG benchmark 通过率过低: {passed}/{len(data['queries'])}"
        )

    def t_demo_and_session_api():
        static_index = ROOT / "static" / "index.html"
        assert static_index.exists(), "前端 Demo 文件缺失"

        from api_server import app

        routes = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/sessions/{session_id}/history" in routes
        assert "/chat/stream" in routes

    results.append(check("依赖与模块导入", t_imports))
    results.append(check("知识库文件", t_knowledge_files))
    results.append(check("RAG 混合检索", t_rag_retrieval))
    results.append(check("多 Agent 编排图", t_orchestrator_graph))
    results.append(check("Tool Registry 与工具", t_tools))
    results.append(check("Benchmark 问题集", t_benchmark_file))
    results.append(check("RAG Benchmark 评估", t_rag_benchmark_eval))
    results.append(check("前端 Demo 与会话 API", t_demo_and_session_api))

    if args.with_llm:
        results.append(check("LLM API 连通", t_llm))

    passed = sum(results)
    total = len(results)
    print(f"\n结果: {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
