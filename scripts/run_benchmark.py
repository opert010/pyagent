"""Benchmark 批量评估脚本。

用法:
    python scripts/run_benchmark.py              # RAG 关键词覆盖率（默认，不调用 LLM）
    python scripts/run_benchmark.py --dry-run    # 仅校验 benchmark 文件格式
    python scripts/run_benchmark.py --full       # 完整多 Agent 编排（需 ZHIPU_API_KEY，耗时较长）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

BENCHMARK_FILE = ROOT / "benchmarks" / "material_queries.json"


def load_benchmarks() -> list[dict]:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise ValueError("benchmark 文件中没有 queries")
    for item in queries:
        assert "id" in item and "query" in item, f"缺少 id/query: {item}"
        assert "rag_keywords" in item, f"缺少 rag_keywords: {item.get('id')}"
    return queries


def eval_rag_query(query: str, keywords: list[str]) -> dict:
    from rag import search_knowledge_base

    result = search_knowledge_base(query)
    has_source = "来源:" in result
    matched = [kw for kw in keywords if kw.lower() in result.lower()]
    coverage = len(matched) / len(keywords) if keywords else 1.0
    return {
        "has_source": has_source,
        "matched_keywords": matched,
        "coverage": coverage,
        "passed": has_source and coverage >= 0.5,
        "result_preview": result[:200] + ("..." if len(result) > 200 else ""),
    }


def run_rag_benchmark(queries: list[dict]) -> list[dict]:
    results = []
    for item in queries:
        eval_result = eval_rag_query(item["query"], item["rag_keywords"])
        results.append({"id": item["id"], "query": item["query"], **eval_result})
        status = "PASS" if eval_result["passed"] else "FAIL"
        kw_info = ", ".join(eval_result["matched_keywords"]) or "无"
        print(f"[{status}] {item['id']}: 来源={eval_result['has_source']}, 关键词={kw_info}")
    return results


def run_full_benchmark(queries: list[dict], limit: int | None = None) -> list[dict]:
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from orchestrator import get_orchestrator_graph

    load_dotenv()
    if not os.getenv("ZHIPU_API_KEY"):
        raise SystemExit("完整 benchmark 需要配置 ZHIPU_API_KEY")

    graph = get_orchestrator_graph(checkpointer=MemorySaver())
    results = []
    subset = queries[:limit] if limit else queries

    for index, item in enumerate(subset, start=1):
        session_id = f"benchmark-{item['id']}"
        config = {"configurable": {"thread_id": session_id}}
        print(f"\n[{index}/{len(subset)}] {item['id']}: {item['query']}")
        start = time.time()

        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=item["query"])]},
                config=config,
            )
            elapsed = time.time() - start
            agents_used = {t["agent"] for t in state.get("task_plan", [])}
            expected = set(item.get("expected_agents", []))
            agent_overlap = bool(agents_used & expected) if expected else True
            has_answer = bool(state.get("final_answer") or state.get("messages"))

            passed = has_answer and agent_overlap
            results.append(
                {
                    "id": item["id"],
                    "passed": passed,
                    "elapsed_s": round(elapsed, 1),
                    "agents_used": sorted(agents_used),
                    "expected_agents": sorted(expected),
                }
            )
            print(f"  -> {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s) agents={sorted(agents_used)}")
        except Exception as exc:
            results.append({"id": item["id"], "passed": False, "error": str(exc)})
            print(f"  -> FAIL: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="材料研发 benchmark 评估")
    parser.add_argument("--dry-run", action="store_true", help="仅校验 benchmark 文件")
    parser.add_argument("--full", action="store_true", help="完整多 Agent 编排（慢，需 API Key）")
    parser.add_argument("--limit", type=int, default=None, help="限制 full 模式问题数量")
    args = parser.parse_args()

    queries = load_benchmarks()
    print(f"加载 benchmark: {len(queries)} 个问题")

    if args.dry_run:
        print("[PASS] benchmark 文件格式校验通过")
        sys.exit(0)

    if args.full:
        results = run_full_benchmark(queries, limit=args.limit)
    else:
        results = run_rag_benchmark(queries)

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"\n结果: {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
