"""知识库入库脚本。

用法:
    python scripts/ingest_knowledge.py           # 增量/首次入库
    python scripts/ingest_knowledge.py --rebuild # 清空并重建向量库

支持格式: .md / .txt / .pdf（PDF 按页入库）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import (  # noqa: E402
    KNOWLEDGE_DIR,
    PERSIST_DIR,
    ingest_knowledge,
    load_source_documents,
    reset_retriever_cache,
    split_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建/重建 Chroma 知识库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除现有 chroma_db 后全量重建",
    )
    args = parser.parse_args()

    source_docs = load_source_documents()
    if not source_docs:
        raise SystemExit(f"未找到知识文件，请先在 {KNOWLEDGE_DIR} 目录添加文档。")

    chunks = split_documents(source_docs)
    print(f"加载文档: {len(source_docs)} 个")
    print(f"分块数量: {len(chunks)} 个")

    ingest_knowledge(rebuild=args.rebuild)
    reset_retriever_cache()

    action = "重建" if args.rebuild else "入库"
    print(f"{action}完成，向量库路径: {PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    main()
