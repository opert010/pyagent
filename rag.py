"""知识库加载、入库与混合检索（RAG）模块。"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KNOWLEDGE_DIR = Path("./knowledge")
LEGACY_KNOWLEDGE_FILE = Path("./knowledge.txt")
PERSIST_DIR = Path("./chroma_db")
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
HYBRID_WEIGHTS = [0.5, 0.5]

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

_retriever_cache = None


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="embedding-2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=ZHIPU_API_KEY,
    )


def _tokenize_for_bm25(text: str) -> list[str]:
    """中英文混合分词，兼顾化学式与中文术语。"""
    return re.findall(r"[A-Za-z0-9_\-\.]+|[\u4e00-\u9fff]", text)


def _infer_category(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) > 1:
        return parts[0]
    return "general"


def _load_file_documents(file_path: Path, relative: str) -> list[Document]:
    """加载单个知识文件，PDF 按页展开为多个 Document。"""
    base_metadata = {
        "source": relative,
        "category": _infer_category(relative),
        "filename": file_path.name,
    }

    if file_path.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(file_path)).load()
        for doc in docs:
            doc.metadata.update(base_metadata)
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        return [doc for doc in docs if doc.page_content.strip()]

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    return [Document(page_content=text, metadata=base_metadata)]


def load_source_documents() -> list[Document]:
    """从 knowledge/ 目录及 legacy knowledge.txt 加载原始文档。"""
    documents: list[Document] = []

    if KNOWLEDGE_DIR.exists():
        for file_path in sorted(KNOWLEDGE_DIR.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = file_path.relative_to(KNOWLEDGE_DIR).as_posix()
            documents.extend(_load_file_documents(file_path, relative))

    if LEGACY_KNOWLEDGE_FILE.exists():
        legacy_text = LEGACY_KNOWLEDGE_FILE.read_text(encoding="utf-8")
        if legacy_text.strip():
            documents.append(
                Document(
                    page_content=legacy_text,
                    metadata={
                        "source": LEGACY_KNOWLEDGE_FILE.name,
                        "category": "legacy",
                        "filename": LEGACY_KNOWLEDGE_FILE.name,
                    },
                )
            )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """按段落结构分块，并保留元数据。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_knowledge(rebuild: bool = False) -> Chroma:
    """扫描知识目录，分块后写入 Chroma 向量库。"""
    reset_retriever_cache()

    source_docs = load_source_documents()
    if not source_docs:
        raise FileNotFoundError(
            f"未找到知识文件，请在 {KNOWLEDGE_DIR} 下添加 .md/.txt/.pdf 文档。"
        )

    chunks = split_documents(source_docs)
    embeddings = get_embeddings()

    if rebuild and PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        try:
            shutil.rmtree(PERSIST_DIR)
        except PermissionError:
            # Windows 下向量库文件可能被占用，改用 Chroma API 清空集合
            vectorstore = Chroma(
                persist_directory=str(PERSIST_DIR),
                embedding_function=embeddings,
            )
            try:
                vectorstore.delete_collection()
            except Exception:
                pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vectorstore


def get_vectorstore() -> Chroma:
    """获取向量库；不存在时自动入库。"""
    embeddings = get_embeddings()
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        return Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )

    return ingest_knowledge(rebuild=False)


def _documents_from_vectorstore(vectorstore: Chroma) -> list[Document]:
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    for content, metadata in zip(data["documents"], data["metadatas"]):
        if content:
            documents.append(Document(page_content=content, metadata=metadata or {}))
    return documents


def get_retriever():
    """混合检索器：向量语义检索 + BM25 关键词检索。"""
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    vectorstore = get_vectorstore()
    all_docs = _documents_from_vectorstore(vectorstore)
    if not all_docs:
        raise RuntimeError("向量库为空，请运行 python scripts/ingest_knowledge.py --rebuild")

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    bm25_retriever = BM25Retriever.from_documents(
        all_docs,
        preprocess_func=_tokenize_for_bm25,
    )
    bm25_retriever.k = TOP_K

    _retriever_cache = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=HYBRID_WEIGHTS,
    )
    return _retriever_cache


def reset_retriever_cache() -> None:
    """入库重建后刷新检索器缓存。"""
    global _retriever_cache
    _retriever_cache = None


def format_search_results(documents: list[Document]) -> str:
    """格式化检索结果，附带引用来源。"""
    if not documents:
        return "未检索到相关知识，请尝试更换关键词或补充知识库。"

    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        category = metadata.get("category", "general")
        page = metadata.get("page")
        page_info = f" | 页码: {page}" if page else ""
        lines.append(f"[{index}] 来源: {source} | 分类: {category}{page_info}")
        lines.append(doc.page_content.strip())
        lines.append("")
    return "\n".join(lines).strip()


def search_knowledge_base(query: str) -> str:
    """检索材料研发知识库，返回带引用来源的文本片段。"""
    retriever = get_retriever()
    results = retriever.invoke(query)
    return format_search_results(results)
