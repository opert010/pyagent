import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from orchestrator import get_orchestrator_graph
from rag import (
    KNOWLEDGE_DIR,
    SUPPORTED_EXTENSIONS,
    ingest_knowledge,
    load_source_documents,
    reset_retriever_cache,
    split_documents,
)
from tools.registry import list_tool_catalog

app = FastAPI(title="AI4Material Multi-Agent API")

memory = MemorySaver()
graph = get_orchestrator_graph(checkpointer=memory)

STREAM_NODES = ("planner", "researcher", "simulation", "analyst", "lab", "reviewer")


class ChatRequest(BaseModel):
    session_id: str
    query: str


class RebuildResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    message: str


class UploadResponse(BaseModel):
    status: str
    saved_path: str
    documents: int
    chunks: int
    rebuilt: bool
    message: str


def _extract_message_content(node_output: dict) -> str | None:
    messages = node_output.get("messages")
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, AIMessage) and last.content:
        return last.content
    content = getattr(last, "content", None)
    return content if content else None


def _safe_category(category: str) -> str:
    cleaned = "".join(c for c in category.strip() if c.isalnum() or c in "-_")
    return cleaned or "general"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="无效文件名")
    return name


async def generate_response(session_id: str, query: str):
    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [HumanMessage(content=query)]}

    async for event in graph.astream(input_state, config=config):
        for node_name, node_output in event.items():
            if node_name not in STREAM_NODES:
                continue

            content = _extract_message_content(node_output)
            if not content:
                continue

            payload = {
                "type": "final" if node_name == "reviewer" else "node",
                "node": node_name,
                "token": content,
            }
            yield json.dumps(payload, ensure_ascii=False) + "\n"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "AI4Material Multi-Agent API",
        "tools": list_tool_catalog(),
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口（多 Agent 编排）。"""
    return StreamingResponse(
        generate_response(request.session_id, request.query),
        media_type="application/x-ndjson",
    )


@app.post("/knowledge/rebuild", response_model=RebuildResponse)
async def rebuild_knowledge():
    """全量重建向量知识库。"""
    source_docs = load_source_documents()
    if not source_docs:
        raise HTTPException(status_code=400, detail="knowledge/ 目录下没有可用文档")

    chunks = split_documents(source_docs)
    ingest_knowledge(rebuild=True)
    reset_retriever_cache()

    return RebuildResponse(
        status="ok",
        documents=len(source_docs),
        chunks=len(chunks),
        message="知识库重建完成",
    )


@app.post("/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    category: str = Form(default="general"),
    rebuild: bool = Form(default=True),
):
    """上传知识文档到 knowledge/{category}/ 并可选重建向量库。"""
    filename = _safe_filename(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，仅支持: {supported}")

    category_dir = KNOWLEDGE_DIR / _safe_category(category)
    category_dir.mkdir(parents=True, exist_ok=True)
    dest = category_dir / filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_count = 0
    chunk_count = 0
    if rebuild:
        source_docs = load_source_documents()
        chunks = split_documents(source_docs)
        doc_count = len(source_docs)
        chunk_count = len(chunks)
        ingest_knowledge(rebuild=True)
        reset_retriever_cache()

    return UploadResponse(
        status="ok",
        saved_path=str(dest.as_posix()),
        documents=doc_count,
        chunks=chunk_count,
        rebuilt=rebuild,
        message="上传成功" + ("，知识库已重建" if rebuild else "，请调用 /knowledge/rebuild 重建索引"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
