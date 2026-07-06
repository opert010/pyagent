import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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

STATIC_DIR = Path(__file__).resolve().parent / "static"

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


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    task_plan: list[dict]
    tool_results: dict
    final_answer: str
    current_agent: str


def _serialize_message(msg) -> dict:
    """将 LangChain 消息或元组格式化为 JSON 可序列化结构。"""
    if isinstance(msg, tuple) and len(msg) == 2:
        role, content = msg
        return {"role": str(role), "content": content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, BaseMessage):
        return {"role": msg.type, "content": msg.content}
    return {"role": "unknown", "content": str(msg)}


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


@app.get("/")
async def root():
    """重定向到 Web Demo。"""
    return RedirectResponse(url="/demo/")


@app.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """获取会话完整状态（消息、任务计划、工具结果）。"""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在或尚无记录")

    values = snapshot.values
    messages = values.get("messages") or []
    if not messages and not values.get("final_answer"):
        raise HTTPException(status_code=404, detail="会话不存在或尚无记录")

    return SessionHistoryResponse(
        session_id=session_id,
        messages=[_serialize_message(m) for m in messages],
        task_plan=values.get("task_plan") or [],
        tool_results=values.get("tool_results") or {},
        final_answer=values.get("final_answer") or "",
        current_agent=values.get("current_agent") or "",
    )


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


if STATIC_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(STATIC_DIR), html=True), name="demo")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
