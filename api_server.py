import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from orchestrator import get_orchestrator_graph

app = FastAPI(title="AI4Material Multi-Agent API")

memory = MemorySaver()
graph = get_orchestrator_graph(checkpointer=memory)

# 需要向前端流式输出的编排节点
STREAM_NODES = ("planner", "researcher", "simulation", "analyst", "lab", "reviewer")


class ChatRequest(BaseModel):
    session_id: str
    query: str


def _extract_message_content(node_output: dict) -> str | None:
    """从节点更新中提取最新 AI 消息内容。"""
    messages = node_output.get("messages")
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, AIMessage) and last.content:
        return last.content
    content = getattr(last, "content", None)
    return content if content else None


async def generate_response(session_id: str, query: str):
    """多 Agent 编排流式响应，按节点输出 NDJSON 事件。"""
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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口（多 Agent 编排）。"""
    return StreamingResponse(
        generate_response(request.session_id, request.query),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
