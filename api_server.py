import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from my_agent import get_research_agent
from langgraph.checkpoint.memory import MemorySaver

app = FastAPI()

# 初始化内存检查点
memory = MemorySaver()
# 初始化 Agent
agent = get_research_agent(checkpointer=memory)


class ChatRequest(BaseModel):
    session_id: str
    query: str


async def generate_response(session_id: str, query: str):
    """
    异步生成器，用于处理流式输出
    """
    config = {"configurable": {"thread_id": session_id}}

    # 使用 astream 逐块获取响应
    # 注意：根据你的具体 LangGraph 结构，chunk 的获取方式可能略有不同
    async for event in agent.astream({"messages": [("user", query)]}, config=config):
        # LangGraph 可能会返回不同类型的节点更新，这里我们只提取 AIMessage 的内容
        if "agent" in event:
            content = event["agent"]["messages"][-1].content
            if content:
                # 为了让前端易于处理，我们将内容包装为 JSON 字符串块
                yield json.dumps({"token": content}, ensure_ascii=False) + "\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口
    """
    return StreamingResponse(
        generate_response(request.session_id, request.query),
        media_type="application/x-ndjson"  # 使用 ndjson 格式，前端更易于按行读取
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)