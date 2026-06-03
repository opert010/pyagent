from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver  # 核心：内存检查点
from my_agent import get_research_agent

app = FastAPI(title="AI Research Agent API with Memory")

# 1. 创建内存检查点（在生产中可替换为 RedisSaver 持久化到 Redis）
memory = MemorySaver()

# 2. 修改 Agent 获取逻辑，挂载检查点
# 假设你在 get_research_agent 中接受 checkpointer 参数
agent = get_research_agent(checkpointer=memory)


class ChatRequest(BaseModel):
    session_id: str  # 每个用户/会话一个唯一的 ID
    query: str


@app.post("/chat")
async def chat(request: ChatRequest):
    # 配置 config，将 session_id 绑定到执行任务中
    config = {"configurable": {"thread_id": request.session_id}}

    # 携带 config 调用
    response = agent.invoke({"messages": request.query}, config=config)

    result_text = response['messages'][-1].content
    return {"reply": result_text}