from fastapi import FastAPI
from pydantic import BaseModel
from my_agent import get_research_agent

# 1. 初始化 FastAPI 应用
app = FastAPI(title="AI Research Agent API")

# 2. 获取 Agent 实例
agent = get_research_agent()


# 3. 定义请求体模型
class QueryRequest(BaseModel):
    query: str


# 4. 定义 API 接口
@app.post("/chat")
async def chat(request: QueryRequest):
    # 调用 agent 执行任务
    response = agent.invoke({"messages": request.query})

    # 提取最后一条消息的内容
    # 注意：根据你之前的打印结果，响应结构比较复杂
    # 我们提取最后一条 AIMessage 的 content
    result_text = response['messages'][-1].content

    return {"reply": result_text}

# 运行命令: uvicorn api_server:app --reload