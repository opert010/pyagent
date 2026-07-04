# 建议先运行：pip install -U langchain-text-splitters langchain-core langchain-chroma rank_bm25
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from rag import search_knowledge_base

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

RESEARCH_AGENT_PROMPT = (
    "你是集成电路材料研发助手，擅长封装材料、工艺与物性分析。"
    "回答材料、工艺、模拟、表征相关问题前，必须先调用 search_knowledge_base 检索知识库。"
    "回答时需引用检索结果中的来源，不得编造物性参数或工艺条件。"
)


def get_research_agent(checkpointer=None):
    llm = ChatOpenAI(
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=ZHIPU_API_KEY,
    )

    return create_react_agent(
        model=llm,
        tools=[search_knowledge_base],
        checkpointer=checkpointer,
        prompt=RESEARCH_AGENT_PROMPT,
    )
