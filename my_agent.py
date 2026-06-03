# 建议先运行：pip install -U langchain-text-splitters langchain-core langchain-chroma
import os
from dotenv import load_dotenv

# 1. 大模型与 Embedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2. 向量库 (最新规范)
from langchain_chroma import Chroma

# 3. 文本处理 (必须从新包导入)
from langchain_text_splitters import CharacterTextSplitter

# 4. 核心模式 (从 core 导入)
from langchain_core.documents import Document

# 5. Agent 构建
from langgraph.prebuilt import create_react_agent

# 加载配置
load_dotenv()

# 常量配置
PERSIST_DIR = "./chroma_db"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")


# 1. 知识库检索模块
def get_retriever():
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=ZHIPU_API_KEY
    )

    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings).as_retriever()

    # 第一次初始化向量库
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectorstore.as_retriever()


# 2. 工具函数
def search_knowledge_base(query: str):
    """当用户询问技术细节时，调用此工具查询知识库"""
    retriever = get_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])


# 3. Agent 构建工厂函数
def get_research_agent(checkpointer=None):
    llm = ChatOpenAI(
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=ZHIPU_API_KEY
    )

    # 尝试将 state_modifier 改为 prompt 或直接删除该参数进行测试
    # 如果你的版本不支持 state_modifier，可以直接使用 message_modifier
    return create_react_agent(
        model=llm,
        tools=[search_knowledge_base],
        checkpointer=checkpointer,
        prompt="你是一个资深的研发助理，负责辅助进行技术架构迁移分析。在回答问题前，务必优先使用 search_knowledge_base 查询知识库。"
    )