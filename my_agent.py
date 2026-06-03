import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter  # 注意这里改动了
from langchain_core.documents import Document              # 注意这里改动了
from deepagents import create_deep_agent
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 全局变量：保持向量库持久化在本地
PERSIST_DIR = "./chroma_db"


def get_retriever():
    # 1. 如果已经存在数据库目录，直接加载，不重复创建
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=OpenAIEmbeddings()).as_retriever()

    # 2. 如果不存在，则读取文件并创建（只在第一次运行）
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    # 将向量数据持久化到本地文件夹
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR
    )
    return vectorstore.as_retriever()


def search_knowledge_base(query: str):
    """当用户询问技术细节时，调用此工具查询知识库"""
    retriever = get_retriever()
    results = retriever.get_relevant_documents(query)
    # 将检索到的内容拼接成字符串返回给 Agent
    return "\n".join([doc.page_content for doc in results])


def get_research_agent():
    llm = ChatOpenAI(
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=os.getenv("ZHIPU_API_KEY")
    )

    return create_deep_agent(
        model=llm,
        tools=[search_knowledge_base],
        system_prompt="你是研发助理。遇到不清楚的技术概念，请务必先调用 search_knowledge_base 工具查询知识库。"
    )