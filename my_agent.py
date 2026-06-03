import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 修改前：from langchain_community.vectorstores import Chroma
# 修改后：
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter  # 注意这里改动了
from langchain_core.documents import Document              # 注意这里改动了
from deepagents import create_deep_agent
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 全局变量：保持向量库持久化在本地
PERSIST_DIR = "./chroma_db"


def get_retriever():
    # 显式获取 Key
    zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    if not zhipu_api_key:
        raise ValueError("请确保环境变量 ZHIPU_API_KEY 已配置")

    # 配置 Embeddings
    # 注意：如果使用智谱 Embedding，base_url 必须指向智谱的嵌入式接口地址
    embeddings = OpenAIEmbeddings(
        model="embedding-2",  # 智谱的向量模型名称
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=zhipu_api_key
    )

    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings).as_retriever()

    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,  # 使用配置好的 embedding
        persist_directory=PERSIST_DIR
    )
    return vectorstore.as_retriever()


def search_knowledge_base(query: str):
    """当用户询问技术细节时，调用此工具查询知识库"""
    retriever = get_retriever()
    # 替换前：results = retriever.get_relevant_documents(query)
    # 替换后：使用 invoke 方法，这是 LangChain 现在的统一调用标准
    results = retriever.invoke(query)

    # 打印结果看看（调试用）
    # print(f"检索到 {len(results)} 条相关文档")

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