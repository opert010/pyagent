# my_agent.py
import os
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

# 1. 定义一个工具函数，Agent 可以调用它
def get_system_info(query: str):
    """当 Agent 需要获取系统运行时信息时调用此函数"""
    return f"当前系统运行在 Python 环境，已加载基础知识库。用户问题是: {query}"

# 2. 定义 Agent 的工厂函数
def get_research_agent():
    # 1. 显式初始化智谱的 Chat 模型
    # 智谱的 base_url 和 api_key 需要通过环境变量或参数传入
    llm = ChatOpenAI(
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=""  # 建议正式项目用 os.environ.get("ZHIPU_API_KEY")
    )

    # 2. 将 model 对象直接传给 Agent
    return create_deep_agent(
        model=llm,
        tools=[get_system_info],
        system_prompt="你是一个资深的 Java 转 Python 的 AI 研发助理。"
    )

# 3. 可以在这个文件里写简单的单元测试（类似 JUnit 的 main 方法）
if __name__ == "__main__":
    # 使用环境变量获取 API Key，避免硬编码
    agent = get_research_agent()
    result = agent.invoke({"messages": "请分析当前系统的基本情况"})
    print(result)