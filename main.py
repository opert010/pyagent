import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage

from my_agent import get_research_agent

DEMO_QUERY = "低介电常数封装材料有哪些？请结合知识库说明并引用来源。"


def main():
    print("系统初始化中...")
    agent = get_research_agent()

    print("开始执行任务...")
    print(f"问题: {DEMO_QUERY}\n")

    response = agent.invoke({"messages": [HumanMessage(content=DEMO_QUERY)]})
    print("Agent 回复:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
