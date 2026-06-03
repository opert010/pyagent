import sys
import os
# 获取当前脚本所在目录并添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# main.py
from my_agent import get_research_agent


def main():
    print("系统初始化中...")
    agent = get_research_agent()

    # 执行业务流程
    print("开始执行任务...")
    response = agent.invoke({"messages": "分析一下如何将 Java 的 Spring Boot 架构迁移到 Python FastAPI"})
    print(f"Agent 回复: {response}")


if __name__ == "__main__":
    main()