
---

# pyagent: 企业级 AI 智能体工程化底座

`pyagent` 是一个基于 **LangGraph** 生态构建的 AI 工程化落地原型（PoC），旨在解决大语言模型（LLM）在复杂业务场景中落地难、状态难管理、服务难集成的问题。

该项目将 AI 推理能力封装为标准的 **FastAPI 微服务**，为后端工程师提供了一套可复用的 AI 工程化模版。

## 🚀 核心架构设计

项目采用“模块化+组件化”设计，具备以下核心能力：

* **智能调度引擎**：基于 `LangGraph` 构建状态机，实现复杂的 ReAct 推理逻辑。
* **企业级知识检索（RAG）**：集成 `Chroma` 向量数据库，支持对技术文档、业务规则的精准语义匹配。
* **会话状态持久化**：内置 `Checkpointer` 机制，支持多轮对话的上下文记忆与状态管理。
* **标准化服务化**：通过 `FastAPI` 封装 API 接口，原生支持异步 IO，满足高并发集成需求。

## 🛠 技术栈

* **框架**: `LangGraph` (状态编排), `LangChain` (组件集成), `FastAPI` (接口服务)
* **向量库**: `Chroma` (向量存储与检索)
* **大模型**: `GLM-4` (基于智谱开放平台 API)
* **部署**: `Docker`/`Uvicorn`

## 📂 项目结构

```text
pyagent/
├── app/
│   ├── main.py          # FastAPI 服务入口
│   ├── my_agent.py      # Agent 核心逻辑 (ReAct 调度)
│   ├── database.py      # Chroma 向量库操作
│   └── utils/           # 工具函数与配置管理
├── .env                 # 环境变量 (API Key 配置)
├── requirements.txt     # 依赖清单
└── Dockerfile           # 容器化部署配置

```

## ⚡ 快速开始

1. **克隆项目**:
```bash

```



git clone https://github.com/opert010/pyagent.git

cd pyagent

```

2.  **配置环境**:
    在根目录创建 `.env` 文件，填入你的 API Key:
    ```text
ZHIPU_API_KEY=your_actual_api_key

```

3. **安装依赖**:
```bash

```



pip install -r requirements.txt

```

4.  **启动服务**:
    ```bash
uvicorn app.main:app --reload

```

5. **交互测试**:
访问 `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` 查看 Swagger 调试界面，即可通过接口发送查询。

## 📈 设计思考（Engineering Insights）

本项目不仅是一个练手项目，更是对生产环境 AI 工程化落地的一次深度思考：

* **解耦设计**：Agent 逻辑与业务层通过标准 API 隔离，支持平滑迁移至云端。
* **状态可控**：通过 `Checkpointer` 确保 Agent 执行链路可追溯，降低 AI 不确定性带来的风险。
* **高可用考量**：采用异步架构，为后续接入 Redis 持久化存储和分布式任务调度预留了接口。

## 🤝 贡献说明

本框架正在持续迭代中。若您有关于 **Agent 工作流优化** 或 **AI 生产环境部署** 的想法，欢迎提交 Issue 或 Pull Request。

---
