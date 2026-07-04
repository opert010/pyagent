# pyagent

面向**集成电路材料研发**场景的 AI 智能体项目（AI4Material PoC）。基于 LangGraph 多 Agent 编排、RAG 知识检索、Tool Registry 工具层与 FastAPI 服务化，支持材料知识查询、模拟方案、物性分析与实验工艺检索。

## 核心能力

- **多 Agent 编排**：Planner → Researcher / Simulation / Analyst / Lab → Reviewer
- **RAG 知识检索**：Chroma 向量库 + BM25 混合检索，支持来源引用
- **Tool Registry**：集中注册工具，按 Agent 白名单分配，便于扩展与对接外部系统
- **工具调用**：VASP 模拟（Mock）、物性对比、数值统计、工艺 SOP 检索、ELN 记录（Mock）
- **流式 API**：NDJSON 格式输出各 Agent 节点结果
- **知识库管理**：支持文档上传与向量库重建（`.md` / `.txt` / `.pdf`）
- **会话状态**：LangGraph Checkpointer 支持多轮对话（当前为内存模式）

## 技术栈

| 类别 | 选型 |
|------|------|
| Agent 编排 | LangGraph、LangChain |
| Web 服务 | FastAPI、Uvicorn |
| 向量库 | Chroma |
| 检索 | 向量语义检索 + BM25（rank_bm25） |
| 大模型 | 智谱 GLM-4（OpenAI 兼容 API） |
| Embedding | 智谱 embedding-2 |

## 项目结构

```text
pyagent/
├── api_server.py              # FastAPI 服务（流式对话 + 知识库 API）
├── orchestrator.py            # LangGraph 多 Agent 编排
├── my_agent.py                # 单 Agent 工厂（ReAct + RAG）
├── rag.py                     # 知识库加载、分块、混合检索
├── main.py                    # 命令行单 Agent 测试入口
├── knowledge/                 # 材料领域知识文档
│   ├── materials/             # 材料物性
│   ├── process/               # 工艺 SOP
│   ├── simulation/            # 模拟计算
│   ├── lab/                   # 表征方法
│   └── faq/                   # 常见问题
├── tools/                     # Agent 工具层
│   ├── registry.py            # Tool Registry（注册表 + Agent 白名单）
│   ├── knowledge_tools.py     # 知识库检索
│   ├── simulation_tools.py    # VASP 模拟（Mock）
│   ├── analyst_tools.py       # 物性对比、数值分析
│   └── lab_tools.py           # 工艺 SOP、ELN 记录（Mock）
├── scripts/
│   ├── ingest_knowledge.py    # 知识库入库
│   └── verify_baseline.py     # 基线验证
├── requirements.txt
├── .env                       # 环境变量（需自行创建，不入库）
└── chroma_db/                 # 向量库（运行时生成，不入库）
```

## 快速开始

### 1. 克隆与进入项目

```bash
git clone https://github.com/opert010/pyagent.git
cd pyagent
```

### 2. 创建虚拟环境（推荐）

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env`：

```text
ZHIPU_API_KEY=your_api_key_here
```

### 5. 构建知识库

```bash
python scripts/ingest_knowledge.py --rebuild
```

### 6. 基线验证

```bash
python scripts/verify_baseline.py
# 可选：额外验证 LLM API
python scripts/verify_baseline.py --with-llm
```

## 使用方式

### 命令行测试

```bash
# 单 Agent + RAG
python main.py

# 多 Agent 端到端（耗时较长，多次 LLM 调用）
python orchestrator.py
```

### 启动 API 服务

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

浏览器访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看 Swagger 文档。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含工具目录） |
| POST | `/chat/stream` | 多 Agent 流式对话（NDJSON） |
| POST | `/knowledge/rebuild` | 全量重建向量库 |
| POST | `/knowledge/upload` | 上传知识文档 |

### 健康检查与工具目录

```bash
GET /health
```

响应示例：

```json
{
  "status": "ok",
  "service": "AI4Material Multi-Agent API",
  "tools": [
    {
      "name": "search_knowledge_base",
      "category": "knowledge",
      "agents": "researcher",
      "description": "检索材料研发知识库，返回带引用来源的文本片段"
    }
  ]
}
```

### 流式对话

**请求：**

```json
POST /chat/stream
{
  "session_id": "session-001",
  "query": "设计一种低介电常数封装材料，并评估热稳定性"
}
```

**响应（NDJSON，每行一条）：**

```json
{"type": "node", "node": "planner", "token": "任务规划完成..."}
{"type": "node", "node": "researcher", "token": "【researcher】..."}
{"type": "final", "node": "reviewer", "token": "最终报告..."}
```

### 上传知识文档

在 Swagger 中调用 `POST /knowledge/upload`：

| 参数 | 说明 |
|------|------|
| `file` | `.md` / `.txt` / `.pdf` 文件 |
| `category` | 分类目录，如 `materials`、`process` |
| `rebuild` | 上传后是否自动重建索引（默认 `true`） |

## 多 Agent 工作流

```text
用户请求
   ↓
Planner（任务分解）
   ↓
Researcher / Simulation / Analyst / Lab（按子任务顺序执行）
   ↓
Reviewer（汇总审核）
   ↓
最终答案
```

## 工具层（Tool Registry）

工具通过 `tools/registry.py` 集中注册，`orchestrator.py` 与 `my_agent.py` 通过 `get_tools_for_agent()` 按 Agent 获取白名单工具。

| Agent | 工具 | 说明 |
|-------|------|------|
| Researcher | `search_knowledge_base` | RAG 知识库检索，返回引用来源 |
| Simulation | `submit_vasp_job` | 提交 VASP 任务（Mock） |
| | `get_vasp_incar_template` | 获取 INCAR 模板 |
| | `get_simulation_job_status` | 查询任务状态 |
| Analyst | `compare_material_properties` | 对比材料物性（k、Td 等） |
| | `analyze_numeric_data` | 数值序列统计分析 |
| Lab | `search_process_sop` | 检索工艺 / 表征 SOP |
| | `write_experiment_record` | 写入 ELN 实验记录（Mock） |

扩展新工具时：

1. 在 `tools/` 对应模块实现 handler（含清晰 docstring）
2. 在 `tools/registry.py` 的 `TOOL_REGISTRY` 中注册
3. 在 `AGENT_TOOL_NAMES` 中分配给目标 Agent

```python
from tools.registry import get_tools_for_agent, list_tool_catalog

tools = get_tools_for_agent("simulation")
catalog = list_tool_catalog()
```

## 知识库维护

将文档放入 `knowledge/` 对应子目录后重建：

```bash
python scripts/ingest_knowledge.py --rebuild
```

| 格式 | 说明 |
|------|------|
| `.md` / `.txt` | 按标题与段落分块 |
| `.pdf` | 按页入库，检索结果含页码 |

也可通过 API 上传：`POST /knowledge/upload`，或调用 `POST /knowledge/rebuild` 刷新索引。

## 常见问题

**ModuleNotFoundError**

请确认已激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/verify_baseline.py
```

**检索结果为空**

```bash
python scripts/ingest_knowledge.py --rebuild
```

**Windows 下重建向量库失败**

服务运行中可能占用向量库文件。优先调用 `POST /knowledge/rebuild`（已处理文件占用），或停止服务后再执行入库脚本。

**多 Agent 响应很慢**

正常现象。完整流程包含 Planner + 多个 Agent + Reviewer，通常需 2–5 分钟。调试时可先运行 `python main.py` 或 `python scripts/verify_baseline.py`。

## 后续规划

- 会话状态 API（`GET /sessions/{id}/state`）
- Checkpointer 持久化（Redis / PostgreSQL）
- Simulation 对接 Slurm / HPC 脚本模板
- ELN / LIMS 真实 API 对接
- 流式对话前端 Demo
- Docker 化部署

## 许可证

本项目为工程化 PoC，持续迭代中。欢迎提交 Issue 或 Pull Request。
