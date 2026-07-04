# pyagent

面向**集成电路材料研发**场景的 AI 智能体项目（AI4Material PoC）。基于 LangGraph 多 Agent 编排、RAG 知识检索与 FastAPI 服务化，支持材料知识查询、模拟方案、物性分析与实验工艺检索。

## 核心能力

- **多 Agent 编排**：Planner → Researcher / Simulation / Analyst / Lab → Reviewer
- **RAG 知识检索**：Chroma 向量库 + BM25 混合检索，支持来源引用
- **工具调用**：VASP 模拟（Mock）、物性对比、数值统计、工艺 SOP 检索
- **流式 API**：NDJSON 格式输出各 Agent 节点结果
- **知识库管理**：支持文档上传与向量库重建（`.md` / `.txt` / `.pdf`）
- **会话状态**：LangGraph Checkpointer 支持多轮对话

## 技术栈

| 类别 | 选型 |
|------|------|
| Agent 编排 | LangGraph、LangChain |
| Web 服务 | FastAPI、Uvicorn |
| 向量库 | Chroma |
| 大模型 | 智谱 GLM-4（OpenAI 兼容 API） |
| Embedding | 智谱 embedding-2 |

## 项目结构

```text
pyagent/
├── api_server.py          # FastAPI 服务（多 Agent 流式接口 + 知识库 API）
├── orchestrator.py        # LangGraph 多 Agent 编排骨架
├── my_agent.py            # 单 Agent 工厂（ReAct + RAG）
├── rag.py                 # 知识库加载、分块、混合检索
├── main.py                # 命令行单 Agent 测试入口
├── knowledge/             # 材料领域知识文档
│   ├── materials/         # 材料物性
│   ├── process/           # 工艺 SOP
│   ├── simulation/        # 模拟计算
│   ├── lab/               # 表征方法
│   └── faq/               # 常见问题
├── tools/                 # Agent 工具层
│   ├── simulation_tools.py
│   ├── analyst_tools.py
│   └── lab_tools.py
├── scripts/
│   ├── ingest_knowledge.py    # 知识库入库
│   └── verify_baseline.py     # 基线验证
├── requirements.txt
├── .env                   # 环境变量（需自行创建，不入库）
└── chroma_db/             # 向量库（运行时生成，不入库）
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
| GET | `/health` | 健康检查 |
| POST | `/chat/stream` | 多 Agent 流式对话（NDJSON） |
| POST | `/knowledge/rebuild` | 全量重建向量库 |
| POST | `/knowledge/upload` | 上传知识文档 |

### 流式对话示例

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

- `file`：`.md` / `.txt` / `.pdf` 文件
- `category`：分类目录，如 `materials`、`process`
- `rebuild`：是否上传后自动重建索引（默认 `true`）

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

| Agent | 工具 |
|-------|------|
| Researcher | `search_knowledge_base`（RAG 检索） |
| Simulation | `submit_vasp_job`、`get_vasp_incar_template`、`get_simulation_job_status` |
| Analyst | `compare_material_properties`、`analyze_numeric_data` |
| Lab | `search_process_sop` |

## 知识库维护

将文档放入 `knowledge/` 对应子目录后重建：

```bash
python scripts/ingest_knowledge.py --rebuild
```

支持格式：`.md`、`.txt`、`.pdf`（PDF 按页入库，检索结果含页码）。

更新知识库后，若 API 服务已在运行，调用 `POST /knowledge/rebuild` 或重启服务以刷新检索缓存。

## 常见问题

**ModuleNotFoundError**

请确认已激活虚拟环境，并使用项目内的 Python：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/verify_baseline.py
```

**检索结果为空**

删除 `chroma_db/` 后重新入库：

```bash
python scripts/ingest_knowledge.py --rebuild
```

**Windows 下重建向量库失败**

服务运行中可能占用向量库文件。可先调用 `POST /knowledge/rebuild`（已处理文件占用），或停止服务后再执行入库脚本。

## 后续规划

- Simulation 对接真实 HPC / Slurm 集群
- Checkpointer 持久化（Redis / PostgreSQL）
- ELN / LIMS 实验记录对接
- 前端流式对话 UI

## 许可证

本项目为工程化 PoC，持续迭代中。欢迎提交 Issue 或 Pull Request。
