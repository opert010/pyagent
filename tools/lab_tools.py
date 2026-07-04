"""Lab Agent 工具：工艺 SOP 检索与 ELN 实验记录（Mock）。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from tools.knowledge_tools import search_knowledge_base

# ELN 实验记录 Mock 存储
_ELN_STORE: dict[str, dict] = {}


def search_process_sop(keyword: str) -> str:
    """检索实验/封装工艺 SOP 与表征方法。

    Args:
        keyword: 检索关键词，如 "BCB 固化"、"TGA 测试"、"ELN"
    """
    query = f"工艺 SOP 实验流程 {keyword}"
    return search_knowledge_base(query)


def write_experiment_record(title: str, steps: str, notes: str = "") -> str:
    """写入实验记录到 ELN（Mock）。

    Args:
        title: 实验标题，如 "BCB 旋涂固化验证"
        steps: 实验步骤摘要
        notes: 可选备注（安全事项、异常记录等）
    """
    record_id = f"eln-{uuid.uuid4().hex[:8]}"
    record = {
        "record_id": record_id,
        "title": title.strip(),
        "steps": steps.strip(),
        "notes": notes.strip(),
        "status": "draft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Mock ELN 记录，尚未对接真实 LIMS/ELN 系统",
    }
    _ELN_STORE[record_id] = record

    return json.dumps(record, ensure_ascii=False, indent=2)
