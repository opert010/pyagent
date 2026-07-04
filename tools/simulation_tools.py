"""Simulation Agent 工具（VASP Mock，可后续对接 HPC/Slurm）。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

# 内存任务队列（Mock）
_JOB_STORE: dict[str, dict] = {}

INCAR_TEMPLATES = {
    "dielectric": """# DFPT 介电常数计算
SYSTEM = Dielectric constant (DFPT)
IBRION = 8
LEPSILON = .TRUE.
LPEAD = .TRUE.
ENCUT = 520
EDIFF = 1E-6
NSW = 0
""",
    "relax": """# 结构优化
SYSTEM = Structure relaxation
ISIF = 3
IBRION = 2
NSW = 100
ENCUT = 520
EDIFF = 1E-5
""",
    "static": """# 高精度静态自洽
SYSTEM = Static SCF
IBRION = -1
NSW = 0
ENCUT = 520
EDIFF = 1E-8
""",
}


def get_vasp_incar_template(calculation_type: str) -> str:
    """获取 VASP INCAR 参数模板。

    Args:
        calculation_type: 计算类型，可选 dielectric（介电常数）、relax（结构优化）、static（静态自洽）
    """
    key = calculation_type.strip().lower()
    if key not in INCAR_TEMPLATES:
        supported = ", ".join(INCAR_TEMPLATES.keys())
        return f"未知计算类型: {calculation_type}。支持的类型: {supported}"

    return f"【{key} 模板】\n{INCAR_TEMPLATES[key]}"


def submit_vasp_job(material: str, calculation_type: str = "dielectric") -> str:
    """提交 VASP 模拟任务（Mock）。

    Args:
        material: 材料名称或化学式，如 SiO2、BCB
        calculation_type: 计算类型，默认 dielectric
    """
    key = calculation_type.strip().lower()
    if key not in INCAR_TEMPLATES:
        return get_vasp_incar_template(calculation_type)

    job_id = f"vasp-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    _JOB_STORE[job_id] = {
        "job_id": job_id,
        "material": material,
        "calculation_type": key,
        "status": "queued",
        "submitted_at": now,
        "note": "Mock 任务，尚未对接真实 HPC 集群",
    }

    return json.dumps(
        {
            "job_id": job_id,
            "material": material,
            "calculation_type": key,
            "status": "queued",
            "message": "任务已提交（Mock）。可使用 get_simulation_job_status 查询状态。",
            "suggested_incar": INCAR_TEMPLATES[key],
        },
        ensure_ascii=False,
        indent=2,
    )


def get_simulation_job_status(job_id: str) -> str:
    """查询 VASP 模拟任务状态（Mock）。

    Args:
        job_id: submit_vasp_job 返回的任务 ID
    """
    job = _JOB_STORE.get(job_id.strip())
    if not job:
        return f"未找到任务: {job_id}。请确认 job_id 是否正确。"

    # Mock：查询时自动推进为 running
    if job["status"] == "queued":
        job["status"] = "running"

    return json.dumps(job, ensure_ascii=False, indent=2)
