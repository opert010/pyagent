"""Analyst Agent 工具：材料物性对比与数值分析。"""

from __future__ import annotations

import json
import statistics

# 封装材料参考物性库（演示数据，可对接真实数据库）
MATERIAL_PROPERTIES: dict[str, dict[str, float | str]] = {
    "SiO2": {"k": 3.9, "Td_C": 1000, "type": "无机"},
    "PI": {"k": 3.0, "Td_C": 450, "type": "有机"},
    "BCB": {"k": 2.65, "Td_C": 450, "type": "有机"},
    "SiCOH": {"k": 2.2, "Td_C": 400, "type": "多孔无机"},
}


def compare_material_properties(materials: str, properties: str = "k,Td_C") -> str:
    """对比多种封装材料的物性参数。

    Args:
        materials: 逗号分隔的材料名，如 "SiO2,PI,BCB"
        properties: 逗号分隔的属性名，默认 "k,Td_C"（介电常数、热分解温度）
    """
    names = [m.strip() for m in materials.split(",") if m.strip()]
    props = [p.strip() for p in properties.split(",") if p.strip()]

    rows: list[dict] = []
    unknown: list[str] = []

    for name in names:
        key = next((k for k in MATERIAL_PROPERTIES if k.lower() == name.lower()), None)
        if not key:
            unknown.append(name)
            continue
        row = {"material": key, "type": MATERIAL_PROPERTIES[key].get("type", "")}
        for prop in props:
            row[prop] = MATERIAL_PROPERTIES[key].get(prop, "N/A")
        rows.append(row)

    result = {"comparison": rows, "properties": props}
    if unknown:
        result["unknown_materials"] = unknown
        result["available_materials"] = list(MATERIAL_PROPERTIES.keys())

    if not rows:
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 简要统计
    numeric_summary: dict[str, dict] = {}
    for prop in props:
        values = [r[prop] for r in rows if isinstance(r.get(prop), (int, float))]
        if values:
            numeric_summary[prop] = {
                "min": min(values),
                "max": max(values),
                "mean": round(statistics.mean(values), 3),
            }
    if numeric_summary:
        result["summary"] = numeric_summary

    return json.dumps(result, ensure_ascii=False, indent=2)


def analyze_numeric_data(values: str, labels: str = "") -> str:
    """对一组数值做统计分析。

    Args:
        values: 逗号分隔的数值，如 "2.6,2.8,3.0,3.2"
        labels: 可选，逗号分隔的标签，与 values 一一对应
    """
    nums: list[float] = []
    for part in values.split(","):
        part = part.strip()
        if part:
            nums.append(float(part))

    if not nums:
        return "未提供有效数值。"

    label_list = [x.strip() for x in labels.split(",") if x.strip()] if labels else []

    result = {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "mean": round(statistics.mean(nums), 4),
        "stdev": round(statistics.stdev(nums), 4) if len(nums) > 1 else 0,
    }

    if label_list and len(label_list) == len(nums):
        result["labeled_values"] = dict(zip(label_list, nums))

    return json.dumps(result, ensure_ascii=False, indent=2)
