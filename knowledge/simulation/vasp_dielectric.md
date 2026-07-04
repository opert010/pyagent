# VASP 计算介电常数参数指南

## 适用场景
第一性原理（DFT）计算可用于预测晶体材料的静态介电常数，适用于 SiO₂、HfO₂ 等无机材料。

## 关键 INCAR 参数
```
IBRION = 8          # DFPT 计算
LEPSILON = .TRUE.   # 计算介电张量
LPEAD = .TRUE.      # 使用 modern 算法
ENCUT = 520         # 截断能，依 POTCAR 调整
EDIFF = 1E-6        # 电子步收敛
NSW = 0             # 离子步为 0（仅电子自洽 + DFPT）
```

## 计算流程
1. 结构优化（ISIF=3）获得平衡晶格
2. 高精度静态自洽（EDIFF=1E-8）
3. DFPT 计算介电张量 ε∞
4. 如需包含离子贡献，启用 LEPSILON + 适当 k 点密度

## 注意事项
- k 点密度对介电常数敏感，建议 ≥ 6×6×6（体相材料）
- 带隙低估会影响介电常数，必要时使用 HSE06 或 scissor 修正
- 非晶材料需构建 amorphous 模型或使用 MD + CLAMP 方法
