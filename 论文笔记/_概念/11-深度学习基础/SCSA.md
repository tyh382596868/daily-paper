---
type: concept
aliases: [SCSA, Same-Candidate Selection Audit, 同候选选择审计]
---

# SCSA

## 定义

[[TRM]] 论文提出的一种诊断工具：在 [[CEM]] 实际采到的**同一批候选**上，比较不同终端代价的**排序质量**，把「采样到好候选」与「排序到好候选」两个因素**解耦**。

## 数学形式

对一批候选 $\{\mathbf{a}^{(i)}\}_{i=1}^N$，给定参考真值代价 $c^*$（如 oracle / 测地距离）与待审计代价 $c$:

$$
\rho = \mathrm{Spearman}(c^{(i)}, c^{*(i)})
$$

$$
\text{Best-rank percentile} = \frac{\mathrm{rank}_c(\arg\min_i c^{*(i)})}{N}
$$

## 核心要点

1. **解耦目的**: 闭环成功率混合了「采样质量」+「排序质量」，SCSA 只看排序
2. **两项指标**:
   - [[Spearman 相关系数]] $\rho$ vs 欧氏 / 测地距离
   - oracle-best 候选在该代价下的百分位排名（越低越好）
3. **典型结果 (TRM, hard n100, LeWM seed 3072)**:
   - 原始 latent MSE: $\rho_\text{geo}=0.018$, best-rank 31.71 percentile
   - TRM true labels: $\rho_\text{geo}=0.729$, best-rank **3.86 percentile**
   - Shuffled labels: $\rho_\text{geo}=0.119$, best-rank 34.00 percentile
4. **在 PushT 上的诊断价值**: 即便闭环成功率提升有限（接触瓶颈），SCSA 能告诉你排序确实改善了（true ρ=0.957）

## 代表工作

- [[TRM]]: 该诊断工具的原始提出与应用

## 相关概念

- [[TRM]]
- [[CEM]]
- [[Spearman 相关系数]]
- [[Latent MPC]]
