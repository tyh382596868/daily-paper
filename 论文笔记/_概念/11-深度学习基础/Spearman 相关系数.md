---
type: concept
aliases: [Spearman 相关系数, Spearman Correlation, Spearman Rank Correlation, Spearman ρ]
---

# Spearman 相关系数

## 定义

Spearman 等级相关系数，用**两组数据各自的排名**之间的 [[Pearson 相关系数]] 衡量单调相关性，不要求线性、对 outlier 鲁棒。

## 数学形式

把 $x_i, y_i$ 转成排名 $\mathrm{rg}(x_i), \mathrm{rg}(y_i)$，则:
$$
\rho_s = \frac{\mathrm{cov}(\mathrm{rg}(X), \mathrm{rg}(Y))}{\sigma_{\mathrm{rg}(X)} \, \sigma_{\mathrm{rg}(Y)}}
$$

无 tied ranks 时简化为:
$$
\rho_s = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}, \quad d_i = \mathrm{rg}(x_i) - \mathrm{rg}(y_i)
$$

## 核心要点

1. **范围**: $\rho_s \in [-1, 1]$
2. **vs Pearson**: 只看排序一致性，不看数值——适合**单调但非线性**的关系
3. **vs Kendall τ**: Spearman 比 Kendall 略不稳健但对大样本计算更快
4. **典型应用 ([[SCSA]])**: [[TRM]] 用 Spearman 衡量「学习型代价」与「真值代价（如测地距离）」的排序一致性。例如 hard n100 LeWM seed 3072 上：
   - Raw latent MSE: $\rho=0.018$
   - TRM true labels: $\rho=0.729$
5. **不需要分布假设**: 非参数统计量，适合任意有序数据

## 代表工作

- Spearman, 1904: 原始定义
- [[TRM]]: 用作 [[SCSA]] 的核心指标

## 相关概念

- [[Pearson 相关系数]]
- [[Kendall Tau]]
- [[SCSA]]
