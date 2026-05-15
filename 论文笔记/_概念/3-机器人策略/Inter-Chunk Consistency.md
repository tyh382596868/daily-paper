---
type: concept
aliases: [Inter-Chunk Consistency, ICC, ICC-L2, 块间一致性]
---

# Inter-Chunk Consistency

## 定义

度量 chunk-based [[VLA]] 在**重规划**时动作抖动程度的指标：在 $t$ 时刻预测的 chunk 与 $t + r$ 时刻重规划后的 chunk，在公共未来时间窗口上做 L2 距离平均，值越低代表策略越自洽。

## 数学形式

$$
\text{ICC}_t = \frac{1}{H - r} \sum_{j = r}^{H-1} \big\| \hat{a}_{t+j}^{(t)} - \hat{a}_{t+j}^{(t+r)} \big\|_2^2
$$

## 核心要点

1. **正交于成功率**: 即便最终成功，高 ICC 也意味着轨迹不顺滑、对噪声敏感
2. **混叠的直接症状**: [[观测混叠]] 会让相邻时刻的预测意图跳变，ICC 飙升
3. **扫描重规划间隔** $r$: 同一方法在不同 $r$ 上的 ICC 曲线刻画其"短期承诺稳定性"
4. **配套指标**: 与 90 分位 ICC 一起报告，避免均值掩盖长尾抖动

## 代表工作

- [[IntentVLA]]: 在 [[AliasBench]] 上 IntentVLA 把均值 ICC-L2 降低 17.6%，90 分位降低 21.7%

## 相关概念

- [[Action Chunking]]
- [[观测混叠]]
- [[历史条件化策略]]
