---
type: concept
aliases: [Horizon-Matched Supervision, Horizon-Aware Supervision, 跨 Horizon 监督]
---

# Horizon-Matched Supervision

## 定义

训练辅助度量 / 代价网络时，刻意**让监督信号的时间跨度分布匹配下游规划器实际使用的 horizon 分布**的设计准则。在 [[TRM]] 中是「成对头训练对的时间间隔 $\Delta$ 必须均衡覆盖整个 episode」的具体实现。

## 数学形式

监督标签:
$$
y_{ij} = |t_i - t_j|
$$

Balanced 采样: 先均匀采 $\Delta \sim \mathcal{U}[1, L_e-1]$，再均匀采起点 $t \sim \mathcal{U}[0, L_e-\Delta-1]$。

## 核心要点

1. **核心论断**: 训练分布与推理分布必须匹配——下游 MPC 用长 horizon 终端排序，但短 horizon 对会让头只学到局部度量
2. **[[TRM]] 关键证据 (Table 3)**: 同样 100k 训练对预算
   - Max-$\Delta=50$（短 horizon）→ 35.0% success
   - Random full-episode → 90.0%
   - Balanced full-episode → **97.5%**
3. **不是数据量问题**: 100k 对是充足的，差距完全来自**时间跨度分布**
4. **设计准则推广**: 任何要在长 horizon 上排序的辅助监督都应遵循此准则——含 value learning、reward shaping 等

## 代表工作

- [[TRM]]: 该术语的原始提出

## 相关概念

- [[TRM]]
- [[Pairwise Ranking Head]]
- [[Latent MPC]]
