---
type: concept
aliases: [动作块, 动作分块, Action Chunk]
---

# Action Chunking

## 定义

动作分块（Action Chunking）是一种机器人策略的输出方式，指策略在每个时间步一次性预测未来 $H$ 步的动作序列，而非逐步预测单步动作。

## 数学形式

$$
\mathbf{A}_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}] \in \mathbb{R}^{H \times d_a}
$$

其中 $H$ 为 chunk 长度（典型值 8~16），$d_a$ 为单步动作维度（如 7 维末端执行器控制）。

## 核心要点

1. **减少复合误差**: 一次预测多步可以利用全局依赖，避免逐步预测的误差累积
2. **提高推理效率**: 多步动作一次生成，降低对推理频率的要求
3. **与扩散/流匹配结合**: Diffusion Policy、Flow Matching 等生成模型常以 chunk 为单位建模动作分布
4. **滑动窗口执行**: 实际执行时通常采用时序集成（temporal ensemble），对相邻 chunk 的重叠动作加权平均

## 代表工作

- [[OA-WAM]]: 16 步动作块，Flow Matching 解码
- [[Pi05|π₀.₅]]: 动作分块 + 扩散策略

## 相关概念

- [[Flow Matching]]
- [[VLA]]
- [[World Action Model]]
