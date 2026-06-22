---
type: concept
aliases: [MoT, Mixture-of-Transformers, 混合变换器]
---

# 混合专家 Transformer

## 定义

将多个专门化的 Transformer 子网络（"专家"）并行部署，每个专家负责处理特定模态或任务的 Token，由路由机制分配输入，从而在单一框架内高效联合建模多模态序列。

## 数学形式

对于每个 Token $x_i$，路由函数 $r(x_i) \in \{1, \ldots, K\}$ 决定使用哪个专家 $E_k$：

$$
\text{MoT}(x_i) = E_{r(x_i)}(x_i)
$$

在 Kairos 的 World-Action Model 中，路由按 Token 类型（视频 vs 动作）硬路由：
- 视频 Token → Video DiT（全规模）
- 动作 Token → Action DiT（约 1/5 规模）

## 核心要点

1. **条件计算**：不同 Token 类型走不同专家，计算成本随模态数线性而非二次增长
2. **专家规模不对称**：Action DiT 显著小于 Video DiT，动作预测高效推理
3. **权重初始化**：Action DiT 权重由 Video DiT 插值初始化，加速收敛、复用视频先验

## 代表工作

- [[Kairos]]: 在 World-Action Model 中使用 Video DiT + Action DiT 的 MoT 结构

## 相关概念

- [[DiT]]
- [[门控线性注意力]]
