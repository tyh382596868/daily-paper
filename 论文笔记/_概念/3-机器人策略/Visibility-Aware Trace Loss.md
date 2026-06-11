---
type: concept
aliases: [Visibility-Aware Trace Decoder, Visual Latent Plan Encoder, Visibility Mask Loss, Multi-View Trace Loss]
---

# Visibility-Aware Trace Loss

## 定义

一种训练辅助损失，用于在多视角机器人操作中监督末端执行器轨迹预测；核心设计是引入可见性标志，当末端执行器在某相机视角不可见时自动屏蔽坐标损失，避免 ill-posed 梯度。由 [[SeeTraceAct]] 提出。

## 数学形式

$$
\mathcal{L}_{\text{trace}}^{k} = \frac{\sum_{i=1}^{N} v^k_i \cdot \| \hat{T}^k_i - T^k_i \|_2^2}{\max\!\left(\sum_i v^k_i, 1\right)} + \lambda_v \cdot \text{BCE}(\hat{v}^k_i, v^k_i)
$$

多视角总损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda_t \sum_{k=1}^{K} \mathcal{L}_{\text{trace}}^{k}
$$

## 核心要点

1. **可见性标志**: 对每视角每帧预测二值可见性 $\hat{v}^k_i$，与坐标预测并行输出
2. **掩码梯度**: 坐标损失仅在真实可见帧（$v^k_i=1$）上计算，推理端无此约束
3. **双分支解码**: 轨迹解码器同时输出坐标回归头和可见性分类头
4. **训练专用**: 推理时完全丢弃解码器，无额外计算开销

## 符号说明

- $v^k_i \in \{0,1\}$: 第 $k$ 视角第 $i$ 帧真实可见性标签
- $T^k_i$: 通过正向运动学 + 相机投影计算的真实末端轨迹坐标
- $\lambda_t, \lambda_v$: 损失权重超参数

## 代表工作

- [[SeeTraceAct]]: 提出并验证该损失函数

## 相关概念

- [[TraceVLA]]
- [[Demo-Conditioned VLA]]
- [[Cross-Embodiment Learning]]
