---
type: concept
aliases: [关键帧一致性训练, 关键帧加权损失]
---

# Keyframe-Consistent Training（关键帧一致性训练）

## 定义

一种针对长时域机器人操作的训练策略，通过在关键帧附近的时间步赋予更高的损失权重，强化模型对任务状态转变时刻的学习。

## 数学形式

$$
w_t = \begin{cases} \lambda & \text{if } \min_i |t - k_i| \leq \delta \\ 1 & \text{otherwise} \end{cases}
$$

$$
\mathcal{L} = \sum_t w_t \cdot \mathcal{L}_t^{\text{VLA}}
$$

其中 $k_i$ 为关键帧时间步，$\lambda = 8.0$ 为放大系数，$\delta = 3$ 为邻域半径。

## 核心要点

1. **非均匀加权**: 关键帧邻域的损失权重远高于普通帧（8 倍），迫使模型在状态转变时学得更精准
2. **与关键帧检测联动**: 训练时需先检测关键帧位置（[[Event Saliency Score]]），才能确定加权区域
3. **消融验证**: 去掉关键帧一致性训练后，Cover Blocks 任务 TSR 从 9/12 降至 3/12

## 代表工作

- [[KEMO]]: 提出此训练策略，与事件驱动关键帧检测和门控残差融合协同使用

## 相关概念

- [[Event Saliency Score]]
- [[Gated Residual Fusion]]
- [[VLA]]
