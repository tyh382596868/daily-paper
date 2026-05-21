---
type: concept
aliases: [加加速度, 加速度变化率, 加加速度指标]
---

# Jerk

## 定义

加速度的时间导数（位置的三阶导数），单位 $\text{rad/s}^3$ 或 $\text{m/s}^3$。机器人控制中用作**轨迹平滑度**的核心评价指标：jerk 越小，运动越柔顺、对硬件磨损越小。

## 数学形式

$$
\mathbf{j}(t) = \frac{d\mathbf{a}(t)}{dt} = \frac{d^2 \mathbf{v}(t)}{dt^2} = \frac{d^3 \mathbf{x}(t)}{dt^3}
$$

## 核心要点

1. 高 jerk 会导致：电机震荡、关节磨损、抓取失稳、视觉跟踪抖动；
2. 评估时通常报告 **Avg Jerk** 和 **Max Jerk** 两个指标；
3. [[Action Chunking]] 方法的块间边界容易产生 jerk 尖峰；
4. [[AR-VLA]] 中 Avg=7.89，比 [[OpenVLA]] (10.13) 低 22%，是真正跨时间 AR 的直接收益。

## 代表工作

- [[Diffusion Policy]] / [[ACT (Action Chunking Transformer)|ACT]]：作为标准评估指标
- [[AR-VLA]]：用 jerk 论证时序连贯性

## 相关概念

- [[轨迹平滑]]
- [[Action Chunking]]
- [[Inter-Chunk Consistency]]
