---
type: concept
aliases: [时间方向, 时间之箭, Arrow of Time]
---

# Arrow of Time

## 定义

物理学/认知科学概念：时间具有不可逆方向性。在视觉数据中表现为"事件按因果顺序展开"——打碎的杯子不会自己拼回，水不会逆流。深度学习中，被用作探测模型是否学到时序因果的信号。

## 核心要点

1. 视觉上的 Arrow of Time 主要由 **不可逆事件** 提供（碰撞、相变、扩散）
2. 用作 [[Video Diffusion Model|VDM]] 的探针：如果模型能区分正向/反向视频，说明感知到了时间方向
3. 感知到时间方向 ≠ 理解因果（[[YoCausal]] 给出实证）

## 代表工作

- [[YoCausal]]: 用 [[Reverse Surprise Index|RSI]] 量化 VDM 的时间方向感知
- Wei et al. "Learning and Using the Arrow of Time" (CVPR 2018): 经典前作

## 相关概念

- [[Causality Cognition Index]]
- [[Counterfactual]]
- [[World Model]]
- [[Violation of Expectation]]
