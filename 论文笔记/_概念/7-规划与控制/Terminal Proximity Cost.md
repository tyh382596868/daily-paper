---
type: concept
aliases: [Terminal Proximity Cost, 终端邻近代价, Latent MSE 终端代价]
---

# Terminal Proximity Cost

## 定义

[[Latent MPC]] 默认的候选打分方式：用候选轨迹**最后一帧**的预测 latent $\hat z_{t+H}$ 与目标 latent $z_g$ 的欧氏距离作为代价。

## 数学形式

$$
c_{\text{lat}}(\mathbf{a}_{0:H-1}) = \| \hat{\mathbf{z}}_{t+H} - \mathbf{z}_g \|_2^2
$$

## 核心要点

1. **隐含假设**: latent 距离的每一维对"可达性"权重相同——这个假设几乎总是错的
2. **失败模式**: 当任务相关变量（如 XY 位置）只占 latent 一小部分子空间，欧氏距离会被其余维度淹没。[[TRM]] 论文实验中 XY-rowspace 只占 latent MSE <1%
3. **直接表现**: hard [[TwoRoom]] 上 [[LeWM]] 仅 7.0% 成功率，[[PLDM]] 32.7%；oracle task-state 代价同 manifest 100%
4. **修复路径**:
   - 子空间手术（用 XY-rowspace MSE → 90.8%）
   - 学习型替代代价（[[TRM]] → 97.0%）
   - [[Hybrid Terminal Cost]]（与原始 latent MSE 加权混合，PushT 上更安全）

## 代表工作

- [[LeWM]]、[[PLDM]]、[[DINO-WM]]: 标配该代价
- [[TRM]]: 系统诊断该代价的失败并提出替代

## 相关概念

- [[Latent MPC]]
- [[TRM]]
- [[Hybrid Terminal Cost]]
- [[Subspace Surgery]]
