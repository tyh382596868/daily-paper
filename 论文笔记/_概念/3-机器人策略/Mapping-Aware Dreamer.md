---
type: concept
aliases: [MAD, Mapping-Aware Dreamer]
---

# Mapping-Aware Dreamer

## 定义

**Mapping-Aware Dreamer (MAD)** 是 Zhang et al. (2026) 提出的[[潜空间世界模型]]，把 [[Dreamer]] 风格 [[循环状态空间模型|RSSM]] 的视觉重建头**替换为机器人本体坐标系下的 [[占用栅格图|OGM]] + [[可见性栅格图|VGM]] 重建头**，使 latent 显式编码"局部几何 + 观测历史"，专为四旋翼自主飞行设计。

## 关键组件

- **编码器**：深度图 ($18 \times 32$) 经 CNN + 本体感知 9 维经 MLP
- **RSSM**：confirm 化离散 categorical latent $z_t$ + 确定性 $h_t$
- **解码头**：OGM、VGM、本体感知、奖励、继续 — 五个独立头
- **OGM/VGM 配合 BCE 掩码**：occupancy BCE 只在 $g^{vis}=1$ 的体素上算

## 下游策略

同一个 MAD 表征支持三种下游：

- **MAD-Dreamer**：[[DreamerV3]] 流程，在想象 rollout 上做 actor-critic
- **MAD-PPO**：冻结 MAD 编码器 + [[PPO]] 策略头
- **MAD-SHAC**：冻结 MAD 编码器 + [[SHAC]] 策略头

## 关键结果

- 仿真 (DiffAero/Gazebo) 森林峰速 **9.66 m/s**
- 室内走廊 5×40 m 完成 8.36 s / 峰速 6.37 m/s
- 户外森林实飞峰速 **5.05 m/s**（零调参 [[Sim-to-Real|sim-to-real]]）
- 冻结表征可迁移到竞速 (gate-passing) 任务

## 代表工作

- [[MAD]]（即本论文）

## 关联概念

- [[Dreamer]] / [[DreamerV3]]
- [[循环状态空间模型|RSSM]]
- [[占用栅格图|OGM]] / [[可见性栅格图|VGM]]
- [[DiffAero]]
