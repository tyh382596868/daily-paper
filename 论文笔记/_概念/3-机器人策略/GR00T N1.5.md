---
type: concept
aliases: [GR00T N1.5, GROOT N1.5, GR00T-N1.5]
---

# GR00T N1.5

## 定义

NVIDIA 推出的通用人形机器人 [[VLA]] 基础模型 N1 系列的 1.5 版本，介于 N1 和 [[GR00T N1.7]] 之间，是当前空间/记忆型 VLA 研究中最常用的 baseline 之一。

## 核心要点

1. NVIDIA 通用人形 VLA 系列的中间版本（N1 → N1.5 → N1.7）
2. VLM 主干 + [[DiT]] 动作头 + [[Action Chunking|动作块]] 输出
3. 在 RoboCasa、[[SimplerEnv]] 等 benchmark 上作为强 baseline
4. 在 [[OOV (Out-of-Vision)|OOV]] 场景下表现差（目标不可见时反应式失败）

## 代表工作

- [[SOMA]]: 主干即 GR00T N1.5，在其上插入空间记忆模块
- 多个 2026 年 VLA 论文的对比基线

## 相关概念

- [[GR00T N1.7]]
- [[VLA]]
- [[DiT]]
- [[Action Chunking]]
- [[SimplerEnv]]
- [[RoboCasa]]
