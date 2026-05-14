---
type: concept
aliases: [Dreamer4, DreamerV4]
---

# Dreamer 4

## 定义

Dreamer 系列的最新版本（2025 年），延续基于 latent dynamics 的 model-based RL 路线，并扩展到机器人 manipulation 与具身推理场景，是当前 imagination-driven 策略学习的代表。

## 核心要点

1. **Latent-space rollout**: 在压缩潜空间中做长程想象，避免像素生成开销。
2. **Embodied 扩展**: 综述中作为 RL-for-VLA 的关键节点出现。
3. 不属于狭义 WAM（其 action 生成与 world 建模仍可视为可分模块），但被综述列为 WAM 的概念前身。

## 代表工作

- [[WAM-Survey]] Sec. 3.3 中作为 model-based RL 的现代代表。

## 相关概念

- [[DreamerV3]]
- [[世界模型]]
- [[World Action Model]]
