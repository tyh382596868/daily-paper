---
type: concept
aliases: [AnchorDream]
---

# AnchorDream

## 定义

通过把渲染的机器人轨迹视频作为 anchoring 条件喂给视频扩散模型，从而避免 [[embodiment hallucination|具身幻觉]] 的一类机器人数据合成方法。RoboDream 的最直接前辈/同期工作。

## 核心要点

1. 用 IK 渲染机器人 only 视频作为条件，确保生成视频中机器人姿态严格可控
2. 需要 **任务特定微调** — 每换一个场景就要重新训练 → 这是 RoboDream 重点改进的地方
3. 在论文中作为 ablation baseline 出现

## 代表工作

- [[RoboDream]]: 在 AnchorDream 基础上引入组合式条件（场景 + 物体先验），实现零样本泛化

## 相关概念

- [[embodiment hallucination]]
- [[Embodiment Anchoring]]
- [[Video Diffusion Model]]
- [[Compositional World Models]]
