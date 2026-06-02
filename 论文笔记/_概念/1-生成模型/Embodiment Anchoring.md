---
type: concept
aliases: [具身锚定, Embodiment Anchoring]
---

# Embodiment Anchoring

## 定义

在视频生成中**显式渲染机器人本体轨迹**并以像素级条件方式喂给视频扩散模型，使生成视频中机器人姿态、关节、夹爪状态严格遵循输入运动学的技术。是对抗 [[embodiment hallucination|具身幻觉]] 的核心手段。

## 核心要点

1. 由仿真器（如 [[IsaacLab]]）按真实 / 检索得到的关节轨迹渲染机器人 only 视频
2. 渲染视频经 [[VAE|3D Causal VAE]] 编码后**通道维拼接**到带噪 latent，实现 pixel-aligned conditioning
3. 视频扩散模型只需"画"环境与物体，而不需要"想"机器人怎么动 — 大幅降低生成难度
4. 与"文本条件 + 自由想象"路线（[[DreamZero]]）形成鲜明对比

## 代表工作

- [[RoboDream]]: 用 IsaacLab 渲染 $v_{rob}$ 作为 anchoring
- [[AnchorDream]]: 同思路前作，但需任务特定微调

## 相关概念

- [[embodiment hallucination]]
- [[IsaacLab]]
- [[Compositional Conditioning]]
- [[Video Diffusion Model]]
