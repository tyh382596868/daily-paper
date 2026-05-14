---
type: concept
aliases: [Dream VLA]
---

# DreamVLA

## 定义

预测**结构化世界知识**（动力学、空间、语义）而非原始未来像素的统一 [[VLA]]，是 [[RobotWM-Survey]] 中"隐式/潜在未来建模"子类的代表。

## 核心要点

1. 区别于 [[GR-1]]/[[WorldVLA]] 的"显式未来图像预测"，DreamVLA 在 MLLM 内部潜空间做世界建模
2. 把世界建模解耦为三类知识：dynamics（如何变）、spatial（在哪里）、semantic（是什么）
3. 不需要解码像素，推理效率比像素级 WM 高
4. [[RobotWM-Survey]] Section 3.5 中归类为"latent/implicit future modeling"

## 代表工作

- Zhang et al., 2025e: DreamVLA 原始论文
- CoWVLA: 类似潜空间路线，使用 latent motion 与紧凑视觉目标
- UniVLA: post-training 阶段做世界建模

## 相关概念

- [[VLA]]
- [[GR-1]]
- [[WorldVLA]]
- [[UniVLA]]
- [[FLARE]]
- [[RobotWM-Survey]]
