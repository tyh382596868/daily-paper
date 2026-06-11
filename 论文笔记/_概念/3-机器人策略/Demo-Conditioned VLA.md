---
type: concept
aliases: [Demo-Conditioned Policy, One-Shot Demo VLA, 演示条件化VLA]
---

# Demo-Conditioned VLA

## 定义

给定单个（或少量）演示视频作为 in-context 条件，机器人策略在推理时执行相同或类似任务的范式；属于 [[VLA]] 的子类，强调跨体 / 跨场景的 one-shot 泛化能力。

## 核心要点

1. **One-shot 泛化**: 每个新任务仅提供一条演示视频，策略从演示中提取任务意图，无需微调
2. **跨体演示**: 演示可来自不同机体（人形机器人、人类手部），需桥接跨体外观差距
3. **条件化机制**: 通常通过 cross-attention 将演示特征注入策略骨干，或编码为潜在向量
4. **评测挑战**: 需同时评测已见任务（同分布）和未见任务（跨分布）泛化性

## 代表工作

- [[SeeTraceAct]]: Visibility-aware latent planning，通过轨迹预测提升空间定位精度
- [[GR00T N1]]: 人形机器人基础模型，支持演示条件化
- [[SWIM]]: 另一类 demo-conditioned 基线方法

## 相关概念

- [[VLA]]
- [[Cross-Embodiment Learning]]
- [[Action Chunking]]
- [[Visual Latent Plan Encoder]]
