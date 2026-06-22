---
type: concept
aliases: [PAI-Bench, Physical AI Benchmark]
---

# PAI-Bench

## 定义
PAI-Bench（Physical AI Benchmark）是一个用于评测具身 AI / 世界模型视频生成质量的基准测试集，包含 Domain Score（领域任务相关性）、Quality Score（视觉生成质量）和 Overall Score（综合）三个核心指标维度，设有机器人操作、自动驾驶等多个子域。

## 核心要点
1. **Domain Score**: 衡量生成视频与目标任务领域的贴合程度，包含任务完成度和物理合规性等。
2. **Quality Score**: 衡量视频的视觉质量，包括帧一致性、清晰度等感知指标。
3. **Overall Score**: Domain 与 Quality 的综合分数，反映世界模型整体能力。
4. 专门设有 robot domain 子集，适合评估机器人操作世界模型。

## 代表工作
- [[RewardAgent]]: 在 PAI-Bench robot 子集上评测 DynDiff-GRPO，Cosmos-Predict2.5 Overall +1.33，Kairos-3.0-Robot Overall +1.53。

## 相关概念
- [[AgiBotWorld-Beta]]
- [[Embodied AI]]
- [[Video Diffusion Model]]
