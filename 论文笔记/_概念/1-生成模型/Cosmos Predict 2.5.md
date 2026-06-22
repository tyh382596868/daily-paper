---
type: concept
aliases: [Cosmos Predict, NVIDIA Cosmos]
---

# Cosmos Predict 2.5

## 定义

Cosmos Predict 2.5 是 NVIDIA 推出的通用物理 AI 视频预测模型，可在给定初始帧和文本/动作条件下生成未来帧序列，在机器人操作世界模型评估中常作为通用视频预测基线。

## 核心要点

1. **通用物理 AI**: 面向物理仿真和机器人场景设计，强调物理一致性
2. **基线角色**: 在 [[MemWorld]] 的世界模型一致性评估中作为无动作条件的通用基线，第三视角 PSNR 22.80，SSIM 0.819
3. **与专用模型差距**: 专用动作条件模型（CtrlWorld、MemWorld）在机器人操作场景指标上明显优于 Cosmos Predict 2.5
4. **预训练规模**: 基于大规模视频数据预训练，具备通用场景理解能力

## 代表工作

- [[MemWorld]]: 作为对比基线（Table 1 中第三视角 PSNR 22.80，低于 Mem-World 的 25.30）

## 相关概念

- [[视频扩散模型]]
- [[Action-Conditioned World Model]]
- [[1-生成模型]]
