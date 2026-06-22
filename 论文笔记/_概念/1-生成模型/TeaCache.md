---
type: concept
aliases: [Tea Cache, 残差缓存加速, Timestep Embedding Aware Cache]
---

# TeaCache

## 定义

一种针对扩散模型（特别是 [[Diffusion Transformer]]）推理加速的方法，通过在时间步稳定区间内缓存并复用 DiT 块的残差输出，避免重复的前向计算，显著降低推理延迟。

## 核心要点

1. **残差复用**: 检测时间步之间 DiT 块输出变化较小的"稳定区域"，直接复用上一步的残差
2. **选择性计算**: 仅在变化较大的时间步（通常是早期去噪步）重新计算，稳定步直接跳过
3. **无需重训练**: 作为推理时插件，对模型权重无改动
4. **与量化兼容**: 可与 INT8/FP8 量化、序列并行等其他加速手段叠加使用

## 代表工作

- [[DreamXWorld]]: 在推理加速模块中使用 TeaCache 进行残差复用，配合 INT8 SageAttention 和 FP8 量化实现 8 卡 RTX 5090 上 16 FPS

## 相关概念

- [[Diffusion Transformer]]: 主要应用目标架构
- [[自回归流式推理]]: 配套推理范式
- [[少步蒸馏]]: 另一类推理加速方向
