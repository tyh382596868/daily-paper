---
type: concept
aliases: [NVIDIA FP4, FP4 quantization]
---

# NVFP4

## 定义

**NVFP4** 是 NVIDIA 在 Blackwell 架构上引入的 4-bit 浮点量化格式（E2M1 + block-wise scaling），用于在保持精度损失可控的前提下把大模型推理吞吐提升约 2×，并大幅降低显存占用。

## 核心要点

1. **格式**: 2 位指数 + 1 位尾数 + 1 位符号 + per-block scale (FP8)
2. **硬件原生**: Blackwell GPU 提供 FP4 张量核
3. **典型加速**: vs BF16 约 2×；vs FP8 约 1.5×
4. **精度保持**: 通过 calibration + smooth quantization 控制掉点在 <1%

## 代表工作

- [[Cosmos3]]: 部署时支持 BF16 / FP8 / NVFP4 三档量化
- 大量 LLM 推理框架（vLLM / TensorRT-LLM）已集成

## 关联

- vLLM
- FP8 / BF16
- [[Cosmos3]] 部署栈
