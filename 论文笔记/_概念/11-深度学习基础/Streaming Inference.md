---
type: concept
aliases: [流式推理, 流式生成, Streaming Generation, 边算边推]
---

# Streaming Inference

## 定义

一种推理范式，允许模型在生成完整序列之前就开始输出/执行已完成的部分结果，通过客户端-服务端流式管线将计算延迟与执行时间重叠，降低有效反应延迟。

## 数学形式

设生成总延迟 $T_{gen}$，执行时间 $T_{exec}$：

- 非流式：反应延迟 $= T_{gen}$
- 流式：反应延迟 $= T_{first\_token}$（第一个可执行 token 到达时间）

## 核心要点

1. **延迟-质量解耦**: 首 token 就绪即可开始执行，无需等待完整序列
2. **计算-执行重叠**: 后续 token 生成与前面 token 执行并行，掩藏推理延迟
3. **动作块分割**: 配合 [[Action Chunking]]，流式输出各时间步动作
4. **异步管线**: 通常需要 Client（执行端）和 Server（推理端）分离架构

## 代表工作

- [[FASTER]]: 专门研究 VLA 流式推理的论文，在消费级 GPU 上将反应延迟降低 2.6×
- [[UniVLA]]: 潜在的流式推理优化方向

## 相关概念

- [[Action Chunking]]: 流式推理的执行粒度单位
- [[Autoregressive Transformer]]: 流式推理的主要适用架构
- [[VLA]]: 流式推理的核心应用场景
