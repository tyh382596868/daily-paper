---
type: concept
aliases: [Wan2.2-TI2V, Wan Video Model]
---

# Wan2.2

## 定义

阿里巴巴开源的大规模视频生成模型，基于[[扩散变换器|DiT]]架构和[[Flow Matching]]训练目标，支持文本到视频（T2V）和图像到视频（I2V，即 TI2V）等多种生成模式。

## 核心要点

1. 采用 DiT（[[扩散变换器]]）架构，以 token 化的视频 patch 为输入
2. 使用 [[Flow Matching]] 作为训练目标，推理效率高于 DDPM
3. TI2V（Text+Image to Video）模式：以参考图像为第一帧，文本为条件，生成后续帧
4. 在 WorldArena 基准上作为强基线（P3CScore 60.83），但缺乏精确动作控制能力
5. 代码开源：https://github.com/Wan-Video/Wan2.2

## 代表工作

- [[EA-WM]]: 以 Wan2.2-TI2V 为主干，通过 LoRA 微调引入 KVAF 动作条件和事件感知融合模块

## 相关概念

- [[视频扩散模型]]
- [[扩散变换器]]
- [[Flow Matching]]
- [[LoRA]]
