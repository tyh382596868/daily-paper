---
type: concept
aliases: [H100, NVIDIA H100, H200, H100 GPU, H200 GPU]
---

# H100 GPU

## 定义

NVIDIA H100（及其升级版 H200）是基于 Hopper 架构的数据中心 GPU，专为大规模 AI 训练和推理设计。

## 核心要点

1. **H100**: 80GB HBM3 显存，SXM5 版本提供 3.35 TB/s 显存带宽，FP8 峰值算力 3958 TFLOPS。
2. **H200**: 141GB HBM3e 显存，显存带宽提升至 4.8 TB/s，适合更大模型训练。
3. **NVLink**: 支持 NVLink 4.0，多卡互联带宽 900 GB/s，适合分布式大模型训练。
4. 当前（2025-2026）VLA / LLM 训练的主流硬件平台。

## 代表工作

- [[ROAD-VLA]]: 使用 NVIDIA H100/H200 GPU + 64 并行环境做 VLA 在线 RL

## 相关概念

- [[OpenVLA]]
- [[VLA]]
