---
type: concept
aliases: [NoMaD, No Map Diffusion]
---

# NoMaD

## 定义
基于扩散模型的无地图视觉导航框架，生成多假设轨迹并通过目标图像条件引导导航。

## 核心要点
1. 无需预建地图，直接从 RGB 图像 + 目标图像生成候选轨迹
2. 扩散模型生成多假设，配合 scoring function 选择最优轨迹
3. GNM/ViNT/NoMaD 是递进的无地图导航方法系列

## 代表工作
- 原始论文：Sridhar et al., 2023
- [[Slow-Brain]]：VLM-Augmented Urban Navigation（2606.20458）中作为基础框架

## 相关概念
- [[GNM]]
- [[Diffusion Policy]]
- [[NaVILA]]
