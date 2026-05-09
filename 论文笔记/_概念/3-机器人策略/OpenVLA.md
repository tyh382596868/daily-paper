---
type: concept
aliases: [Open Vision-Language-Action]
---

# OpenVLA

## 定义
基于 Llama 2 的开源 Vision-Language-Action 模型，在 Open X-Embodiment 数据集上训练，支持多种机器人平台的通用控制。

## 数学形式
$$a = \text{OpenVLA}_\theta(I, l)$$
其中 $I$ 为 RGB 图像，$l$ 为语言指令，$a$ 为离散化的机器人动作 token 序列。

## 核心要点
1. 将动作向量离散化为 token，用语言模型 next-token prediction 框架预测动作
2. 基于 Prismatic VLM（SigLIP + Llama 2）
3. 在 Open X-Embodiment 970K 轨迹上训练

## 代表工作
- Kim et al., 2024: OpenVLA 原始论文

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[LoRA]]
