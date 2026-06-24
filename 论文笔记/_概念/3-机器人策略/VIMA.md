---
type: concept
aliases: [VIMA, Visual Imitation with Masked Autoencoding]
---

# VIMA

## 定义
多模态 Transformer 机器人操作模型，将视觉-语言指令（含参考图像、文字、物体 token）统一为 prompt，用自回归 Transformer 预测动作序列。支持多种任务类型的零样本泛化。

## 数学形式
$$a_{1:T} = \pi_\theta(\text{prompt}, o_{1:T})$$

其中 prompt 包含任务描述（文字 + 图像示例），$o_t$ 为当前观测，自回归输出动作序列。

## 核心要点
1. 多模态 prompt：将任务图像、语言描述、参考物体 token 拼接为统一序列
2. 17 类任务评测：包括 novel object、visual goal、one-shot video demo 等多种泛化场景
3. 与 GPT 类模型同框架，验证 scaling 对机器人策略的有效性

## 代表工作
- [[VIMA]]: Jiang et al., 2022 (NVIDIA + Stanford)，原始论文
- [[Autonomous-Video-WM]]: 以 VIMA 数据集作为自演化世界模型的评测环境

## 相关概念
- [[OpenVLA]] — 同类 VLA 方向
- [[RT-2]] — scaling 路线对比
- [[VLA]] — 上层方向概念
