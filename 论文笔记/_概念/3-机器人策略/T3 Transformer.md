---
type: concept
aliases: [T3, Tactile Transformer, T3 Tactile Encoder]
---

# T3 Transformer

## 定义
T3（Tactile Transformer）是一个专为触觉信号设计的预训练 Transformer 编码器，通过对大规模触觉数据进行自监督预训练，学习通用的触觉表征。

## 核心要点
1. 以 ViT 为骨干，专门在触觉图像数据上预训练
2. 输出 [CLS] token 作为压缩的触觉特征向量
3. 在 [[FTP-1]] 中用作图像型触觉传感器（如 GelSight）的共享编码器，跨传感器共享参数

## 代表工作
- [[FTP-1]]: 使用预训练 T3 作为图像型传感器编码器，轻量 ViT 与 T3 联合提取触觉令牌

## 相关概念
- [[MTTS]]
- [[ViT]]
- [[触觉传感器]]
- [[GelSight]]
