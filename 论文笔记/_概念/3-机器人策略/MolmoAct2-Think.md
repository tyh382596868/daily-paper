---
type: concept
aliases: [MolmoAct2-Think, MolmoThink]
---

# MolmoAct2-Think

## 定义

MolmoAct2 的具身推理增强变体，在标准 VLA 推理流程中融入自适应深度 token 预测：仅对场景发生变化的空间区域重新预测深度 token，静态区域复用缓存，在提升几何感知能力的同时降低推理延迟。

## 数学形式

$$
m_{t,i} = \mathbf{1}\!\left[\cos(x_{t,i},\, x_{t-1,i}) < 0.996\right]
$$

当 patch 余弦相似度低于阈值时，触发该区域的深度 token 重新预测。

## 核心要点

1. 10×10 量化深度网格（每格 128 个深度码），通过 VQ-VAE + Depth Anything V2 生成
2. 逐层门控机制控制深度条件化强度（初始偏置 -4 保守激活）
3. 在 LIBERO 基准达到 98.1%，超越所有对比方法

## 代表工作

- [[MolmoAct2]]: MolmoAct2-Think 是 MolmoAct2 的推理增强版本

## 相关概念

- [[Flow Matching]]
- [[具身推理]]
- [[逐层 KV 连接]]
