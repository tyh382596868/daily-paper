---
type: concept
aliases: [Molmo Act2 Think, adaptive depth reasoning]
---

# MolmoAct2-Think

## 定义

MolmoAct2-Think 是 MolmoAct2 的自适应深度推理变体，通过检测相邻帧深度嵌入的余弦相似度，仅对发生变化的场景区域重新预测深度 token，静态区域复用缓冲值，在提升性能的同时降低推理延迟。

## 数学形式

更新掩码：$m_{t,i} = \mathbb{1}[\cos(x_{t,i}, x_{t-1,i}) < 0.996]$

逐层门控：$g_\ell = \sigma(w_\ell c_\ell + b_\ell)$，初始偏置 $b_\ell = -4$

## 核心要点

1. 深度图量化为 10×10 网格，128 个学习深度码值
2. 余弦相似度阈值 0.996，触发局部深度 token 更新
3. LIBERO 平均成功率 98.1%，超越 π₀.₅（96.9%）

## 代表工作

- [[MolmoAct2]]：MolmoAct2-Think 是其自适应推理版本

## 相关概念

- [[逐层 KV 连接]]
- [[Flow Matching]]
- [[VLA]]
