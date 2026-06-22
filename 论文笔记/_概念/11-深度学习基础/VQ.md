---
type: concept
aliases: [Vector Quantization, 向量量化, VQ-VAE, Codebook]
---

# VQ（向量量化）

## 定义
Vector Quantization（VQ）是一种将连续向量映射到离散码本（codebook）中最近邻向量的技术，用于压缩、离散表示学习和生成模型。

## 数学形式
给定连续向量 $z_e$，从码本 $\{e_k\}_{k=1}^K$ 中选择最近邻：

$$z_q = e_k, \quad k = \arg\min_j \| z_e - e_j \|_2$$

训练时用 straight-through estimator 传递梯度，损失：

$$\mathcal{L}_{VQ} = \| sg[z_e] - z_q \|^2 + \beta \| z_e - sg[z_q] \|^2$$

其中 $sg[\cdot]$ 为 stop-gradient。

## 核心要点
1. 提供离散的、有限词汇量的潜变量表示
2. 在 VQ-VAE 中用于图像/视频生成，在 VLA 中用于动作离散化
3. Latent Action 框架（UniVLA、LAPA）用 VQ 将人类视频中的运动量化为可跨 embodiment 的动作 token
4. 码本更新通常用 EMA 或 commit loss

## 代表工作
- [[UniVLA]]: 用 VQ 离散化 latent action 跨 embodiment 训练
- [[LAPA]]: Motion-focused VQ codebook 训练跨 embodiment VLA
- [[VQ-VAE]]: VQ 的奠基性工作

## 相关概念
- [[Codebook]]
- [[UniVLA]]
- [[LAPA]]
- [[Discrete Representation Learning]]
