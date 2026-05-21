---
type: concept
aliases: [扩散强迫, Diffusion Forcing, DF]
---

# Diffusion Forcing

## 定义
Diffusion Forcing（扩散强迫）是一种序列生成训练范式，扩展标准扩散模型使序列中**不同时间步可以带有不同的噪声水平**，从而把扩散去噪与自回归式因果预测统一起来。

## 数学形式
$$
\mathcal{L}_{\mathrm{DF}}=\sum_{i}\left\|v_{\theta}(\mathbf{z}_{i}^{\sigma_{i}},\sigma_{i},\mathbf{c})-(\boldsymbol{\epsilon}_{i}-\mathbf{z}_{i})\right\|_{2}^{2}
$$
其中 $\mathbf{z}_i^{\sigma_i}=(1-\sigma_i)\mathbf{z}_i+\sigma_i\boldsymbol{\epsilon}_i$，$\sigma_i$ 为第 $i$ 帧的独立噪声水平。

## 核心要点
1. 历史帧分配低噪声、目标/未来帧分配高噪声，在保持扩散训练稳定性的同时强制因果（action/history-conditioned）预测。
2. 介于全序列扩散与逐 token 自回归（teacher forcing）之间，可灵活控制每帧的去噪程度。
3. 在视频世界模型中用于做块级（chunk-level）滚动预测。

## 代表工作
- [[PROWL]]: 世界模型用块级扩散强迫做因果视频预测，历史帧低噪声、目标 chunk 高噪声，损失只施加在目标 chunk 上

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[World Model]]
- [[Teacher Forcing]]
