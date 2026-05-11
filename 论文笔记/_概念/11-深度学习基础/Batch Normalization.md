---
type: concept
aliases: [BatchNorm, BN, 批归一化, Batch Normalization]
---

# Batch Normalization

## 定义

Batch Normalization 在每个 mini-batch 上对每个特征维度独立做归一化（减均值、除方差），再用可学习的仿射参数 $\gamma,\beta$ 恢复表示能力，从而稳定训练、加速收敛、缓解协变量偏移。

## 数学形式

对 batch $\{x_i\}_{i=1}^B$ 的某一维：

$$
\mu = \frac{1}{B}\sum_i x_i,\quad
\sigma^2 = \frac{1}{B}\sum_i (x_i - \mu)^2
$$

$$
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},\quad y_i = \gamma\hat x_i + \beta
$$

推理时使用训练阶段累积的 running mean / variance。

## 核心要点

1. **稳定训练**: 减少梯度爆炸 / 消失，允许使用更大学习率
2. **batch 维统计**: 与 LayerNorm（在 feature 维上归一化）形成互补
3. **在自监督中**: BatchNorm 与对比 / 分布匹配损失天然契合——[[LeWM]] 在 encoder 末端用 BN 替换 LayerNorm，是因为 LayerNorm 会让每个样本特征向量自动归一化为单位 norm，破坏 [[SIGReg]] 想要的各向同性高斯分布
4. **缺点**: batch size 太小时统计不稳；分布式训练时需要同步统计量（SyncBN）

## 相关概念

- [[MLP]]
- [[Transformer]]
- LayerNorm
