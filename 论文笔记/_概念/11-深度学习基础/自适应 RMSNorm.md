---
type: concept
aliases: [Adaptive RMSNorm, AdaRMSNorm, 条件归一化]
---

# 自适应 RMSNorm

## 定义
自适应 RMSNorm（Adaptive RMSNorm）是 RMSNorm 的条件化变体，将外部条件信号（如本体状态）通过小网络映射为仿射参数，对特征进行条件归一化，实现多模态信息融合。

## 数学形式

$$
\mathbf{h}' = \gamma(c) \cdot \mathrm{RMSNorm}(\mathbf{h}) + \beta(c)
$$

其中 $c$ 为条件信号，$\gamma(c)$ 和 $\beta(c)$ 由轻量 MLP 预测。

## 核心要点
1. 通过条件信号 $c$ 预测缩放 $\gamma(c)$ 和偏移 $\beta(c)$，实现特征空间的自适应调整
2. 在 [[FTP-1]] 中用于将本体状态 $\mathbf{s}_t$ 融合进 Transformer 隐藏状态
3. 类似 DiT 中的 AdaLN（Adaptive Layer Norm），但用 RMSNorm 替代 LayerNorm

## 代表工作
- [[FTP-1]]: 用 Adaptive RMSNorm 融合本体状态到动作专家中

## 相关概念
- [[RMSNorm]]
- [[流匹配]]
- [[π0.5]]
