---
type: concept
aliases: [交叉熵, Cross-Entropy Loss, CE Loss]
---

# Cross-Entropy

## 定义
衡量预测概率分布与真实标签分布之间差异的损失函数，是分类任务和语言模型训练的标准目标。

## 数学形式

对于 $C$ 类分类，真实标签 one-hot 向量 $y$，预测概率 $\hat{p}$：

$$
\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{p}_c
$$

在语言模型中（next-token prediction），对序列 $z_{1:T}$ 的交叉熵为：

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(z_t \mid z_{<t})
$$

## 核心要点

1. **等价于 KL 散度加常数**：$\mathcal{L}_{\text{CE}}(p, q) = H(p) + D_{\text{KL}}(p \| q)$，最小化交叉熵等价于最小化 KL 散度
2. **与 softmax 结合**：数值稳定的实现通常将 softmax 与 log 合并（log-sum-exp trick）
3. **语言模型训练基础**：GPT、BERT 等均以交叉熵为核心训练目标

## 代表工作

- [[LabVLA]]: 后训练阶段辅助任务损失 $\mathcal{L}_{\text{CE}}^{(j)}$
- [[VLA]]: VLM backbone 语言建模目标

## 相关概念

- [[Next-Token Prediction]]
- [[FAST Action Tokenizer]]
- [[Knowledge Distillation]]
