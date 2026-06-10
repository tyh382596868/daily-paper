---
type: concept
aliases: [InfoNCE Loss, Noise Contrastive Estimation, NCE Loss, 对比噪声估计损失]
---

# InfoNCE 损失

## 定义

InfoNCE（Noise Contrastive Estimation）损失是一种用于表示学习的对比损失函数，通过将同一语义下的正样本对拉近、负样本对推远，在嵌入空间中学习结构化表示。

## 数学形式

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp(\text{sim}(z_k^+, z_k^{++}) / \tau)}{\sum_{m=1}^{N} \exp(\text{sim}(z_k^+, z_m^{++}) / \tau)}
$$

其中 $\text{sim}(\cdot, \cdot)$ 通常为余弦相似度，$\tau$ 为温度参数。

## 核心要点

1. **正样本对**: 同一数据点的两个视角/模态，应在隐空间距离最近
2. **负样本对**: batch 内其他样本，被当作负例推远
3. **温度参数 $\tau$**: 控制分布的尖锐程度，较小的 $\tau$ 使梯度更集中于难负例
4. **与 MI 的关系**: InfoNCE 是互信息下界，最大化 InfoNCE 等价于最大化两个视角之间的互信息

## 代表工作

- [[CPC]]: 最初提出 InfoNCE 的工作（van den Oord et al., 2018）
- [[SimCLR]]: 图像自监督对比学习经典方法
- [[CLIP]]: 图文跨模态对齐中大规模应用
- [[LAD]]: 用于跨机器人本体动作空间的语义对齐

## 相关概念

- [[对比学习]]
- [[Diffusion Policy]]
- [[Cross-Embodiment Learning]]
