---
type: concept
aliases: [门控残差融合, Gated Fusion]
---

# Gated Residual Fusion（门控残差融合）

## 定义

一种通过可学习门控网络动态调节辅助信息注入量的残差融合机制。门控输出决定辅助特征对主特征的贡献比例，可通过负偏置初始化实现"渐进式开放"——训练初期关闭记忆通道，随训练推进逐渐开放。

## 数学形式

$$
\hat{X} = X + \sigma(g([X;\, X'])) \cdot X'
$$

其中 $X$ 为主特征，$X'$ 为辅助（记忆）特征，$g(\cdot)$ 为门控 MLP，$\sigma(\cdot)$ 为 Sigmoid 激活。

## 核心要点

1. **负偏置初始化**: 初始化 $g$ 的偏置为负值，使 $\sigma(\cdot) \approx 0$，训练早期抑制辅助信息避免干扰
2. **残差结构**: 若门控完全关闭（$\sigma \to 0$），退化为原始主特征 $X$，保证最坏情况不会劣于无记忆基线
3. **关键性验证**: KEMO 消融实验中，去掉门控融合后 TSR 从 9/12 降为 0/12，是所有组件中影响最大的

## 代表工作

- [[KEMO]]: 将 Gated Residual Fusion 用于将历史关键帧记忆 token 注入 VLA 当前帧特征

## 相关概念

- [[Masked Cross-Attention]]
- [[Cross-Attention]]
- [[Adaptive Gating]]
- [[Event Saliency Score]]
