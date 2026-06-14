---
type: concept
aliases: [自我-他者区分, Self-Other Recognition, 自他区分]
---

# Self-Other Distinction（自我-他者区分）

## 定义

在共享场景中，机器人（或智能体）识别哪个视觉实体属于自身身体的能力，无需预设身份标签或运动学模型。

## 数学形式

利用本体感知特征与视觉 mask 特征之间的余弦相似度，结合 soft 注意力加权：

$$
\alpha_k = \frac{\exp(s_k/\tau_a)}{\sum_{\ell}\exp(s_\ell/\tau_a)}, \quad s_k = \bm{f}_{\text{state}}^{\top} \bm{f}_{\text{image},k}
$$

## 核心要点

1. 灵感来自发展心理学——婴儿通过本体感知-视觉时间同步性建立自我认知
2. 本体感知与自身视觉观测天然同步，与他者观测不同步，可作为区分信号
3. 通过[[InfoNCE 损失|InfoNCE 对比学习]]无监督训练，无需任何身份标注

## 代表工作

- [[HumanoidSelfModel]]: 提出注意力引导对比学习框架，在人形机器人上实现 >99.5% 自我-他者区分准确率

## 相关概念

- [[本体感知]]
- [[注意力引导对比学习]]
- [[Self-Supervised Learning]]
- [[InfoNCE 损失]]
