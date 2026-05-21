---
type: concept
aliases: [Historical Dropout, 历史随机掩码]
---

# Stochastic History Masking

## 定义

训练自回归模型时，对历史 token 序列按一定概率（如 0.6）做**逐 token 随机掩码**，迫使模型在历史被破坏时回退到非历史条件（如 VL prefix），缓解过度依赖历史导致的 [[Causal Confusion Paradox|因果混淆]]。

## 数学形式

$$
\mathcal{L} = \sum_{k} \mathcal{L}\big(x_{H+k} \mid \mathcal{M}_k \odot \mathbf{x}_{\text{past}},\, \Phi(v, l),\, \mathbf{x}_{H:H+k-1}\big)
$$

其中 $\mathcal{M}_k \in \{0,1\}^H$ 是每个 token 独立采样的二元 mask。

## 核心要点

1. **mask rate = 0**：模型学到"复制历史"，验证误差最低但部署成功率为 0；
2. **mask rate = 0.6**：[[AR-VLA]] 中的甜点值，平衡历史利用与 VL 依赖；
3. **mask rate = 1.0**：等价于反应式训练，丢失历史能力；
4. 比 standard dropout 更激进——直接置零整个历史 token。

## 代表工作

- [[AR-VLA]]：揭示 0.6 mask 是甜点，提出"causal confusion paradox"

## 相关概念

- [[Causal Confusion Paradox]]
- [[Dropout]]
- [[Teacher Forcing]]
- [[自回归动作专家]]
