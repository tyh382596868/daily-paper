---
type: concept
aliases: [Adam with Weight Decay, Decoupled Adam]
---

# AdamW

## 定义

Adam 的解耦权重衰减版本,把 $L_2$ 正则从梯度中剥离,作为独立的权重衰减项作用在参数上,是深度学习训练的事实标准[[优化器]]。

## 数学形式

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1} \right)
$$

## 核心要点

1. **元素级二阶矩归一化**: 每个参数独立调步长,与矩阵几何结构无关。
2. **解耦权重衰减**: $\lambda \theta_{t-1}$ 不经过 $\hat{v}_t$ 的归一化,与 Adam 中的 $L_2$ 正则不等价。
3. **默认超参**: $\beta_1 = 0.9, \beta_2 = 0.95\sim0.999, \varepsilon = 10^{-8}$。
4. **优势**: 鲁棒、调参容易、几乎所有 Transformer 训练默认选择。
5. **局限**: 不利用矩阵秩结构,被 [[Muon]] / [[Pion]] 等[[矩阵感知优化器]]在特定场景超越。

## 代表工作

- [[Pion]]: 在 VLA/RLVR 后训练里把 AdamW 作为强基线对比

## 相关概念

- [[Muon]]
- [[Pion]]
- [[预训练]]
