---
type: concept
aliases: [Observational Consistency, 观测一致性, OC]
---

# Observational Consistency

## 定义

要求 [[VLA]] 策略在观测（视觉特征、本体状态）受到扰动时仍产生稳定预测的约束，属于[[多一致性约束]]中针对观测扰动变换的一致性。

## 核心要点

1. 缓解"伪视觉理解"问题：策略对视觉/本体感知扰动过于敏感。
2. [[RoVLA]] 对视觉 token 和机器人状态施加 [[对抗扰动]]（沿 $\nabla\mathcal{L}_{\text{EC}}$ 方向），再要求扰动分支预测对齐干净分支。
3. 干净分支用 stop-gradient $\operatorname{sg}(\cdot)$ 包裹作为不动目标，梯度只更新扰动分支，防止污染未扰动分支。
4. 与 [[Evolutionary Consistency|EC]] 协同：EC 提供扰动方向，OC 利用扰动样本强化鲁棒性。

## 数学形式

$$
\mathcal{L}_{\text{OC}} = \frac{1}{2}\sum_{i=1}^{2} \big\| \hat{\mathbf{v}}_{\text{pert}}^{\tau_i} - \operatorname{sg}\!\big(\hat{\mathbf{v}}_{\text{clean}}^{\tau_i}\big) \big\|_2^2
$$

## 代表工作

- [[RoVLA]]: 用对抗扰动 + stop-gradient 对齐实现观测一致性约束。

## 相关概念

- [[多一致性约束]]
- [[对抗扰动]]
- [[Evolutionary Consistency]]
- [[VLA]]
