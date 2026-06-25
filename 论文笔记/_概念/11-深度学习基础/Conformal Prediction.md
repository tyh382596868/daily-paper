---
type: concept
aliases: [Conformal Inference, 保形预测]
---

# Conformal Prediction

## 定义
分布无关的统计推断框架，在任意黑盒模型上构造有限样本覆盖率保证的预测区间，无需分布假设，只需可交换性条件。

## 数学形式
给定校准集 $\{(x_i, y_i)\}_{i=1}^n$ 和显著性水平 $\alpha$，构造预测集 $C(x_{n+1})$ 使得：
$$P(y_{n+1} \in C(x_{n+1})) \geq 1 - \alpha$$

## 核心要点
1. **无分布假设**：只要求数据可交换（exchangeability），不需要高斯或独立同分布
2. **有限样本保证**：覆盖率保证是精确的，不是渐近的
3. **black-box 兼容**：可以包裹任意模型（神经网络、决策树等）
4. 在世界模型中：用校准集误差统计为 rollout 可信步数提供上界认证

## 代表工作
- [[ConformalWM]]: 用 conformal prediction 为 equivariant world model 认证可信 horizon

## 相关概念
- [[DreamerV3]]
- [[JEPA]]
