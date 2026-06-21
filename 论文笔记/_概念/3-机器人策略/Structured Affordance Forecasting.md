---
type: concept
aliases: [结构化 Affordance 预测]
---

# Structured Affordance Forecasting

## 定义
将 affordance 预测作为 VLA 从感知到动作之间的结构化中间表示，明确编码可交互区域和交互方式。

## 数学形式
$$a = \pi(o, l, \hat{\text{aff}})$$
其中 $\hat{\text{aff}}$ 为预测的 affordance map。

## 核心要点
1. VLA 先预测 affordance map（哪里可以操作、如何操作）
2. 基于 affordance 生成精确的操作动作
3. 提升对新物体和新场景的泛化

## 代表工作
- [[AffordanceVLA]]: 提出 Structured Affordance Forecasting 作为 VLA 中间表示

## 相关概念
