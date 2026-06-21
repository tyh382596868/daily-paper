---
type: concept
aliases: [导航世界动作模型]
---

# Navigation World Action Model

## 定义
将世界模型预测与扩散 Transformer 导航策略结合，用预测的未来视觉状态驱动目标条件导航。

## 数学形式
$$a_t = \pi(o_t, \hat{o}_{t+1:t+H})$$
其中 $\hat{o}$ 为世界模型预测的未来观测。

## 核心要点
1. 世界模型预测未来视觉状态作为导航目标表示
2. 扩散 Transformer 将预测状态转化为导航动作
3. 支持语言指定的目标条件导航

## 代表工作
- [[NavWAM]]: 提出 Navigation World Action Model

## 相关概念
