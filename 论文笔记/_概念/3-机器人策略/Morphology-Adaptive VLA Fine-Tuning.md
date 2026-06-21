---
type: concept
aliases: [形态自适应 VLA 微调]
---

# Morphology-Adaptive VLA Fine-Tuning

## 定义
针对不同机器人形态对 VLA 模型进行条件化微调，桥接并行夹爪到灵巧手等形态间的操作差距。

## 数学形式
$$\theta^* = \arg\min_\theta \mathcal{L}(\pi_\theta(o, \text{intent}, m), a)$$
其中 $m$ 为形态描述。

## 核心要点
1. 保留预训练 VLA 的语义理解能力
2. 用意图条件化适配不同末端执行器的动作空间
3. 解决 VLA 的形态迁移问题

## 代表工作
- [[Bridging the Morphology Gap]]: 提出形态自适应 VLA 微调方法

## 相关概念
