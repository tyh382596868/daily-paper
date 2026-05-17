---
type: concept
aliases: [一致性蒸馏]
---

# Consistency Distillation

## 定义
把多步扩散模型蒸馏成 1-4 步采样器的经典方法，通过强制 $f_\theta(z_t, t) = f_\theta(z_{t-\Delta}, t-\Delta)$ 的 endpoint 一致性来训练。代价是破坏原始 ODE 的 test-time scaling 行为。

## 数学形式
(待补充)

## 核心要点
1. (待补充)

## 代表工作
- (待补充)

## 相关概念
- (待补充)
