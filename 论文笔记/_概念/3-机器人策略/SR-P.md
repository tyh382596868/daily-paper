---
type: concept
aliases: [Progress Success Rate, 进度型成功率]
---

# SR-P (Progress Success Rate)

## 定义

机器人任务执行过程中已达成子检查点比例的连续值（$[0,1]$）。

## 数学形式

$$
\mathrm{SR\text{-}P} = \frac{1}{N}\sum_{i=1}^N \frac{1}{|\mathcal{K}_i|}\sum_{k\in\mathcal{K}_i} \mathbb{1}[\text{checkpoint } k \text{ reached}]
$$

## 核心要点

1. 适合长程任务，提供"接近成功"的连续梯度
2. 与 [[SR-B]] 配合，避免"二值全零"现象
3. 检查点设计需依赖任务先验

## 代表工作
- [[Dream-exe]]：Level 3 长程任务的主要区分指标
- CALVIN long-horizon benchmark

## 相关概念
- [[SR-B]]
- [[Dream.exe Benchmark]]
