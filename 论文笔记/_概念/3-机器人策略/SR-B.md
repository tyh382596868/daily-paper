---
type: concept
aliases: [Binary Success Rate, 二值成功率]
---

# SR-B (Binary Success Rate)

## 定义

机器人任务评测中最终是否完成任务的 0/1 平均值。

## 数学形式

$$
\mathrm{SR\text{-}B} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[s_i = \text{success}]
$$

## 核心要点

1. 行业标准，简单直观
2. 长程任务下方差大、区分度低（很多模型全 0）
3. 常与 [[SR-P]] 并用以提供细粒度信号

## 代表工作
- [[Dream-exe]]
- [[LIBERO]] / [[CALVIN]] / [[RoboCasa]] 等标准 benchmark

## 相关概念
- [[SR-P]]
- [[Dream.exe Benchmark]]
