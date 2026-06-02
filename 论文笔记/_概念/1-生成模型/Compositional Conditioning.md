---
type: concept
aliases: [组合式条件, Compositional Conditioning]
---

# Compositional Conditioning

## 定义

把视频/图像生成的条件信号**沿物理意义解耦**为多路独立输入（如：机器人运动学、场景背景、物体外观），让模型在生成时可对每一路独立替换，从而实现组合泛化（compositional generalization）。

## 数学形式

$$
p_\theta(o \mid c_1, c_2, \ldots, c_K) \quad \text{with each } c_k \text{ semantically disentangled}
$$

## 核心要点

1. 与单一文本/单图条件相反 — 显式地拆分"做什么 / 在哪做 / 用什么做"
2. 每个条件分支可通过不同机制注入（通道拼接、cross-attention、extended self-attention）
3. 训练时随机替换某一路即可获得 disentanglement，无需额外正则
4. [[RoboDream]] 是机器人数据合成方向的典型应用

## 代表工作

- [[RoboDream]]: $(v_{rob}, I_s, I_o, \ell)$ 四路条件
- [[Controllable 世界模型]] 一族工作

## 相关概念

- [[Compositional World Models]]
- [[Controllable 世界模型]]
- [[扩展自注意力]]
