---
type: concept
aliases: [Beta 分布时间步采样, Beta Timestep Sampling, 非均匀时间步采样]
---

# Beta 分布时间步采样

## 定义

在 [[Flow Matching|流匹配]] / 扩散模型训练中，用 Beta 分布（而非均匀分布）采样时间步 $\tau$，使训练或正则化更聚焦于某些去噪阶段的策略。

## 核心要点

1. 均匀采样让所有去噪阶段被同等对待；Beta 分布可偏置采样到去噪后期（大 $\tau$，接近干净样本）或前期。
2. 形状参数控制偏置方向与强度，配合缩放常数 $s$ 把分布映射到 $[0,1]$。
3. [[RoVLA]] 用 $\operatorname{Beta}((s-\tau)/s; 1.5, 1)$、$s=0.999$，让 [[Evolutionary Consistency|演化一致性]] 约束更多落在接近干净动作的去噪后期。

## 数学形式

$$
p(\tau) = \operatorname{Beta}\!\left(\frac{s-\tau}{s};\; 1.5,\; 1\right), \quad s = 0.999
$$

## 代表工作

- [[RoVLA]]: 用 Beta 分布采样时间步，强化去噪后期的演化一致性约束。

## 相关概念

- [[Flow Matching]]
- [[Evolutionary Consistency]]
- [[扩散变换器]]
