---
type: concept
aliases: [Hypernet, 超网络, 元网络]
---

# Hypernetwork

## 定义

Hypernetwork（超网络）是一类**用一个网络生成另一个网络权重**的元学习架构。给定条件 $z$，超网络 $h_\phi(z) \to \theta$ 输出"主网络" $f_\theta$ 的全部或部分参数；从而把"输入 → 输出"的映射变成"条件 → 函数族"的映射。

## 数学形式

$$
\theta = h_\phi(z), \quad y = f_\theta(x)
$$

或对部分参数做调制：

$$
\hat{\mathbf{W}} = \mathbf{W} \odot (1 + \gamma(z)), \quad \hat{\mathbf{b}} = \mathbf{b} + \beta(z)
$$

后者即 [[Grouped Hyper-Modulation]] / FiLM-like 调制。

## 核心要点

1. **条件化函数族**：把任务/上下文编码成参数，避免拼接条件到输入；
2. **轻量化**：常用乘性/加性调制（而非全权重重生成）以控制参数量；
3. **与 [[SIREN]] 配合**：超网络生成 SIREN 各层 $\omega$ / $\phi$ 调制系数，可表达任意频谱；
4. **风险**：纯权重生成会增加优化难度，调制式更稳定。

## 代表工作

- **HyperNetworks** (Ha et al., 2017)：开创性工作
- **FiLM** (Perez et al., 2018)：乘性 + 加性的简化调制范式
- **HyperVLA / HyperTASR**：把 hypernet 用于 VLA / 任务自适应
- [[NIAF]]：MLLM 作为 SIREN 的"频谱调制超网络"

## 相关概念

- [[Grouped Hyper-Modulation]]
- [[SIREN]]
- [[Neural Implicit Representation]]
