---
type: concept
aliases: [捷径流匹配, Shortcut FM, 捷径 Flow Matching]
---

# Shortcut Flow Matching

## 定义

[[Flow Matching]] 的加速变体，通过单步回归 + 自一致性 bootstrap 训练去噪器，使模型在推理时仅需极少（如 4）Euler 步即可生成高质量样本。

## 数学形式

训练目标同时优化：
1. **单步回归**：直接从噪声 $x_\sigma$ 回归干净目标 $x_1$
2. **自一致性**：多步展开与单步预测之间的一致性约束

推理时通过 $S$ 步 Euler 积分：
$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, a, c)$$

其中额外引入"捷径条件" token 编码当前噪声水平 $\sigma$ 和步长 $d$，使模型感知推理步数。

## 核心要点

1. 相比标准 Flow Matching 的 50-100 步 NFE，捷径版本通常 4 步 Euler 即可
2. 通过 shortcut-conditioning token 告知模型当前采样步长，实现步数自适应
3. 单步回归提供初始化，自一致性保证多步质量收敛

## 代表工作

- [[MMBench2]]：将 Shortcut Flow Matching 用于视觉世界模型动力学建模（Frans et al., 2025）
- [[Dreamer 4]]：同样基于此目标训练动力学模型

## 相关概念

- [[Flow Matching]]
- [[Rectified Flow]]
- [[Consistency Model]]
- [[世界模型]]
