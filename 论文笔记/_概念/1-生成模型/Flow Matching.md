---
type: concept
aliases: [流匹配, Conditional Flow Matching, CFM]
---

# Flow Matching（流匹配）

## 定义

Flow Matching 是一种生成模型训练范式，通过回归从噪声到数据的线性插值路径上的速度场来学习生成分布，训练稳定且推理高效。

## 数学形式

线性插值：$x_t = (1-t)\varepsilon + ta$，速度场 $u = a - \varepsilon$

损失：$\mathcal{L}_{\text{flow}} = \mathbb{E}\left[\|f_\theta(x_t, t, c) - u\|_2^2\right]$

## 核心要点

1. 在噪声和目标之间定义直线轨迹，速度场为常数，训练简单
2. 无需离散化 SDE，可用少步 ODE 推理
3. 通过上下文条件 $c$ 实现条件生成

## 代表工作

- [[π₀]]：将 Flow Matching 引入机器人 VLA 动作生成
- [[MolmoAct2]]：使用多流样本损失增强 Flow Matching 训练

## 相关概念

- [[VLA]]
- [[Action Chunking]]
