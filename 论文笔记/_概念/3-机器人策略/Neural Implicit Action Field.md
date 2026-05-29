---
type: concept
aliases: [NIAF, Implicit Action Field, 隐式动作场, 神经隐式动作场]
---

# Neural Implicit Action Field

## 定义

由 [[NIAF]] 提出的动作表示范式：把机器人的整段动作 chunk 抽象为**一个对规范化时间 $\tau \in [-1, 1]$ 的连续函数** $\mathcal{A}(\tau) \in \mathbb{R}^d$，由 [[SIREN]] 参数化，参数本身又由 [[MLLM]] 的 [[Grouped Hyper-Modulation]] 输出条件生成。

## 数学形式

$$
\mathcal{A}: [-1, 1] \to \mathbb{R}^d, \quad \tau \mapsto \mathcal{A}(\tau; \theta(z_{lang}, z_{vis}, z_{state}))
$$

任意时刻 $\tau$ 都可解析求得**位置、速度、加速度、jerk**：

$$
\dot{\mathcal{A}}(\tau) = \frac{2}{T} \nabla_\tau \mathcal{A}, \qquad \dddot{\mathcal{A}}(\tau) = \big(\tfrac{2}{T}\big)^3 \nabla^3_\tau \mathcal{A}
$$

## 核心要点

1. **分辨率与训练完全解耦**：训练用 50 步动作 chunk，推理可直接 query 任意 $K$ 步；
2. **$C^\infty$ 平滑**：来自 [[SIREN]] 的正弦激活，无块间 jerk 尖峰；
3. **解析高阶导数**：[[Impedance Control]] 的 D 项不再需要数值微分；
4. **作为通用动作头**：可替换 [[OpenVLA-OFT]] / [[Pi05]] / [[Florence-2]] 的动作模块。

## 代表工作

- [[NIAF]]：原始提出者，在 [[CALVIN]] / [[LIBERO]] 与真机上达到 SOTA

## 相关概念

- [[SIREN]]
- [[Grouped Hyper-Modulation]]
- [[Neural Implicit Representation]]
- [[Action Chunking]]：被超越的旧范式
- [[Impedance Control]]：直接受益方
- [[Jerk]]
