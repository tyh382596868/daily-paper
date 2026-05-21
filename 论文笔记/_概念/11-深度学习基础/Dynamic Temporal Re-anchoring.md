---
type: concept
aliases: [DTR, 动态时序重锚定]
---

# Dynamic Temporal Re-anchoring

## 定义

利用 [[RoPE]] 的**相对位置不变性**，把 VL prefix 的位置编码动态锚定到 "VLM 刚刷新时的当前 timestep"，使得训练短序列（如 25 步）的相对偏移分布与推理长序列（如 500 步）一致，从而实现长序列外推。

## 数学形式

RoPE 的位移不变性：

$$
\text{Score}(q_{m+T},\, k_{n+T}^{VL}) = \text{Score}(q_m,\, k_n^{VL})
$$

只要 $\Delta t = m - n$ 不变，注意力分数就不变。

## 核心要点

1. 解决"训练短、推理长"的位置编码外推问题；
2. 关键是保持 VL prefix 与当前 query 的**相对偏移** $\Delta t$ 与训练时一致；
3. 不需要修改 RoPE 实现，只需要重新分配 VL token 的位置索引；
4. 移除 / 用静态位置编码会使性能崩塌（[[AR-VLA]] 中分别降到 29.2% / 3.1%）。

## 代表工作

- [[AR-VLA]]：让短序列预训练的 AR 专家泛化到 500 步真机 rollout

## 相关概念

- [[RoPE]]
- [[Hybrid KV Cache]]
- [[位移不变性]]
