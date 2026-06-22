---
type: concept
aliases: [等变动作头, Equivariant DiT, SO(2)-Equivariant Flow-Matching DiT]
---

# EquiActor

## 定义

EquiVLA 的动作头模块，首个 SO(2)-等变 Flow-Matching Diffusion Transformer，将标准 DiT 的所有可学习层替换为 G-可操控层（G-steerable layers），使整个动作去噪过程满足精确 SO(2) 等变性。

## 数学形式

**动作表示分解**：

$$
g \cdot \mathbf{a}_t = \begin{cases}
(\rho_1^3 \oplus (\rho_1 \oplus \rho_0) \oplus \rho_0)(g)\, \mathbf{a}_t & \text{绝对控制} \\[4pt]
P^{-1}(\rho_0^6 \oplus \rho_1^4 \oplus \rho_2)(g)\, P\, \mathbf{a}_t & \text{相对控制}
\end{cases}
$$

**等变注意力分数**（几何内积）：

$$
\langle q, k \rangle = \sum_g q[g] \cdot k[g]
$$

## 核心要点

1. **G-可操控层**：所有 Q/K/V 投影、FFN、动作编解码器均用正则到正则的等变线性层替换
2. **动作分解**：按物理语义将 7 维动作分解为不可约表示直和——向量量用 $\rho_1$，标量量用 $\rho_0$
3. **等变注意力**：正则表示正交性保证几何内积为 G-不变量，可用作注意力分数
4. **从头训练**：G-可操控约束与标准 DiT 不兼容，无法复用预训练权重
5. **精确等变**：当配合不变的 VLM context token 时，满足精确（非近似）SO(2) 等变

## 代表工作

- [[EquiVLA]]: EquiActor 的提出论文，16 层等变 DiT，192 通道，基于 escnn 库实现

## 相关概念

- [[SO(2)等变性]]
- [[扩散变换器]]
- [[Flow Matching]]
- [[EquiPerceptor]]
