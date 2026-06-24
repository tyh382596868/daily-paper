---
type: concept
aliases: [Projective RoPE, Projective Relative Position Encoding]
---

# PRoPE

## 定义
**Projective Relative Position Encoding** —— 把相机内参 $K$ 与世界到相机外参 $T^{cw}$ 合并为 $4\times 4$ 投影矩阵并以块对角形式注入到 [[自注意力]] 的 Q/K/V，使两 token 间的相对项化简为相对相机变换。它是把 [[RoPE]] 从平面 2D 旋转推广到任意 3D 投影几何的形式。

## 数学形式

$$
\tilde P_i = \begin{bmatrix} K_i & 0 \\ 0 & 1 \end{bmatrix} T_i^{cw}, \quad
\text{Attn}_{\text{PRoPE}}(Q,K,V) = D^{\text{PRoPE}} \odot \text{Attn}\bigl( (D^{\text{PRoPE}})^\top \odot Q, (D^{\text{PRoPE}})^{-1} \odot K, (D^{\text{PRoPE}})^{-1} \odot V \bigr)
$$

两帧间的相对项归约为 $\tilde P_{i(t_1)}\tilde P_{i(t_2)}^{-1}$，即**相对相机变换**。

## 核心要点

1. **与 RoPE 同构**: 保留 RoPE "通过共轭变换在点积中自动产生相对量"的性质。
2. **不破坏注意力结构**: 不引入 cross-attention 旁路，蒸馏友好。
3. **块对角注入**: 一半通道用于相机几何，一半保留 2D [[RoPE]] 空间编码。

## 代表工作

- [[minWM]]: 用 PRoPE 给 T2V/TI2V 基础模型注入相机条件，并保证 [[非对称 DMD]] 蒸馏全程不破坏控制能力。
- [[G3VLA]]: 将 PRoPE 用于 VLA 多视图跨视图注意力，编码相机内外参定义的投影位置关系，提升机器人操作空间推理能力。

## 相关概念

- [[RoPE]]
- [[自注意力]]
- [[Camera-Controlled 视频生成]]
