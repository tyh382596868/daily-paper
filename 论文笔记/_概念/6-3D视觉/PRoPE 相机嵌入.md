---
type: concept
aliases: [PRoPE, Projective RoPE]
---

# PRoPE 相机嵌入

## 定义

把相机内外参表示成 4×4 的 **lifted projective matrix** $\widetilde{P}_i$，再以 block-diagonal 形式作用到 self-attention 的 $Q,K,V$ 上，使 attention 显式感知 token 所属帧的相机几何关系。是 [[RoPE]] 在相机条件上的几何推广。

## 数学形式

$$
\widetilde{P}_{i}=\begin{bmatrix}[K_{i}\;0]\,T_{i}^{cw}\\ e_{4}^{\top}\end{bmatrix}\in\mathbb{R}^{4\times 4}
$$

$$
\mathrm{Attn}_{\mathrm{PRoPE}}(Q,K,V)=D^{\mathrm{PRoPE}}\odot\mathrm{Attn}\!\left((D^{\mathrm{PRoPE}})^{\top}\odot Q,\,(D^{\mathrm{PRoPE}})^{-1}\odot K,\,(D^{\mathrm{PRoPE}})^{-1}\odot V\right)
$$

## 核心要点

1. 不破坏 [[RoPE]] 的相对位置假设，**只新增相机几何信号**；
2. 与 [[Plücker Embedding]] 互补：PRoPE 走 attention 算子路线，Plücker 走 token 拼接路线；
3. 投影矩阵形式保留了"齐次坐标 + 投影变换"的几何意义，比简单 MLP 编码外参更稳定。

## 代表工作

- [[minWM]]: 在 [[Wan2.1]] / HY1.5 双向基座上注入 PRoPE，得到相机可控的多步教师，再蒸馏成 few-step AR 学生。

## 相关概念

- [[RoPE]]
- [[Plücker Embedding]]
- [[相机投影]]
- [[相机可控微调]]
