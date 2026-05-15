---
type: concept
aliases: [Plücker Coordinates, Plücker Ray, 普吕克坐标]
---

# Plücker Embedding

## 定义

用 6 维 Plücker 坐标 $(\mathbf{d}, \mathbf{m})$ 表示三维空间中一条光线：$\mathbf{d}$ 是方向向量，$\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 是矩向量，编码光线相对原点的位置。该表示对相机原点位置具有平移不变性（沿光线移动 $\mathbf{o}$ 不改变 $\mathbf{m}$），是相机条件控制的常用几何嵌入。

## 数学形式

$$
\bm{\rho}_{r,p} = (\mathbf{d}_{r,p}, \, \mathbf{o}_r \times \mathbf{d}_{r,p}) \in \mathbb{R}^6
$$

## 核心要点

1. 比简单拼接相机外参更几何稳定（无歧义、无奇异性）
2. 像素级 Plücker raymap：每像素一条光线，可表达完整视角信息
3. 在视频扩散模型中常作为 6 通道条件图，通过 zero-init patch embedder 加到 latent
4. 与 [[RoPE]] / UCPE 等 latent-frame 级条件互补

## 代表工作

- [[SANA-WM]]: 细分支以 raw-frame 速率（8 帧每 latent stride）注入 48 维 Plücker 张量
- [[CtrlWorld]]: 用 Plücker 编码做相机控制

## 相关概念

- [[相机投影]]
- [[Ray-Local UCPE]]
- [[RoPE]]
