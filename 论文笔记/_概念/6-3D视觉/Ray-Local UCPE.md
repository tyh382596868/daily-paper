---
type: concept
aliases: [Unified Camera Pose Embedding, UCPE, 光线局部相机位姿嵌入]
---

# Ray-Local UCPE

## 定义

一种相机条件嵌入策略：对每个 latent 像素，先用相机内外参反投射出光线方向，构造局部正交基 $\mathbf{D} = [\mathbf{x}, \mathbf{y}, \mathbf{z}]$，再把它与 [[RoPE]] 拼接后施加在 Query 的"几何通道"上，让 [[自注意力]] 可以按相机视角对齐地检索几何信息。

## 数学形式

$$
\widetilde{\mathbf{Q}}^c_i = (\mathbf{D}_i^\top \oplus \mathrm{RoPE}_i) \mathbf{Q}^c_i
$$

其中 $\mathbf{D}_i = [\mathbf{x}, \mathbf{y}, \mathbf{z}]$，$\mathbf{z} = \mathrm{norm}(\mathbf{d}_{t,s})$ 为该像素的世界系光线方向，$\mathbf{x}, \mathbf{y}$ 为正交补。

## 核心要点

1. 工作在 latent-frame 速率（粗时间分辨率，但几何方向感强）
2. 只作用于 Query 的"几何通道"，避免污染纹理/语义通道
3. 与 [[RoPE]] 做块对角拼接，复用其 sin/cos 时空位置编码
4. 与 [[Plücker Embedding]] 形成"粗/细"双分支：UCPE 粗时间 + Plücker 细时间

## 代表工作

- [[SANA-WM]]: 粗分支 UCPE + 细分支 Plücker 实现 6-DoF 相机控制；消融显示 UCPE 单独已把旋转误差从 16.93° 降至 7.73°

## 相关概念

- [[Plücker Embedding]]
- [[RoPE]]
- [[相机投影]]
