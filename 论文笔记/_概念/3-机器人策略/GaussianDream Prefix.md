---
type: concept
aliases: [GaussianDream Prefix, GD Prefix, Z_t^GD]
---

# GaussianDream Prefix

## 定义

[[GaussianDream]] 提出的核心表征：一段 $1024 \times 2048$ 维的 token 序列 $Z_t^{GD}$（$32\times32$ 空间 grid），同时编码当前帧的 3D 几何与短期时序动力学。训练时被多个解码 head 共享（重建、未来预测、动作生成），推理时作为唯一传递给动作 head 的"世界模型摘要"。

## 数学形式

$$
Z_t^{GD} = F_\omega(o_{t-K:t}, Q^{GD}) \in \mathbb{R}^{1024 \times 2048}
$$

其中 $F_\omega$ 是 [[VGGT]] backbone + [[TGE]] 时序融合模块的复合，$Q^{GD}$ 是可学习 query。

## 核心要点

1. **统一表征**: 让动作 head、当前重建 head、未来预测 head 共享同一段 prefix，实现"训练时多任务，推理时纯动作"
2. **空间结构**: $32\times32$ grid 保留了空间位置信息，便于后续 Gaussian decoder 做空间上采样
3. **不对称设计的核心**: 推理时只输出 prefix → action，不解码高斯，因此与 baseline VLA 同等推理速度
4. **作为 plug-in**: 理论上可挂接到任意 VLA backbone 之后
5. **维度选择**: 2048 维与 [[Pi05|π₀.₅]] 主干隐藏维度匹配

## 代表工作

- [[GaussianDream]]: 提出该 prefix 作为 3D 世界模型与 VLA 之间的"接口表征"

## 相关概念

- [[GaussianDream]]
- [[TGE]]
- [[VGGT]]
- [[3D Gaussian Splatting]]
- [[VLA]]
