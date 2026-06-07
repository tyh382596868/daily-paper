---
type: concept
aliases: [几何 Adapter, 几何适配器]
---

# Geometry Adapter

## 定义
一个**轻量 MLP + LayerNorm 模块**，插在 VLM 视觉编码器的某个中间层（如 ViT 第 18 层），把 VLM 自身的视觉 patch token 投影到与 3D 基础模型（如 [[VGGT]]）对齐的特征空间。目的是让 VLM 中间表征获得**低层几何先验**（深度、表面、距离），而**不改 VLM 主干结构**、不增加推理输入。

## 数学形式

$$
\mathcal{F}^{Geo} = \mathrm{LN}(\mathrm{MLP}(\mathcal{F}^{Vis}))
$$

对齐损失：
$$
\mathcal{L}_{geo} = 1 - \mathcal{S}(\mathcal{F}^{3D}, \mathcal{F}^{Geo})
$$

其中 $\mathcal{F}^{3D}$ 来自冻结的 [[VGGT]]，$\mathcal{F}^{Vis}$ 是 ViT 中间层 patch token。

## 核心要点
1. **位置选择**: 通常选 ViT 中段（如第 18 层），太浅几何信号不足，太深已被语义主导
2. **维度桥接**: VLM 表征维度往往 ≠ 3D 基础模型维度，MLP 起桥接作用
3. **训练时使用，推理时保留**: VGGT 推理时丢弃，但 Geometry Adapter 已把几何先验"压"进 ViT
4. **优于显式 3D 输入**: [[3DThinkVLA]] Table 8 显示，隐式 adapter 对齐胜过 PTv3 / [[DP3]] 等 explicit encoder

## 代表工作
- [[3DThinkVLA]]: 首次正式命名 Geometry Adapter，配合 VGGT 对齐
- DenseVLM / 各种 VLM-3D fusion 工作中类似设计

## 相关概念
- [[Latent Alignment]]
- [[VGGT]]
- [[Reasoning Adapter]]
- [[LoRA]]
