---
type: concept
aliases: [TGE, Temporal Gaussian Evolver, 时序高斯演化器]
---

# TGE (Temporal Gaussian Evolver)

## 定义

[[GaussianDream]] 提出的时序融合模块。基于多层 attention，将冻结 [[VGGT]] backbone 提取的多帧 3D-aware 特征与可学习的 GaussianDream query token 融合，输出统一的 prefix 表征 $Z_t^{GD}$，同时承载空间几何与时序动力学信息。

## 核心要点

1. **结构**: 12 个 attention block × 8 head，每块串联：
   - 帧内 self-attention（query + patch token 联合）
   - 跨帧 temporal attention（同一空间槽跨 3 帧）
   - MLP（4× channel expansion）
2. **维度变换**: query $Q^{GD}$ 从 2048 维投影到 512 维进入 TGE，输出再投影回 2048 维
3. **输入**: 3 帧 [[VGGT]] patch features（$32\times32 \times 512$ × 3）+ learnable query
4. **输出**: 当前帧的 token 子集作为 GaussianDream prefix
5. **作用**: 让 prefix 同时编码"我在哪"（空间）和"我刚从哪儿来"（时序）

## 代表工作

- [[GaussianDream]]: 提出 TGE 作为时序融合模块，连接 VGGT 与高斯解码 head

## 相关概念

- [[VGGT]]
- [[3D Gaussian Splatting]]
- [[GaussianDream Prefix]]
- [[GaussianDream]]
