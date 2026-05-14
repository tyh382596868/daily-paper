---
type: concept
aliases: [Resampler, Auto-Encoder Resampler, Perceiver Resampler]
---

# Resampler

## 定义

Resampler 是一类用一组**可学习查询**（latent queries）通过交叉注意力把变长/高维特征压缩为**固定长度紧凑 token 序列**的模块，常用作大模型的输入瓶颈。源自 Perceiver / Flamingo，被 [[DAWN]] 等近期工作扩展为 **可双向解码** 的 Auto-Encoder 形式。

## 数学形式

$$
z = R(u) \in \mathbb{R}^{N \times d}
$$

其中 $u$ 是 dense feature（长度可变），$z$ 是固定的 $N$ 个 token。AE 形式还要求存在解码器使 $\hat{u} = R^{-1}(z) \approx u$。

## 核心要点

1. **固定长度输出**: 下游模块计算量可控，与输入分辨率/帧数解耦
2. **可学习查询为瓶颈**: 查询数 $N$ 直接决定信息瓶颈宽度（[[DAWN]] 用 $N=16$ 即可）
3. **AE 变体可双向**: 既能压缩也能反解，方便 latent rollout 后再可视化
4. **PerformanceLatency Tradeoff**: 增大 $N$ 性能增益小但延迟陡增（[[DAWN]] Table 4 验证）

## 代表工作

- Perceiver / Perceiver IO: 原始提出
- Flamingo: 在 VLM 中作为视觉瓶颈
- [[DAWN]]: 扩展为 Auto-Encoder Resampler，作为潜世界 token 瓶颈
- BLIP-2 Q-Former: 同思想的另一变体

## 相关概念

- [[Transformer]]
- [[交叉注意力]]
- [[潜空间世界模型]]
