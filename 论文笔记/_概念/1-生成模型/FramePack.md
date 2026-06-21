---
type: concept
aliases: [FramePack 历史压缩, FramePack-style memory layout]
---

# FramePack

## 定义
一种面向自回归视频生成的历史帧压缩策略，对近期历史帧保持高分辨率表示，对远期历史帧按时间距离指数级降低分辨率（最远约 32× 压缩），在控制 KV 缓存内存的同时保留近期上下文细节。

## 数学形式

$$
r(w) = r_{\max} \cdot \alpha^{w}
$$

其中 $r(w)$ 为距当前窗口 $w$ 步的历史帧分辨率，$\alpha < 1$ 为衰减系数。

## 核心要点

1. **近密远疏**: 最近几个窗口以原始分辨率保存，时间越久压缩率越高
2. **内存可控**: 长时序推理中 KV 缓存大小近似为常数，而非线性增长
3. **可插拔设计**: 作为历史压缩模块可与不同骨干架构结合

## 代表工作
- [[Yume]]: 首次提出 FramePack 式历史压缩的视频世界模型
- [[BiWM]]: 将 FramePack 集成为可插拔历史压缩方案之一

## 相关概念
- [[Bidirectional Autoregressive Diffusion]]: FramePack 常用于双向自回归生成
- [[交互式世界模型]]: FramePack 解决长时序推理的内存瓶颈
