---
type: concept
aliases: [Cross-View Attention, 跨视角注意力, Multi-View Attention]
---

# Cross-View Attention

## 定义

Cross-View Attention 是多视角输入场景下，在**不同相机 / 视角的 token 之间**显式做 attention 的机制。常与时间维 attention 交替施加（"时间 ↔ 视角"双轴注意力），保证多视角输出在几何 / 物体 ID / 运动上一致。

## 核心要点

1. **典型实现**: latent token reshape 成 $[T, V, H \cdot W, C]$，沿 $V$ 维（视角）做 self-attention
2. **几何一致**: 7 相机环视生成时，确保同一物体在重叠 FOV 区域的外观 / 位置一致
3. **运动一致**: 同一 agent 在不同 view 中的运动模式不冲突
4. **交替施加**: 与时间维 attention 交替（时间 → 视角 → 时间 → 视角 …）
5. **替代方案**: 也可以一次性把 (T, V) 展平成 1D 做 full attention，但代价更高

## 代表工作

- [[X-Foresight]] Vision Renderer: 在 latent token 上沿时间 / 视角轴交替施加自注意力
- [[X-World]]: 7 相机环视视频生成
- MagicDrive / Panacea 等 multi-view driving scene generation

## 相关概念

- [[自注意力]]
- [[X-World]]
- [[Vision Renderer]]
