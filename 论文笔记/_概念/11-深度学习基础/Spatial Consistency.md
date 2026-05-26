---
type: concept
aliases: [Spatial Consistency, 空间一致性]
---

# Spatial Consistency

## 定义

视频世界模型评测中衡量**回到起点是否还看到同样场景**的指标。给定闭环轨迹（终点回到起点），比较视频首末帧的感知相似度。

## 数学形式

$$
\mathrm{SC} = \mathrm{DreamSim}(o_0,\, o_T) \quad \text{when } a_{\leq T}\ \text{形成闭环}
$$

## 核心要点

1. **闭环依赖**: 必须设计起止点重合的导航轨迹
2. **常用度量**: DreamSim（学习到的感知距离）/ LPIPS / DINO 余弦
3. **Gated 变种**（[[WBench]] C.2）: 同时检查中间帧最小相似度，避免"虽然回得来但中间漂得很远"
4. **对世界模型的意义**: 是否真正建立了**世界状态**而非纯粹依赖最近帧的渲染惯性

## 代表工作

- [[WBench]]: 提出 C.1 (Spatial Consistency) + C.2 (Gated Spatial Consistency)
- WorldArena / MIND 类基准的核心评测项

## 相关概念

- [[Geometric Consistency]]
- [[Photometric Consistency]]
- [[11-深度学习基础]]
