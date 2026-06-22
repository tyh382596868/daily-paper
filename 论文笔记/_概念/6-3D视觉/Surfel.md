---
type: concept
aliases: [Surface Element, 表面元素, surfel]
---

# Surfel

## 定义

Surfel（Surface Element，表面元素）是一种 3D 场景表示的基本单元，用有向圆盘（位置 + 法向量 + 半径）近似连续表面，常用于点云压缩、3D 重建和视频记忆索引。

## 数学形式

基础 surfel：$s_k = (p_k, n_k, r_k, \mathcal{I}_k)$

扩展（W-VMem）：$s_k = (p_k, n_k, r_k, t_k, m_k)$

- $p_k \in \mathbb{R}^3$: 3D 位置
- $n_k \in \mathbb{R}^3$: 表面法向量（单位向量）
- $r_k > 0$: 半径（表面覆盖范围）
- $\mathcal{I}_k$: 观测帧索引集合（VMem 中）
- $t_k$: 创建/更新时间步（W-VMem 扩展）
- $m_k \in \{0, 1\}$: 操作物标志（W-VMem 扩展）

## 核心要点

1. **紧凑表示**: 相比稠密点云，surfel 用有向圆盘近似局部表面，更紧凑且支持法向插值
2. **几何可见性判断**: 给定相机位姿，可快速判断 surfel 是否可见（法向与视线方向对齐、深度范围内）
3. **视频记忆锚点**: [[VMem]] 和 [[W-VMem]] 用 surfel 索引历史观测帧，实现几何感知的帧检索
4. **动态扩展**: W-VMem 增加时间戳和语义标志，使 surfel 能区分时间阶段和操作对象

## 代表工作

- [[VMem]]: 用 surfel 索引视频记忆实现跨 chunk 一致性
- [[MemWorld]]: 提出 W-VMem，将 surfel 扩展为 4D 时间感知表示用于机器人操作

## 相关概念

- [[3D Gaussian Splatting]]
- [[W-VMem]]
- [[VMem]]
- [[6-3D视觉]]
