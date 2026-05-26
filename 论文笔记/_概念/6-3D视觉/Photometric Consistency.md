---
type: concept
aliases: [Photometric Consistency, 光度一致性]
---

# Photometric Consistency

## 定义

衡量视频生成结果在帧间的纹理 / 颜色稳定性。用深度和相机位姿把帧 $i$ warp 到帧 $j$ 的视角，与真实帧 $j$ 比较 PSNR / L1。

## 数学形式

$$
\mathrm{PC} = \mathrm{PSNR}\!\left(I_j,\, \mathrm{Warp}(I_i,\, D_i,\, T_{i\to j})\right)
$$

## 核心要点

1. **与 [[Geometric Consistency]] 配对**: 几何检查位置，光度检查颜色
2. **失败模式检测**: 色彩漂移、光照闪烁、纹理跳变
3. **WBench 中**: 作为 C.6 (Consistency 第 6 项)
4. **与 V.3 Temporal Flickering 区别**: 后者是相邻帧像素抖动，本指标是基于深度的跨帧 warp 比较，更稳健

## 代表工作

- [[WBench]]: C.6 Photometric Consistency
- 多视图合成、NeRF 训练的核心监督信号

## 相关概念

- [[Geometric Consistency]]
- [[6-3D视觉]]
