---
type: concept
aliases: [Geometric Consistency, 几何一致性]
---

# Geometric Consistency

## 定义

衡量视频生成结果在不同帧之间是否保持几何一致的指标。典型做法：估计每帧深度（如用 Depth Anything 3），把帧 $i$ 的像素根据深度和相机位姿重投影到帧 $j$，计算重投影位移误差。

## 数学形式

$$
\mathrm{GC} = 1 - \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \frac{\|\pi(D_i, K, T_{i\to j}) - p_j\|_2}{d_{\max}}
$$

## 核心要点

1. **依赖**: 单目深度估计模型（Depth Anything 3）+ 相机位姿（[[MegaSaM]]）
2. **失败模式检测**: 拉伸、变形、几何漂移
3. **关联指标**: [[Photometric Consistency]] 同样基于 warp，但比较像素颜色而非几何位置
4. **WBench 中**: 作为 C.5 (Consistency 第 5 项)

## 代表工作

- [[WBench]]: C.5 Geometric Consistency
- 各类 NeRF / 3DGS 评测中也常用

## 相关概念

- [[Photometric Consistency]]
- [[Spatial Consistency]]
- [[相机投影]]
- [[6-3D视觉]]
