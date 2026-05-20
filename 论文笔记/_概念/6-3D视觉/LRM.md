---
type: concept
aliases: [LRM, Large Reconstruction Model, 大重建模型]
---

# LRM

## 定义
Large Reconstruction Model（大重建模型）是一类前馈式的 3D 重建模型，单次前向即可从一张或少量图像预测 3D 表示（如三平面、NeRF、3DGS），无需逐场景优化。

## 核心要点
1. 用大规模 3D 数据预训练 transformer，把「重建」从逐场景优化变成一次前向推理，速度快几个数量级。
2. 输出通常是 triplane / 3D Gaussian / 隐式场，可直接渲染新视角。
3. 与逐场景优化的 NeRF/3DGS 互补：LRM 给快速初始化或直接产物，优化方法做精修。

## 代表工作
- [[PanoWorld]]: 用面向 metric-scale 多房间 360° 输入的前馈全景 LRM，把生成的全景抬升成 3D

## 相关概念
- [[3D Gaussian Splatting]]
- [[世界模型]]
