---
type: concept
aliases: [DL3DV-10K, DL3DV Dataset]
---

# DL3DV

## 定义

大规模真实场景视频数据集（约 10K+ 场景），为每个场景提供多视角图像 + 精确相机参数，可用于 3D 重建、新视角合成与相机条件视频生成。

## 数学形式

不适用（数据集类资源）。

## 核心要点

1. 真实拍摄场景，覆盖室内外多种环境；
2. 提供精确相机内参与外参，是相机条件训练的**高质量 pose 来源**；
3. 在视频世界模型训练中，常用做"3D 重建 + 按指定轨迹重渲染"以构造**带真值相机轨迹**的训练对。

## 代表工作

- [[minWM]]: 用 DL3DV 做重建-重渲染，构造可靠的相机条件训练数据，作为 [[SpatialVID-HQ|SpatialVid]] 失败路线的替代。

## 相关概念

- [[3D Gaussian Splatting]]
- [[Structure-from-Motion]]
- [[Plücker Embedding]]
