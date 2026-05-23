---
type: concept
aliases: [MoGe-2, Metric MoGe]
---

# MoGe-2

## 定义
微软研究院提出的度量尺度（metric-scale）单目深度估计模型，作为 [[Pi3X]] 之外的额外深度先验。

## 核心要点

1. **度量尺度输出**: 不仅给相对深度，还能直接预测真实世界单位（米）
2. **配合 Pi3X**: 在 [[SANA-WM]] 的 [[VIPE|VIPE 标注引擎]] 中作为锚定模块，把长序列相对深度校正到度量尺度
3. **应用**: 相机轨迹标注、3D 重建、几何感知视频生成
4. **训练数据**: 多源监督，涵盖室内/室外/合成数据

## 代表工作
- [[SANA-WM]]: 与 Pi3X 配合提供度量尺度深度先验
- MoGe / MoGe-2（Microsoft Research）
- [[EvoScene-VLA]]: 作为冻结的 Monocular Depth Teacher，监督 VLA 场景槽的局部几何锚

## 相关概念
- [[Pi3X]]
- [[VIPE]]
- [[相机投影]]
