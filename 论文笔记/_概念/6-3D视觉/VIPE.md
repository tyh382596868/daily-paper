---
type: concept
aliases: [VIPE, Video Pose Estimation Engine]
---

# VIPE

## 定义

一种从公开视频自动恢复相机位姿和深度的流水线引擎。原版主要用 Metric3D-Small 做深度估计 + bundle adjustment 做位姿优化；可扩展为度量尺度版本，融合多种深度估计器（[[Pi3X]]、MoGe-2）和每帧独立内参优化。

## 核心要点

1. 输入：纯视频帧；输出：度量尺度相机位姿、深度、内参
2. 在大规模视频数据集（SpatialVID-HQ、MiraData、Sekai 等）上离线标注训练用相机轨迹
3. 改造版本支持：
   - 多帧一致性深度（[[Pi3X]]）
   - 度量尺度补偿（MoGe-2 融合）
   - 每帧独立内参 $(f_x, f_y, c_x, c_y)$
4. 是 [[世界模型]] / 视频生成训练数据标注的关键基础设施

## 代表工作

- [[SANA-WM]]: 改造 VIPE 为度量尺度版本，标注 213K 段训练视频

## 相关概念

- [[Pi3X]]
- [[3D Gaussian Splatting]]
- [[相机投影]]
