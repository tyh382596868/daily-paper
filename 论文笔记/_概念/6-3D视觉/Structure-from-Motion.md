---
type: concept
aliases: [Structure-from-Motion, SfM, 运动恢复结构]
---

# Structure-from-Motion

## 定义
Structure-from-Motion (SfM) 是从多视角 2D 图像同时恢复场景三维结构与相机位姿的经典几何视觉技术。

## 核心要点
1. 通过特征匹配、对极几何与捆绑调整 (bundle adjustment) 联合估计 3D 点云和相机外参。
2. 输出可作为几何先验，约束生成模型满足空间一致性。
3. 在 [[CoME]] 中，空间长期记忆专家用 SfM 把历史观测编码为几何结构，在视觉歧义或回环环境中消歧。

## 代表工作
- [[CoME]]: 空间记忆专家用 SfM 编码历史几何

## 相关概念
- [[相机投影]]
- [[相机位姿评估指标]]
- [[VGGT]]
