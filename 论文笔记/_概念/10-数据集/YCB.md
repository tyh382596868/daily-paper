---
type: concept
aliases: [YCB Object Set, YCB-Video, YCBV, YCB 数据集]
---

# YCB

## 定义

YCB (Yale-CMU-Berkeley) 物体数据集是机器人操作研究的标准基准，包含 ~77 个日常物品（罐头、餐具、玩具、工具等）的高精度 3D 模型、纹理网格、物理属性测量数据，被广泛用于抓取、操作、6D 位姿估计、机器人世界模型研究。

## 数据组成

- **3D 模型**: 高分辨率 mesh + 纹理 UV
- **物理参数**: 质量、摩擦系数、惯性张量
- **真实图像**: RGB-D 多视图，含位姿标注
- **变体**:
  - **YCB-Video (YCBV)**: 21 个物体的视频序列，重点用于 6D 位姿跟踪
  - **YCB-M**: 多视图重建专用版本

## 在仿真中的使用

将 YCB 网格导入 [[ManiSkill]]、PyBullet、Isaac Gym 等仿真器，配合刚体物理引擎产生可控的操作场景；也是 MRO-GWM 等[[Action-Conditioned World Model|动作条件世界模型]]的训练资产。

## 应用场景

1. 6D 物体位姿估计基准 (PoseCNN, FoundationPose 等)
2. 机器人抓取与操作策略学习
3. 多物体刚体动力学世界模型 (MRO-GWM)
4. 仿真到现实迁移评测

## 代表工作

- **MRO-GWM** (2026): 用 YCB 65 物体子集构造 Gaussian 世界模型训练集
- **PoseCNN** (2018): 6D 位姿估计基准
- **DexYCB** (2021): 手物交互数据集

## 关联概念

- [[ManiSkill]]: 常用 YCB 资产的仿真器
- [[3D Gaussian Splatting]]: 可用 YCB 物体重建多物体 GS 场景
- [[Action-Conditioned World Model]]: YCB 是评测多物体动力学预测的常用资产
