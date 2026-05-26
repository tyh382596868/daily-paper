---
type: concept
aliases: [Absolute Trajectory Error, ATE, 绝对轨迹误差]
---

# Absolute Trajectory Error

## 定义

衡量估计轨迹与真值轨迹之间整体偏差的标准指标，常用于 SLAM、VO、视觉导航等任务评测。先对齐两条轨迹（通常用 Horn 方法做相似变换对齐），再计算每帧位姿差的均方根 (RMSE)。

## 数学形式

$$
\mathrm{ATE} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|\,t_i^{\mathrm{est}} - t_i^{\mathrm{gt}}\,\|_2^2}
$$

## 核心要点

1. **对齐后再算**: 估计轨迹与真值通常需先做 Sim(3) 对齐
2. **单位**: 米（或归一化）
3. **配套指标**: RPE（Relative Pose Error）评估短程一致性
4. **在生成评测中的应用**: [[WBench]] 用 [[MegaSaM]] 估计生成视频的相机轨迹后，与目标轨迹算 ATE 作为 [[Navigation Score]] 主项

## 代表工作

- TUM RGB-D / KITTI Odometry 等经典 SLAM benchmark
- [[WBench]]: 把 ATE 引入视频生成评测

## 相关概念

- [[MegaSaM]]
- [[Navigation Score]]
- [[5-导航与定位]]
