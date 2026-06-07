---
type: concept
aliases: [VIO, Visual-Inertial Odometry, 视觉惯性里程计]
---

# VIO（Visual-Inertial Odometry）

## 定义

**视觉惯性里程计**：通过紧耦合 **相机** 和 **IMU** 数据估计 6-DoF 位姿、速度、IMU 偏差等状态。相比纯视觉 odometry，IMU 提供高频运动先验，对快速运动、纹理稀疏帧鲁棒；相比纯 IMU，相机提供长程漂移修正。

## 数学骨架

状态 $x = [p, v, R, b_a, b_g]$，IMU 预积分 + 视觉重投影残差联合优化：

$$
\min_x \sum_i \|r^{IMU}_i(x)\|^2_{\Sigma_I} + \sum_j \|r^{vis}_j(x)\|^2_{\Sigma_C}
$$

## 两大流派

1. **滤波**: MSCKF、ROVIO，扩展卡尔曼或 IMU 状态滤波
2. **优化**: VINS-Mono、ORB-SLAM3 紧耦合、Kimera——滑窗 BA

## 在无人机中的角色

- 提供高带宽位姿估计（>100 Hz），是机载控制和规划闭环的状态源
- [[MAD]] 中真机部署：VIO 输出位姿/速度作为本体感知向量 $d_t$ 的一部分喂给策略
- 相比专用 motion capture 室更适合户外、密林、灾后场景

## 典型限制

- 长程会有漂移（无 loop closure 时）
- 高速/低纹理/光照剧变会失锁
- IMU 偏差初始化需静止或激励

## 代表工作

- VINS-Mono / VINS-Fusion: 港科大开源代表
- ORB-SLAM3: 视觉/视惯 SLAM 综合方案
- Kimera: 实时语义 SLAM
- OpenVINS: 滤波派代表

## 关联概念

- [[SLAM]]
- [[Depth Camera]]
- [[Sim-to-Real]]
