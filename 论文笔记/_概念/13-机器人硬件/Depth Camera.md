---
type: concept
aliases: [Depth Camera, 深度相机, RGB-D Camera, RealSense D435i, RealSense]
---

# Depth Camera

## 定义

**深度相机**：除 RGB 通道外还输出像素级深度图的传感器，常见原理有结构光（Kinect v1）、ToF（Kinect v2、PMD）、主动 IR 立体（**Intel RealSense D435i**）、被动双目（ZED）等。

## 典型规格（RealSense D435i, [[MAD]] 中使用）

| 项 | 值 |
|----|----|
| 帧率 | 30 Hz |
| 深度 FOV | 87° × 58° |
| 量程 | ~0.1–5 m（户外/远场会显著退化） |
| IMU | 内置 6-DoF（可外发做 [[VIO]]） |
| 接口 | USB 3.0 |

## 优劣

**优**: 直接给几何，绕过单目深度估计的尺度模糊；轻量、便宜；机载常用。

**劣**:
- 户外阳光下 IR 立体退化
- 远场 (>5 m) 噪声急升
- 玻璃 / 镜面 / 黑色吸光物体易"穿透"或缺失
- 18×32 等极低分辨率下细障碍（电线、细枝）易丢

## 在机器人/无人机中的角色

- 局部建图（[[占用栅格图|OGM]]）的主要观测源
- [[Sim-to-Real]] 中 sim 渲染的深度通常比 RGB 更易迁移（深度对光照鲁棒）
- 极低分辨率（[[MAD]] 用 18×32）可降低 latency 又适配 RSSM 编码器输入

## 代表工作

- [[MAD]]：18×32 深度图 + RSSM
- 大量 visual navigation / [[Diffusion Policy]] 使用 D435i 作机载传感

## 关联概念

- [[VIO]]
- [[SLAM]]
- [[Sim-to-Real]]
- [[占用栅格图]]
