---
type: concept
aliases: [GelSight Tactile Sensor, 凝胶视觉触觉传感器]
---

# GelSight

## 定义
GelSight 是 MIT 开发的一类基于视觉的触觉传感器，通过弹性凝胶介质和内置摄像头捕捉接触面的高分辨率几何形变，实现精细触觉感知。

## 数学形式
表面法线 $\mathbf{n}$ 从图像梯度 $(p, q)$ 恢复：
$$\mathbf{n} = (-p, -q, 1) / \sqrt{p^2 + q^2 + 1}$$

深度图重建用泊松方程求解：
$$\nabla^2 z = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y}$$

## 核心要点
1. 输出高分辨率 RGB 图像，通过光度立体法重建接触面 3D 形貌
2. 可达亚毫米级分辨率，适合精细操作任务
3. GelSight Mini（小型化版）可安装在机器人手指尖
4. 属于图像型触觉传感器（image-type），与阵列型/状态型传感器并列

## 代表工作
- Yuan et al. (2017): GelSight 原始论文
- [[FTP-1]]: 跨传感器触觉基础策略，包含 GelSight 数据

## 相关概念
- [[AnySkin]]
- [[LEAP]]
- [[Allegro Hand]]
