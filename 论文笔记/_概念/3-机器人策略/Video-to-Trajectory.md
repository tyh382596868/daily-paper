---
type: concept
aliases: [V2T, Video-to-Action, 视频转轨迹]
---

# Video-to-Trajectory

## 定义

把像素空间的视频片段反解为机器人可执行的末端 6-DoF 轨迹与夹爪指令的流水线。

## 核心步骤

1. **区域掩膜初始化**：[[Grounding DINO]] + [[SAM2]] 取末端 / 物体掩膜
2. **2D 跟踪**：[[CoTracker3]] 得到逐帧像素坐标
3. **深度估计 + 3D 提升**：单目视频深度模型（如 [[DVD Depth]]）+ 相机内参反投影
4. **姿态恢复**：[[Kabsch Alignment]] 解析旋转
5. **夹爪推断**：基于物体–末端相对运动 + [[Gripper Action Prior|先验表]]

## 数学形式

输出 $\tau = \{(p_t, R_t, g_t)\}_{t=1}^T$，其中 $p_t \in \mathbb{R}^3$，$R_t \in \mathrm{SO}(3)$，$g_t \in \{0,1\}$。

## 代表工作
- [[Dream-exe]]：完整开源 V2T 流水线
- [[Dreamitate]]
- [[3DFlowAction]]

## 相关概念
- [[CoTracker3]]
- [[Grounding DINO]]
- [[Kabsch Alignment]]
- [[End-Effector Trajectory]]
