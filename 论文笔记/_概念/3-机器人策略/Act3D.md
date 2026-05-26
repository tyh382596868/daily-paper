---
type: concept
aliases: [Act3D, Action-3D]
---

# Act3D

## 定义

一种代表性的 V-3D-A 范式机器人操作策略：把多视图 RGB-D 提升为 3D 体素 / 点云特征，再用 Transformer 在 3D 空间中预测末端执行器的位姿。属于"静态 3D 表征 → 动作"的典型代表。

## 数学形式

$$
\pi: \{I_v, D_v\} \to \text{3D Feature Volume} \to a_{ee}
$$

## 核心要点

1. **静态 3D 表征**：仅建模"当前时刻的 3D 几何"，不预测未来
2. **3D 注意力**：在 3D 空间中做 cross-attention，利用度量几何
3. **关键帧动作**：通常输出关键帧位姿，再用 motion planning 插值
4. **位姿鲁棒**：相比 2D 策略对相机位姿变化更鲁棒

## 代表工作

- Act3D: V-3D-A 范式的代表
- [[GAF]] 与之对比，提出加入时间维度的 V-4D-A 范式

## 相关概念

- [[V-4D-A 范式]]
- [[Diffusion Policy]]
- [[3D Gaussian Splatting]]
