---
type: concept
aliases: [DiffAero]
---

# DiffAero

## 定义

**DiffAero** 是面向四旋翼/无人机的开源 [[Physics Simulator|物理仿真器]]，基于 [[PyTorch]] 实现，提供**可微分四旋翼动力学**与**GPU 并行深度渲染**。

## 关键特性

- 全 PyTorch，自动微分（用于 [[SHAC]] 等可微策略训练）
- GPU 并行成千上万个环境
- 深度相机渲染
- 训练吞吐量级（[[MAD]] 论文中实测）：$1.21 \times 10^5$ env-step/s

## 在 [[MAD]] 中的扩展

作者扩展 DiffAero，加入 **[[占用栅格图|OGM]] / [[可见性栅格图|VGM]] 在线生成模块**，单卡吞吐 $4.84 \times 10^8$ voxel/s，使 latent 重建监督能与策略训练同步。

## 关联概念

- [[Physics Simulator]]
- [[Sim-to-Real]]
- [[PyTorch]]
- [[Gazebo]]：DiffAero 用于训练，Gazebo + [[PX4]] 用于评测
