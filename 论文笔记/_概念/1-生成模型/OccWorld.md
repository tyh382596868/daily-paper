---
type: concept
aliases: [Occupancy World Model]
---

# OccWorld

## 定义
基于 3D 语义占用（semantic occupancy）表示的自动驾驶世界模型，将场景表示为体素化的语义占用网格，预测未来时刻的占用状态变化。

## 核心要点
1. 体素网格表示：每个体素有类别标签（车辆、行人、道路等）
2. 时序预测：从当前占用预测未来 N 帧的占用演化
3. 用于 collision checking 和路径规划

## 代表工作
- [[GaussianWorld]] — 基于 3DGS 的改进版本
- [[VISA]] — 用 VLM 审计 OccWorld 的语义错误

## 相关概念
- [[GaussianWorld]]
- [[BEV]]
