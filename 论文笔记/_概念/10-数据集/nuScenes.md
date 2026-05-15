---
type: concept
aliases: [nuScenes]
---

# nuScenes

## 定义

nuScenes 是由 Motional（前 nuTonomy）发布的大规模多模态[[自动驾驶]]数据集，包含 1000 个 20 秒驾驶片段，覆盖波士顿与新加坡的城市/郊区场景，是端到端自动驾驶研究的事实标准基准之一。

## 核心要点

1. **传感器**: 6 路相机环绕（360°）+ 1 LiDAR + 5 雷达 + GPS/IMU
2. **标注**: 23 类 3D 边界框 + 8 类属性，2 Hz 关键帧标注
3. **规模**: 1000 scenes × 20s ≈ 5.5 小时驾驶
4. **端到端规划协议**: 以 1s/2s/3s 的 **L2 trajectory error** 与 **collision rate** 评估
5. **常见 baseline**: ST-P3, UniAD, VAD, GenAD, BEV-Planner, World4Drive, [[DAWN]]

## 代表工作（在 nuScenes 上的 L2 / Coll. 进展）

- UniAD: 1.03m / 0.31%
- VAD: 0.72m / 0.23%
- GenAD: 0.52m / 0.19%
- World4Drive: 0.50m / 0.16%
- [[DAWN]]: **0.33m / 0.11%**（当前最佳）

## 相关概念

- [[自动驾驶]]
- [[NAVSIM]]
- [[UniAD]]
- [[World4Drive]]
