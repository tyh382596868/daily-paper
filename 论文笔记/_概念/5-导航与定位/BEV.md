---
type: concept
aliases: [BEV, Bird's-Eye View, 鸟瞰图]
---

# BEV (Bird's-Eye View)

## 定义

BEV（鸟瞰图）是把多视角 / 多传感器数据**几何投影到一个俯视的 2D 平面**上得到的统一表示。在 [[自动驾驶]] 中已成为感知 / 预测 / 规划的标准中介空间——所有相机 / 雷达数据先 lift 到 BEV，再做下游任务。

## 核心要点

1. **统一坐标**: 7 相机、LiDAR、Radar 各自的坐标系都映射到自车为原点的鸟瞰平面
2. **几何友好**: 与车辆动力学、地图、规划自然对齐，比 image-space 更易做轨迹推理
3. **占据栅格 / 语义图**: BEV 上常表示为 occupancy / semantic / lane mask 等多通道张量
4. **辅助损失**: 在 token 序列模型里，BEV 也常作为额外监督信号（如 X-Foresight 的 $\mathcal{L}_{bev}$）
5. **典型分辨率**: $200 \times 200$ 网格、覆盖 $\pm 50$ m

## 代表工作

- [[UniAD]]: 全任务 BEV-centric 端到端驾驶
- [[X-Foresight]]: BEV 作为 $\mathcal{L}_{bev}$ 辅助监督
- BEVFormer / BEVFusion 等经典 BEV perception 工作

## 相关概念

- [[自动驾驶]]
- [[相机投影]]
- [[UniAD]]
