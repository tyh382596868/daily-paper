---
type: concept
aliases: [VSI-Bench, Visual Spatial Intelligence Benchmark]
---

# VSI-Bench

## 定义
VSI-Bench（Visual Spatial Intelligence Benchmark）是评估多模态模型在视频/序列输入下空间认知能力的 benchmark，强调"从一段视频中推断空间布局、距离、相对位置"等任务，是 embodied agent 空间智能能力的关键测试场。

## 核心要点
1. **输入形态**：连续帧 / egocentric 视频，而非单图
2. **任务设计**：metric distance、relative position、room-layout reasoning、object-counting across frames
3. **难点**：跨帧空间一致性、out-of-view 物体的记忆推理
4. **比较对象**：与 [[MMSI]]、SpatialRGPT、SpatialBench 等横向报告

## 代表工作
- VSI-Bench 系列工作（embodied / spatial reasoning 文献）
- 应用：[[OVO-S-Bench]] 把 VSI 作为参照 benchmark

## 相关概念
- [[MMSI]]
- [[OVO-S-Bench]]
- [[空间记忆专家]]
