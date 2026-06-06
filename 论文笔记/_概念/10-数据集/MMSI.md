---
type: concept
aliases: [MMSI-Bench, Multi-Modal Spatial Intelligence Benchmark]
---

# MMSI-Bench

## 定义
MMSI-Bench（Multi-Modal Spatial Intelligence Benchmark）是评估多模态大模型空间智能能力的 benchmark，覆盖距离/方向估计、物体关系、3D 几何、跨视图推理等多个空间任务，用于横向对比 VLM / MLLM 在 embodied 场景的空间理解能力。

## 核心要点
1. **任务类型**：单图距离/朝向估计、多图物体关系、视频空间一致性问答
2. **评估维度**：metric 估计、相对关系、3D 几何理解
3. **常被对比**：与 [[VSI]]、SpatialVLM、OmniSpatial、TopViewRS 等同类 benchmark 一起报告
4. **意义**：space intelligence 是 embodied agent 的瓶颈能力，benchmark 帮助定位模型短板

## 代表工作
- Yang et al., "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025
- 应用：[[OVO-S-Bench]] 把 MMSI 作为参照 benchmark 之一

## 相关概念
- [[VSI]]
- [[OVO-S-Bench]]
- [[InternVL]]
