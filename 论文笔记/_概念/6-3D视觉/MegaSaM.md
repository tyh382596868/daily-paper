---
type: concept
aliases: [MegaSaM, MegaSAM]
---

# MegaSaM

## 定义

从单目视频中估计相机轨迹和稠密几何的方法，可视为大规模训练的 SfM/SLAM 模型。在 [[WBench]] 中被用作"用生成视频反推相机轨迹"的关键依赖，使**任何范式生成的视频**都可以与目标轨迹做 [[Absolute Trajectory Error|ATE]] 对比。

## 核心要点

1. **输入**: 单目视频流
2. **输出**: 每帧 6-DoF 相机位姿 + 稠密深度 / 重建
3. **应用价值**: 让导航评测**范式无关**——文本驱动模型也能被测出"实际走了什么路线"
4. **替代方案**: COLMAP / DROID-SLAM / DPVO（精度可能不如 MegaSaM 对生成视频鲁棒）

## 代表工作

- [[WBench]]: 用 MegaSaM 实现 Navigation Score 的跨范式统一评测

## 相关概念

- [[Absolute Trajectory Error]]
- [[相机投影]]
- [[Navigation Score]]
- [[6-3D视觉]]
