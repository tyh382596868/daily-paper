---
type: concept
aliases: [Dream.exe, Dream-exe Benchmark, 可执行视频生成基准]
---

# Dream.exe Benchmark

## 定义

一种把 [[Video Generation|视频生成模型]] 输出转成机器人轨迹并在仿真器里执行、用任务成功率取代视觉打分的评测基准。

## 核心要点

1. 8 个模型（闭源前沿 + 开源 + 机器人专用）× 101 任务 × 3 难度等级
2. 四阶段流水线：视频生成 → VLM 视觉评估 → [[Video-to-Trajectory|V2T]] → [[MuJoCo]] 闭环执行
3. 同时报告 SR-B（[[SR-B|二值]]）与 SR-P（[[SR-P|进度]]）成功率以及 HSD/DYN/NDTW 轨迹相似度
4. 关键发现：视觉合理性与任务成功率 Spearman 相关系数仅 -0.03

## 代表工作
- [[Dream-exe]]：原始基准论文

## 相关概念
- [[Video-to-Trajectory]]
- [[Executable Video Generation]]
- [[VBench]]
- [[Physical Plausibility]]
- [[RoboCasa]]
