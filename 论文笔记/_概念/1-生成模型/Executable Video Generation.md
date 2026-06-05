---
type: concept
aliases: [可执行视频生成, Executable Video, Dream-to-Action]
---

# Executable Video Generation

## 定义

把视频生成模型作为隐式 [[World Model|世界模型]]，要求其产出不仅"看起来合理"，还能被翻译成可在真实 / 仿真环境中执行的动作序列。

## 核心要点

1. 任务成功率作为唯一可证伪的"物理懂不懂"指标
2. 通常配合 [[Video-to-Trajectory]] 或 [[VLA]] decoder 把视频翻译成 action
3. 与传统 [[VBench]] 等视觉指标的差异：plausibility ↑ 不一定意味 success ↑

## 代表工作
- [[Dream-exe]]：评测 8 个主流模型的可执行性
- [[Dreamitate]]
- [[RoboDream]]
- [[DreamVLA]]

## 相关概念
- [[Video Generation]]
- [[World Model]]
- [[Dream.exe Benchmark]]
