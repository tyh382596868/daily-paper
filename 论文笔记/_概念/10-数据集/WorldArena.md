---
type: concept
aliases: [WorldArena Benchmark, P3CScore]
---

# WorldArena

## 定义

专为机器人视频生成质量评估设计的综合基准，基于 RoboTwin 2.0 仿真数据，从物理真实性、3D 精度和可控性三个维度评估生成视频质量，综合指标为 P3CScore。

## 核心要点

1. **6 项评估维度**：
   - **物理真实性**：交互质量（Interaction Quality）、轨迹精度（Trajectory Accuracy）
   - **3D 精度**：深度精度（Depth Accuracy）、透视性（Perspectivity）
   - **可控性**：指令跟随（Instruction Following）、语义对齐（Semantic Alignment）
2. **P3CScore**：Physics、3D、Controllability 三维综合得分（满分约 100）
3. 基于 RoboTwin 2.0 仿真任务，提供标准化的机器人操作视频评估协议
4. 现有最强基线：CogVideoX（71.08），EA-WM 达到 76.60

## 代表工作

- [[EA-WM]]: 在 WorldArena 上达到 SOTA，P3CScore 76.60

## 相关概念

- [[RoboTwin]]
- [[世界模型]]
