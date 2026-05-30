---
type: concept
aliases: [GROOT, GR00T, GR-00T]
---

# GROOT

## 定义
NVIDIA 推出的人形机器人通用基础模型项目（Generalist Robot 00 Technology），是一族面向人形机器人的 VLA 模型 + 数据 + 仿真生态的合集。

## 核心组件
- **GROOT 模型**：基于多模态 LLM 骨干的 VLA，输入视觉 + 语言 → 输出动作
- **Isaac Sim + Isaac Lab**：仿真训练环境
- **Project GR00T 数据**：大规模人形机器人 demonstration 数据

## 在 VLA 生态中的地位
- 与 [[OpenVLA]]、π₀、RDT 同属"基础 VLA"梯队
- NVIDIA 路线代表"sim-first + 仿真到真机"的人形机器人路径
- 后续工作（如 [[Gaze2Act]]、ControlVLA）常以 GROOT 作为骨干或对照

## 相关概念
- [[OpenVLA]]、[[VLA]]、[[IsaacLab]]
