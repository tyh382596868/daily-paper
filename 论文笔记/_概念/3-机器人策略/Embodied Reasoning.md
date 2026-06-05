---
type: concept
aliases: [Embodied Reasoning, 具身推理]
---

# Embodied Reasoning

## 定义

Embodied Reasoning 指机器人/具身智能体在执行动作前进行的**显式中间推理**，通常以自然语言子任务、空间关系、物理预测等形式呈现。是机器人版的 [[Chain-of-Thought|CoT]]。

## 核心要点

1. **显式中间表征**: 子任务文本、affordance 标注、未来轨迹
2. **长程任务收益最大**: 复杂任务必须分解才能稳定执行
3. **可解释性**: 中间推理可供人观察、调试
4. **代价**: 需要标注或自监督生成中间步骤

## 代表工作

- [[WLA]]: 用 [[Language Expert]] 输出子任务文本
- [[CoT-VLA]]: 早期 VLA + CoT 工作
- [[Embodied CoT]]: 系统化 CoT for robotics
- [[ECoT]]: Open-source embodied CoT 数据集

## 相关概念

- [[Chain-of-Thought]]
- [[Language Expert]]
- [[Long-Horizon Manipulation]]
