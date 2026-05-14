---
type: concept
aliases: [GR1, GR-1]
---

# GR-1

## 定义

早期统一 [[VLA]] 模型的代表，联合预测动作与未来图像帧，把"未来图像预测"作为辅助监督信号。

## 核心要点

1. 单一 Transformer backbone 在 token 序列中混合 vision/language/action/future-image
2. 未来图像预测作为辅助任务提升表征质量
3. [[RobotWM-Survey]] Section 3.5 中"显式未来预测"子类的代表
4. 启发了后续 UP-VLA、[[WorldVLA]] 等工作

## 代表工作

- Wu et al., 2024: GR-1 原始论文
- UP-VLA (Zhang et al., 2025c): 把未来预测变成联合训练信号

## 相关概念

- [[VLA]]
- [[WorldVLA]]
- [[DreamVLA]]
- [[世界模型]]
- [[RobotWM-Survey]]
