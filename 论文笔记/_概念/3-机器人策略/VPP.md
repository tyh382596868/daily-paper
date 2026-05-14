---
type: concept
aliases: [Video Prediction Policy]
---

# VPP

## 定义

IDM 风格解耦架构 — 不生成完整未来帧，而是**提取视频扩散模型的潜在预测特征**作为策略的条件，比像素级中间表示更紧凑高效。

## 核心要点

1. 实验证明：用视频扩散模型的**中间潜变量**比完整解码帧更利于策略学习
2. [[RobotWM-Survey]] Section 3.2 IDM 风格中"压缩中间表示"路线的代表
3. 启发了 Video2Act、MimicVideo、TC-IDM 等使用潜变量的后续工作

## 代表工作

- Hu et al., 2025: VPP 原始论文

## 相关概念

- [[Inverse Dynamics Model]]
- [[视频扩散模型]]
- [[策略]]
- [[RobotWM-Survey]]
