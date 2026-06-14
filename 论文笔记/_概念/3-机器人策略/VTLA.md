---
type: concept
aliases: [Vision-Tactile-Language-Action, 视觉触觉语言动作模型]
---

# VTLA

## 定义
VTLA（Vision-Tactile-Language-Action）是在 VLA（Vision-Language-Action）基础上增加触觉模态的机器人策略框架，将视觉、触觉、语言指令统一用于动作生成。

## 核心要点
1. 将触觉信号作为额外输入模态补充视觉信息，对接触密集型任务特别有效
2. 面临的核心挑战：不同传感器的触觉信号异构，难以统一表示
3. [[FTP-1]] 提出的通用化 VTLA 范式解决了跨传感器泛化问题

## 代表工作
- [[FTP-1]]: 首个跨异构传感器的通用 VTLA 基础策略
- [[Tactile-VLA]]: 早期 VTLA 工作，绑定特定传感器

## 相关概念
- [[π0.5]]
- [[MTTS]]
- [[触觉传感器]]
- [[Action Chunking]]
