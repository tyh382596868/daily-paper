---
type: concept
aliases: [Intent-based VLA]
---

# IntentVLA

## 定义
一种将高层意图推理和低层动作执行解耦的 VLA 框架，先预测操作意图（物体、目标状态、动作类型），再据此生成精确动作序列。

## 核心要点
1. 意图模型：预测"做什么"（意图表示）
2. 动作策略：预测"怎么做"（基于意图的条件动作）
3. 解耦使意图可跨任务复用，提升 OOD 泛化

## 代表工作
- 与 LUCID、APT 等强调 action representation 的工作同类

## 相关概念
- [[VLA]]
- [[UniVLA]]
- [[APT]]
