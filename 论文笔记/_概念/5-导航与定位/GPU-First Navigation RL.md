---
type: concept
aliases: [GPU 优先导航 RL]
---

# GPU-First Navigation RL

## 定义
将机器人导航的强化学习训练全程在 GPU 上执行，消除 CPU-GPU 通信瓶颈，实现秒级策略训练。

## 数学形式
无核心公式，为系统设计创新。

## 核心要点
1. 仿真环境、轨迹采集、策略更新全部 GPU 并行
2. 消除 CPU-GPU 数据传输延迟
3. 首次实现 20 秒内完成导航策略训练

## 代表工作
- [[FlashNav]]: 提出 GPU-First 导航训练框架

## 相关概念
