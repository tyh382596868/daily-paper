---
type: concept
aliases: [夹爪先验, Gripper Schedule Prior]
---

# Gripper Action Prior

## 定义

基于任务类型预设的夹爪开合时序先验，用于在缺少显式信号时推断 close/open 时刻。

## 核心要点

1. 典型规则：物体被抬起前 1-2 帧关闭、放下时打开；按键 / 旋钮始终闭合
2. 在 video-to-action 流水线中很关键——光流 / 跟踪难以直接给出夹爪状态
3. [[Dream-exe]] 在论文 Table 7 列出按交互模式的先验表

## 代表工作
- [[Dream-exe]]：显式表格化夹爪先验

## 相关概念
- [[Video-to-Trajectory]]
- [[End-Effector Trajectory]]
- [[Action Chunking]]
