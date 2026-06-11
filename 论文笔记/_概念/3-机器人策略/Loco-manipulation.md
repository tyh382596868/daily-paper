---
type: concept
aliases: [Loco-Manipulation, 移动操作, 全身操控, Whole-body Control]
---

# Loco-manipulation

## 定义

同时结合移动（locomotion）和操作（manipulation）能力的机器人控制任务，要求机器人在行走/移动的同时完成抓取、搬运等操作，是全身控制（Whole-body Control）的核心挑战。

## 核心要点

1. **高维动作空间**: 腿部（移动）+ 臂部（操作）+ 末端执行器，动作维度通常 >20
2. **协调性要求**: 移动时保持操作稳定，操作时维持移动平衡
3. **接触丰富**: 多接触点（脚与地面、手与物体），动力学复杂
4. **VLA 挑战**: 需要长时程规划（去目标位置 → 操作对象 → 返回），对视觉理解和运动控制要求高

## 代表工作

- [[WholebodyVLA]] (ICLR 2026): 全身移动操作的统一 VLA 框架
- [[UniVLA]]: 提出作为潜在扩展方向的全身操控应用

## 相关概念

- [[VLA]]: 解决 Loco-manipulation 的主流框架
- [[Action Chunking]]: 全身控制中的多步动作预测
- [[MDP]]: Loco-manipulation 任务的形式化框架
