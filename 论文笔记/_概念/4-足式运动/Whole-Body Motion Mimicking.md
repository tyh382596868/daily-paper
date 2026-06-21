---
type: concept
aliases: [全身运动模仿]
---

# Whole-Body Motion Mimicking

## 定义
人形机器人同时模仿参考运动的躯干运动和末端执行器运动，实现全身协调控制。

## 数学形式
$$\mathcal{L}_{wb} = \lambda_{loco}\mathcal{L}_{joint} + \lambda_{manip}\mathcal{L}_{ee}$$

## 核心要点
1. 统一处理 locomotion 和 manipulation
2. 支持多模态参考输入（视频、MoCap、文字描述）
3. 全身协调是人形机器人的核心挑战

## 代表工作
- [[M3imic]]: 多模态全身运动模仿控制器
- [[HANDOFF]]: 多教师蒸馏的全身控制

## 相关概念
