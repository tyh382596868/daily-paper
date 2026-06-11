---
type: concept
aliases: [Whole-Body VLA, whole-body loco-manipulation VLA]
---

# WholeBodyVLA

## 定义
面向仿人机器人全身 loco-manipulation 控制的 VLA 框架，将高层语言-视觉理解与低层全身动作生成统一在单一模型中，实现步态与操作的协同控制。

## 核心要点
1. 传统 VLA 仅控制手臂/末端执行器，WholeBodyVLA 扩展到整个身体（腿 + 躯干 + 手臂）
2. 核心挑战：全身自由度高（>30 DOF），步态和操作的时间尺度不同，需要协调调度
3. 接口设计至关重要：VLA 输出应该是任务级语义命令还是关节级轨迹？
4. [[HANDOFF]] 通过「任务空间接口」解决此问题：VLA/规划器输出紧凑的任务空间命令，全身控制器负责展开为关节命令

## 代表工作
- [[HANDOFF]]：多教师 KL 蒸馏，任务空间接口，在 Unitree G1 上实现 SOTA 速度追踪
- WholeBodyVLA（ICLR 2026）：统一潜在 VLA 用于全身 loco-manipulation 控制

## 相关概念
- [[VLA]]（基础框架）
- [[KL 蒸馏]]（HANDOFF 使用的训练方法）
- [[MoE]]（Mixture-of-Experts，HANDOFF 的学生策略架构）
- [[sim-to-real]]（全身控制的部署路径）
