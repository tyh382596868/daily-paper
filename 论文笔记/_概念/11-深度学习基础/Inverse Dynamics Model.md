---
type: concept
aliases: [IDM, 逆动力学模型, Inverse Dynamics]
---

# Inverse Dynamics Model

## 定义

给定观测序列反推产生该序列的动作的模型，是[[世界模型]]的对偶 — 世界模型是 "动作 → 未来观测"，IDM 是 "观测序列 → 动作"。

## 数学形式

$$
p(a_{t+1:t+k} \mid o_{t:t+k})
$$

## 核心要点

1. 在 "predict-then-act" 解耦架构中扮演翻译器：把生成的未来视频翻译为可执行动作
2. 对于"无动作视频"数据训练机器人非常关键（人类操作视频、互联网视频）
3. 比生成式 [[策略]] 简单，因为只需要回归 / 分类动作而无需建模动作分布
4. 也是 [[RLA-WM]] 等方法中的核心组件

## 代表工作

- UniPi: 首个用 video plan + IDM 的方案
- VidMan: 引入 mask 关注动作相关区域
- [[VPP]]: 提取视频潜变量代替原始像素
- [[RobotWM-Survey]] 中归类为 "IDM 风格" 范式

## 相关概念

- [[世界模型]]
- [[Controllable 世界模型]]
- [[策略]]
- [[RobotWM-Survey]]
