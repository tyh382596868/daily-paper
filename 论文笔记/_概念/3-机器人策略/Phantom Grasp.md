---
type: concept
aliases: [幻影抓取, Pseudo-grasp Failure]
---

# Phantom Grasp

## 定义

视频生成模型中常见的失败模式：夹爪未真正闭合却带动物体一起移动，违反接触动力学。

## 核心要点

1. 与"物体悬浮（Levitation）"、"运动学崩溃（Kinematic Breakdown）"并列 [[Dream-exe]] 三大失败模式
2. 反映模型把"视觉相关性"误学成"因果接触"
3. 在视频生成评测中是 plausibility 与 executability 脱节的典型证据

## 代表工作
- [[Dream-exe]]：首次系统化分类此失败

## 相关概念
- [[Physical Plausibility]]
- [[Dream.exe Benchmark]]
- [[Video Generation]]
