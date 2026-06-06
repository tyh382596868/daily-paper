---
type: concept
aliases: [Vision-Embodied Brain]
---

# VeBrain

## 定义
VeBrain 是面向 embodied agent 的多模态大模型，强调"视觉感知 + 具身推理"的一体化能力，常用于 robotics / AR / autonomous driving 场景下的视觉问答、空间理解和动作建议。在 streaming spatial 类 benchmark 中作为开源 baseline 出现。

## 核心要点
1. **定位**：embodied-flavored VLM，关注具身场景下的空间-时间推理
2. **能力侧重**：first-person 视频理解、空间一致性、out-of-view 推理
3. **与 [[RoboBrain2]] 区别**：RoboBrain2 更偏 manipulation/规划，VeBrain 更偏 streaming 感知

## 代表工作
- VeBrain 系列论文（具体见 [[OVO-S-Bench]] 引用）
- 应用：被 [[OVO-S-Bench]] 作为评测对象

## 相关概念
- [[RoboBrain2]]
- [[InternVL]]
- [[OVO-S-Bench]]
