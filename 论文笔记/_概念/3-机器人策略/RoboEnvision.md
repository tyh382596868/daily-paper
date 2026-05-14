---
type: concept
aliases: []
---

# RoboEnvision

## 定义

Cascaded WAM 方法，用 VLM 把任务拆成子任务指令，**非自回归**地生成 keyframe（子任务终止状态），再插值合成完整视频。

## 核心要点

1. **非自回归生成**: 用关键帧+插值替代逐帧自回归，规避长程漂移。
2. 属于 [[Cascaded WAM|Cascaded WAM]] 的 Explicit / Learned-Action 子类。

## 代表工作

- [[WAM-Survey]] 中作为 keyframe-based cascaded WAM 的代表。

## 相关概念

- [[UniPi]]
- [[World Action Model]]
