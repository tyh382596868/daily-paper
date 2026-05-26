---
type: concept
aliases: [SAM2, Segment Anything 2]
---

# SAM2

## 定义

Meta 提出的统一图像与视频分割基础模型，是 [[Segment Anything]] 的视频扩展版。支持点 / 框 / 掩码提示，能在视频中跟踪并分割任意主体。

## 核心要点

1. **核心能力**: 单帧分割 + 视频时序跟踪一体化
2. **关键组件**: 记忆注意力（Memory Attention），在帧间传递目标特征
3. **输入提示**: 点、框、初始掩码
4. **在评测中的应用**:
   - [[WBench]] 用 SAM2 跟踪主体重心，计算 Perspective Consistency（视角切换时主体重心是否稳定）
5. **常见替代**: Cutie, XMem, Track-Anything

## 代表工作

- [[WBench]]: 用于 Perspective Consistency (C.4) 指标
- 视频编辑、机器人操作中物体跟踪标配

## 相关概念

- [[DINOv2]]
- [[CLIP]]
- [[6-3D视觉]]
