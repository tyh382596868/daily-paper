---
type: concept
aliases: [Dream Zero]
---

# DreamZero

## 定义

单 backbone 联合扩散世界模型 + 策略，采用**autoregressive flow-matching with chunk-wise joint denoising**，把视频与动作 token 在共享潜空间内 chunk 级别地联合去噪。

## 核心要点

1. [[RobotWM-Survey]] Section 3.3 中"单 backbone 统一"范式的代表
2. 用 [[Flow Matching|flow matching]] 替代传统扩散，训练稳定且推理快
3. autoregressive + chunk-wise 设计允许在变长 horizon 下灵活推理
4. 与 [[Cosmos-Policy]] 是同期的 unified policy/WM 代表

## 代表工作

- Ye et al., 2026b: DreamZero 原始论文

## 相关概念

- [[Cosmos-Policy]]
- [[UVA]]
- [[Flow Matching]]
- [[扩散变换器]]
- [[世界模型]]
- [[RobotWM-Survey]]
