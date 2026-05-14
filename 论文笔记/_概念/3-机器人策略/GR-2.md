---
type: concept
aliases: [GR2]
---

# GR-2

## 定义

[[GR-1]] 的后继，规模和训练数据进一步扩大的 Autoregressive Joint WAM。沿用"视频 + 动作 token AR"路线，预测未来视频帧与动作 chunk。

## 核心要点

1. **更大规模**: 数据 / 参数全面 scale up。
2. **统一 AR 解码**: 视频 token 与动作 token 在同一因果序列里预测。
3. 属于 [[Joint WAM|Joint WAM]] 的 Autoregressive / Explicit-Decoupled 子类。

## 代表工作

- [[WAM-Survey]] Table 2 中列为 Autoregressive 代表。

## 相关概念

- [[GR-1]]
- [[GR-MG]]
- [[World Action Model]]
