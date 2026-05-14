---
type: concept
aliases: [Unified Video-Action]
---

# UVA

## 定义

单 backbone 统一架构的早期代表，构造**联合视频-动作潜空间**，再用模态特化解码器分别还原视频帧和动作。

## 核心要点

1. 训练时在 $[z^v; z^a]$ 共享潜空间联合去噪
2. 推理时可只调用动作 decoder（跳过视频生成），节省成本
3. [[RobotWM-Survey]] Section 3.3 中"单 backbone 统一"范式的开创性工作之一
4. 启发了后续 UWA、VideoVLA、VideoPolicy 等同类工作

## 代表工作

- Li et al., 2025c: UVA 原始论文

## 相关概念

- [[DreamZero]]
- [[Cosmos-Policy]]
- [[策略]]
- [[世界模型]]
- [[RobotWM-Survey]]
