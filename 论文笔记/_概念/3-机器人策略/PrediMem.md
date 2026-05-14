---
type: concept
aliases: [PrediMem]
---

# PrediMem

## 定义

[[RoboMemArena]] 论文提出的带记忆 [[VLA]] 方法，采用 [[双系统架构|S1/S2 双系统]]，由 [[Qwen3-VL]] 高层规划器负责关键帧选择与子任务分发，低层 [[Pi05|π₀.₅]] 风格的策略负责动作执行；训练时附加 [[预测编码头]] 作为辅助目标。

## 核心要点

1. **双系统异步**：S2 @ 1.06 Hz，S1 @ 3.40 Hz，约 2.92:1 频率比
2. **[[关键帧记忆库|记忆库]]**：最近缓冲 5 帧 + 无上限关键帧缓冲
3. **关键帧抽取**：物理交互锚点（夹爪开闭）∪ 运动学拐点（速度阈值/方向突变）
4. **[[预测编码头|预测编码]] 辅助**：训练时预测下一帧 [[ViT]] 表征，推理时移除
5. 在 RoboMemArena 上 TSR 38.5%，真机 52%，显著超越 [[MemER]]

## 数学形式

预测编码损失：

$$
\mathcal{L}_{\text{Pre}} = \mathrm{MSE}(\hat{Z}_{t+1}, \mathrm{sg}(Z_{t+1})) + (1 - \cos(\hat{Z}_{t+1}, \mathrm{sg}(Z_{t+1})))
$$

总损失：$\mathcal{L}_{S2} = \mathcal{L}_{\text{text}} + 0.1\,\mathcal{L}_{\text{Pre}}$

## 代表工作

- [[RoboMemArena]]: 提出 PrediMem，2026 arXiv

## 相关概念

- [[双系统架构]]
- [[关键帧记忆库]]
- [[预测编码头]]
- [[Pi05]]
- [[Qwen3-VL]]
- [[MemER]]
