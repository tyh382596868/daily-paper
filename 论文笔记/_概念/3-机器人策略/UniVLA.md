---
type: concept
aliases: [UniVLA]
---

# UniVLA

## 定义

UniVLA 是一类用**潜在动作（latent action）**桥接无动作视频与机器人策略的方法：从视频帧对中学一个紧凑潜变量作为"伪动作"标签，再用它训练或条件化策略/视频生成模型。在 [[RLA-WM]] 论文里作为已有潜在动作方法的对比基线。

## 核心要点

1. **潜在动作维度小**（[[RLA-WM]] 论文中报告 $|z|=256$），主要用作模仿学习的代理标签或视频生成条件。
2. **不具预测充分性**：从其潜变量 $z$ 重建未来 DINO token $s_{t+h}$ 严重模糊（[[RLA-WM]] Figure A1），说明潜变量未编码足够的预测信息。
3. **与 RLA 对比**：[[残差潜在动作|RLA]] 强调"从 $z$ 单次前向高保真重建 $s_{t+h}$"这一预测充分性，是 UniVLA / [[AdaWorld]] 这类方法的关键缺失。
4. **从无动作视频学策略**：在 [[RLA-WM]] 的 Table 2 里，UniVLA 平均成功率 28.7%，介于纯 BC（27.2%）与 RLA（35.6%）之间。

## 代表工作

- 作为基线出现在 [[RLA-WM]]（Zhang et al., 2026）
- 同类：[[AdaWorld]]
- **注意**: [[UniVLA-ICLR2026]]（BAAI, Wang et al. 2025）是另一个同名但不同方向的工作，采用统一离散 Token 自回归框架（非潜在动作方法），在 CALVIN/LIBERO/SimplerEnv 上达到 SOTA。

## 相关概念

- [[残差潜在动作]]
- [[AdaWorld]]
- [[World Action Model]]
- [[VLA]]
- [[UniVLA-ICLR2026]]
