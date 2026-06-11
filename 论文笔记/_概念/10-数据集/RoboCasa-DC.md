---
type: concept
aliases: [RoboCasa-DC, Demo-Conditioned RoboCasa]
---

# RoboCasa-DC

## 定义

RoboCasa-DC 是 RoboCasa 的 demo-conditioned 扩展版本，提供 Franka Panda 机械臂执行轨迹与对应 GR-1 人形机器人演示视频的 episode 配对数据，支持 one-shot demo-conditioned VLA 的可复现评测。

## 数学形式

$$
\mathcal{D} = \{(V^{\text{demo}}_i, \tau^{\text{robot}}_i, l_i)\}_{i=1}^{N}
$$

其中 $V^{\text{demo}}$ 为 GR-1 人形演示视频，$\tau^{\text{robot}}$ 为对应 Franka Panda 执行轨迹，$l$ 为语言任务描述。

## 核心要点

1. **四评测设置（2×2）**：同体/跨体 × 类别均衡/精度敏感
2. **跨体配对**：每条机器人轨迹均对应同一任务的 GR-1 人形演示，支持跨 embodiment 评测
3. **精度敏感分类**：特别标注需要精确定位小目标的困难任务子集
4. 基于 RoboCasa 家庭操作场景（厨房、客厅等）

## 代表工作

- [[SeeTraceAct]]: 提出并开源 RoboCasa-DC，作为主要评测 benchmark

## 相关概念

- [[RoboCasa]]
- [[Demo-Conditioned VLA]]
- [[Cross-Embodiment Learning]]
- [[GR-1]]
- [[Franka Panda]]
