---
type: concept
aliases: [ACT, Action Chunking Transformer]
---

# ACT (Action Chunking Transformer)

## 定义

由 Tony Zhao 等人提出的双臂模仿学习框架（ALOHA 工作的核心策略），用 Transformer encoder-decoder 一次性预测**一段动作块** $a_{t:t+k}$，配合 CVAE 训练得到的 latent style code 处理多模态演示。

## 数学形式

$$
\pi_\theta(a_{t:t+k} \mid o_t, z), \quad z \sim q_\phi(z \mid \text{demo})
$$

## 核心要点

1. **Action Chunking**：一次预测 $k$ 步动作（通常 $k=100$），大幅减少 compounding error；
2. **Temporal Ensemble**：每步用多个历史预测的滑动平均，平滑轨迹；
3. **CVAE 训练**：用 latent code $z$ 编码示范的"风格"，推理时令 $z=0$；
4. 是 ALOHA / [[Diffusion Policy]] 双臂任务的强基线；
5. [[AR-VLA]] 的 specialist 版本 AR-Actor 与 ACT 同尺寸（4 层、hidden=512）。

## 代表工作

- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (RSS 2023)
- [[AR-VLA]]：在 ALOHA cube/insert 任务上超越 ACT

## 相关概念

- [[Action Chunking]]
- [[Diffusion Policy]]
- [[Visuomotor Policy]]
- [[ALOHA]]
