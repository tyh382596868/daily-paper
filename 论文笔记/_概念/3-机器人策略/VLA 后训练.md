---
type: concept
aliases: [VLA post-training, VLA fine-tuning, VLA reinforcement fine-tuning]
---

# VLA 后训练

## 定义
在预训练 Vision-Language-Action 模型基础上，通过偏好优化、强化学习或在线微调等方法进一步提升策略在目标任务上的成功率和鲁棒性的训练范式。

## 数学形式
常见目标（基于偏好优化）：
$$\mathcal{L} = -\mathbb{E}_{(a_w, a_l)} \left[ \log \sigma \left( r(a_w) - r(a_l) \right) \right] + \lambda \cdot \mathcal{L}_{\text{proximal}}$$

## 核心要点
1. 预训练 VLA（如 π₀、OpenVLA）提供强泛化的语义理解，但直接部署到新任务成功率不足
2. 主流后训练方法：SFT（有监督微调）、DAgger（在线模仿）、offline RL、偏好优化（DPO/RLHF）
3. flow-matching 动作头与 autoregressive 动作头需要不同的后训练目标：Flow-DPO/FlowPRO vs. 标准 DPO
4. 关键挑战：真实环境奖励设计困难；成功/失败标注成本高；分布偏移（distribution shift）

## 代表工作
- [[FlowPRO]]：无奖励离线偏好优化，专为 flow-matching VLA 设计
- [[ForesightFlow]]：势能引导的 best-of-K 推理改进
- [[PAPO-VLA]]：基于优先级感知偏好优化的 VLA 后训练

## 相关概念
- [[VLA]]（基础模型）
- [[π₀]]（常见 base policy）
- [[Flow-DPO]]（flow-matching 版 DPO）
- [[RLHF]]（更通用的人类反馈强化学习）
- [[DAgger]]（在线模仿学习后训练方法）
