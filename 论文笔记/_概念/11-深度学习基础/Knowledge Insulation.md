---
type: concept
aliases: [知识绝缘, KI, Stop-Gradient Insulation]
---

# Knowledge Insulation

## 定义
在多任务或跨具身 VLA 微调中，通过 stop-gradient 算子切断动作 loss 对 VLM backbone 的梯度反传，防止专域微调破坏预训练获得的通用语言-视觉知识。

## 数学形式

KI 后训练阶段，VLM 隐状态经过 stop-gradient 后再输入 DiT：

$$
\tilde{H}^{\text{KI}}_{\phi,p} = \mathrm{sg}(H^{\text{KI}}_{\phi,p})
$$

$$
\mathcal{L}_{\text{KI}} = \alpha \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{FAST}} + \sum_j \lambda_j \mathcal{L}_{\text{CE}}^{(j)}, \quad \alpha = 10
$$

其中 $\mathrm{sg}(\cdot)$ 前向传值、反向梯度为零，使 $\mathcal{L}_{\text{FM}}$（flow matching loss）仅更新 DiT 参数，不影响 VLM backbone $f_\phi$。

## 核心要点

1. **问题动机**: 直接端到端微调 VLM + DiT 会导致 VLM 遗忘预训练能力（灾难性遗忘）
2. **Stop-gradient 机制**: 前向计算正常传递，反向传播时梯度在 sg 处被截断为零
3. **联合优化**: $\mathcal{L}_{\text{FAST}}$（自回归 token 损失）仍更新 VLM，维持语言能力；只有 $\mathcal{L}_{\text{FM}}$ 被 sg 阻断
4. **区别于 [[Knowledge Distillation]]**: KI 不需要教师模型，通过结构性梯度截断实现保护

## 代表工作

- [[LabVLA]]: 提出 Knowledge Insulation 机制用于实验室 VLA 后训练

## 相关概念

- [[Knowledge Distillation]]
- [[Flow Matching]]
- [[FAST Action Tokenizer]]
- [[VLA]]
