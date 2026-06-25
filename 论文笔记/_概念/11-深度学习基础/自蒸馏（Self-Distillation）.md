---
type: concept
aliases: [Self-Distillation, 自蒸馏, self-distillation]
---

# 自蒸馏（Self-Distillation）

## 定义

不引入外部教师模型，由模型自身（或其历史版本/变体）生成教师分布，再对自身做知识蒸馏。

## 数学形式

$$
\mathcal{L}_{SD} = \mathrm{KL}(p_{\theta_{teacher}}(\cdot|x) \,\|\, p_\theta(\cdot|x))
$$

教师 $p_{\theta_{teacher}}$ 可以是：模型的历史快照（EMA）、通过扰动/采样得到的邻近分布、优势加权的近端分布等。

## 核心要点

1. **无模态鸿沟**: 教师与学生在同一模态操作，避免跨模态信号失效。
2. **灵活扰动**: 可通过修改 logits、采样策略等方式构造近端教师，无需训练独立教师模型。
3. **在线适应**: 在 RL 设置中，可将奖励信号注入自蒸馏过程，转化为密集监督。

## 代表工作

- [[ROAD-VLA]]: 优势引导自蒸馏，将稀疏奖励转化为 token 级 KL 蒸馏目标
- [[Knowledge Distillation]]: 标准知识蒸馏（有外部教师）

## 相关概念

- [[Knowledge Distillation]]
- [[KL 散度]]
- [[优势函数（Advantage Function）]]
