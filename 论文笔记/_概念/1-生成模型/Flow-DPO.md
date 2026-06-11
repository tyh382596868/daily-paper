---
type: concept
aliases: [Flow Matching DPO, Flow-based DPO]
---

# Flow-DPO

## 定义
将 Direct Preference Optimization（DPO）目标适配到 flow-matching 生成模型的方法，通过偏好对（成功/失败演示）直接优化 flow-matching 策略，无需显式奖励函数。

## 数学形式
$$\mathcal{L}_{\text{Flow-DPO}} = -\mathbb{E}_{(x_w, x_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(x_w)}{\pi_{\text{ref}}(x_w)} - \beta \log \frac{\pi_\theta(x_l)}{\pi_{\text{ref}}(x_l)} \right) \right]$$

其中 $x_w$ 为偏好样本（成功演示），$x_l$ 为非偏好样本（失败演示），$\pi_{\text{ref}}$ 为参考策略。

## 核心要点
1. 继承 DPO 无需训练 reward model 的优势，直接从偏好对优化
2. 核心问题：plain Flow-DPO 存在奖励黑客（reward hacking）——隐式奖励幅值无界，策略可能逃离参考分布
3. [[FlowPRO]] 的 RPRO 通过添加近端正则化器解决此问题，锚定隐式奖励绝对幅值

## 代表工作
- [[FlowPRO]]：提出 RPRO 改进 Flow-DPO，在双臂操作任务上超越 Flow-DPO 基线

## 相关概念
- [[Flow Matching]]（底层生成框架）
- [[DPO]]（原始方法）
- [[RLHF]]（奖励学习更一般化的框架）
- [[VLA 后训练]]（应用场景）
