---
type: concept
aliases: [Action-Effect Memory Pretraining]
---

# AEM (Action-Effect Memory Pretraining)

## 定义
港科大（广州）提出的操作策略预训练框架，从视觉-动作历史中学习紧凑的 action-effect 时序表示。

## 数学形式
$$z_\text{AEM} = f_\text{enc}(o_{t-k:t}, a_{t-k:t-1}), \quad a_t = f_\text{policy}(z_\text{AEM})$$

## 核心要点
1. 输入：视觉帧序列 + 动作历史
2. 用 CLS token 编码 action-effect 时序对
3. 在 RoboTwin2 和 ManiFlow 上预训练，与 Diffusion Policy 结合

## 代表工作
- [[AEM]]: arXiv 2606.12499

## 相关概念
- [[Diffusion Policy]]
- [[RoboTwin2]]
