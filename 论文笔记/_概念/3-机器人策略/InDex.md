---
type: concept
aliases: [Intent-Conditioned Dexterous Fine-Tuning]
---

# InDex (Intent-Conditioned Fine-Tuning for Dexterous Manipulation)

## 定义
北航提出的灵巧手 VLA 适配方法：通过 Macro-Reaching 宏动作分解 + Intent-Conditioned LoRA 微调，将低 DoF VLA 适配到高 DoF 灵巧手。

## 核心要点
1. 问题：VLA 预训练在低 DoF 夹爪数据，无法直接控制灵巧手
2. Macro-Reaching：将灵巧操作分解为粗略位置控制 + 精细手部控制
3. Intent-Conditioned LoRA：意图条件化微调
4. 仅仿真验证（has_real_world=False）

## 代表工作
- [[InDex]]: arXiv 2606.12109

## 相关概念
- [[OpenVLA]]
- [[UniVLA]]
- [[LoRA]]
