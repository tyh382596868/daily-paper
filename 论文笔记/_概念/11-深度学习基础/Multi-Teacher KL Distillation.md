---
type: concept
aliases: [多教师 KL 蒸馏]
---

# Multi-Teacher KL Distillation

## 定义
用多个专家教师网络的输出分布对单个学生网络进行 KL 散度蒸馏，整合多种专家能力到一个模型。

## 数学形式
$$\mathcal{L}_{distill} = \sum_i \lambda_i \cdot KL(\pi_{teacher_i}(\cdot|s) \| \pi_{student}(\cdot|s))$$

## 核心要点
1. 每个教师专注不同技能（运动跟踪、步态、跌倒恢复）
2. KL 蒸馏允许学生同时学习所有教师分布
3. 常与 MoE 结构结合，不同场景激活不同专家

## 代表工作
- [[HANDOFF]]: 用多教师 KL 蒸馏训练人形机器人全身控制器

## 相关概念
