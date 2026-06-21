---
type: concept
aliases: [RL 流程自动化, LLM-driven RL]
---

# Agentic RL Automation

## 定义
用 LLM agent 自动完成强化学习工作流中的奖励设计、算法选择和超参调节等环节，减少人工干预。

## 数学形式
无核心公式，属于系统设计范式。

## 核心要点
1. LLM 读取任务描述，生成奖励函数代码
2. 自动调用仿真器运行训练并评估结果
3. 迭代优化奖励和超参直到策略收敛

## 代表工作
- [[HARBOR]]: 提出完整的 Agentic RL 自动化框架
- [[Eureka]]: LLM 奖励设计先驱工作

## 相关概念
