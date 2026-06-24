---
type: concept
aliases: [CogVLA]
---

# CogVLA

## 定义
基于 CogVLM 视觉语言模型骨干的机器人操作 VLA，采用 Fast-Slow 双系统架构：低频 CogVLM 处理语言-视觉理解，高频动作专家处理精细控制。

## 核心要点
1. CogVLM 作为高容量视觉语言理解模块
2. Fast-Slow 解耦：推理效率与语义理解兼顾
3. 在 LIBERO benchmark 系列上评测

## 代表工作
- [[CogVLA]]: 原始论文（ZhipuAI 相关工作）
- [[UniFS]]: 将 CogVLA 作为 Fast-Slow VLA 架构的对比方法

## 相关概念
- [[VLA]] — 上层概念
- [[双系统架构]] — 架构范式
- [[OpenVLA]] — 同类方法对比
