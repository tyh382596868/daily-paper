---
type: concept
aliases: [Observation-Action World Action Model]
---

# OA-WAM

## 定义
基于观测-动作对建模的 World Action Model，将观测序列和动作序列联合编码为统一的 latent 表示，用于机器人操作的策略学习和规划。

## 核心要点
1. 观测和动作在同一 latent 空间建模（unlike 分离的 world model + policy）
2. 可从无标签视频中学习动作 latent（latent action recovery）
3. 与 RepWAM、Genie 等 WAM 框架同类

## 代表工作
- [[RepWAM]] — 同类 WAM，用 RepViTok 改进 tokenizer

## 相关概念
- [[RLA-WM]]
- [[RepWAM]]
- [[WAM-Survey]]
