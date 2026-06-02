---
type: concept
aliases: [ALOHA, ALOHA 平台, A Low-cost Open-source Hardware System]
---

# ALOHA

## 定义

斯坦福开源的低成本双臂遥操作平台（A Low-cost Open-source Hardware system for bimAnual teleoperation），由两臂主-从（leader-follower）结构组成，主要用于采集双手精细操作（折衣、穿针、剥蛋）的模仿学习数据。

## 核心要点

1. 主从同构双臂遥操作 — 操作员物理控制 leader 关节，follower 跟随
2. 采样频率 ~50 Hz，适合 [[Diffusion Policy]] / [[ACT (Action Chunking Transformer)|ACT]] 类策略
3. 是 [[Action Chunking]] 思想的代表平台
4. 与 [[DROID]]（Franka 单臂）形成数据来源互补

## 代表工作

- Zhao et al. 2023: ACT + ALOHA
- Mobile ALOHA, ALOHA 2

## 相关概念

- [[Action Chunking]]
- [[Diffusion Policy]]
- [[DROID]]
