---
type: concept
aliases: [RLBench, RL Benchmark, 机器人学习基准]
---

# RLBench

## 定义

RLBench 是基于 CoppeliaSim 仿真器构建的大规模机器人操作 benchmark，提供 100+ 个多样化的操作任务，支持视觉观测和状态观测两种模式，是评测机器人操作策略泛化能力的标准平台。

## 核心要点

1. **任务多样性**: 涵盖抓取、放置、旋转、按压、装配等多种操作类型
2. **多模态观测**: 支持 RGB、深度、点云等多种感知输入
3. **标准化评测**: 成功率（SR）为主要评测指标，通常每个任务运行多次取平均

## 代表工作

- James et al. "RLBench: The Robot Learning Benchmark & Learning Environment" (RA-L 2020)
- [[GeneralVLA2]]: 在 14 个 RLBench 任务上评测，覆盖 Play_jenga、Insert_block 等高难度任务

## 相关概念

- [[VLA]]
- [[3DAgent]]
