---
type: concept
aliases: [gym, OpenAI Gym, Farama Gymnasium]
---

# Gymnasium

## 定义

Gymnasium 是 Farama 基金会维护的 RL 环境接口标准（OpenAI Gym 的后继者），定义了 `reset()` / `step()` / `observation_space` / `action_space` 等核心 API。几乎所有 RL 和世界模型代码库都基于 Gymnasium 接口构建。

## 核心 API

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

## 核心要点

1. **接口最小集**：只规定环境契约，不规定算法
2. **VectorEnv**：原生支持多环境并行执行
3. **Wrapper 模式**：通过包装器添加观测变换、奖励重塑等
4. **广泛兼容**：[[MuJoCo]]、[[DMControl]]、Atari、[[OGBench]] 都有 Gymnasium 接口

## 在 SWM 中的位置

[[StableWM]] 在 Gymnasium 之上做了一层 `swm.World` 抽象：
- 把多环境并行、policy 注入、数据采集、评测统一进同一个执行循环
- 用 `world.infos` 字典原地写入 observation/state/action/reward，替代 Gym 风格的元组返回
- 保持向下兼容：底层仍是 Gymnasium 环境

## 代表工作

- [[StableWM]]: 在 Gymnasium 上构建 WM 研究层
- [[OGBench]]: 提供 Gymnasium 接口的操作 benchmark

## 相关概念

- [[MuJoCo]]
- [[DMControl]]
- [[World 抽象]]
