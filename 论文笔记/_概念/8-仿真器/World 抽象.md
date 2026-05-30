---
type: concept
aliases: [SWM World, swm.World, World Interface]
---

# World 抽象

## 定义

World 抽象是 [[StableWM]] 提出的环境-策略解耦接口，把 [[Gymnasium]] 风格的环境包装进 `swm.World` 对象，统一管理多并行环境实例、policy 注入、数据采集与评测，所有结果通过原地更新的 `world.infos` 字典暴露给外部。

## 核心 API

```python
import stable_worldmodel as swm
world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(YourPolicy())
world.reset()
world.step()  # infos 在 world.infos 字典中原地更新
```

Policy 实现只需提供 `get_action(info) -> np.ndarray`，形状 `(num_envs, action_dim)`。

## 核心要点

1. **环境-策略解耦**：用户的 policy 不需要知道并行细节
2. **同步并行**：`num_envs` 个环境同步推进，简化批量评测
3. **数据采集 = 评测**：`record_dataset` 和评测共享同一份 step 循环
4. **声明式 FoV**：通过 `options={"variation": [...]}` 控制 [[Factors of Variation]]
5. **统一格式**：观测/动作/状态/奖励都进 `world.infos`，下游消费者只关心字典 key

## 设计哲学

> "你的 codebase 不要动，我提供环境和评估"

World 抽象的目标是让 WM 研究者**不修改自己的训练代码**，只通过实现 `get_action` 协议接入 SWM 的所有评测基础设施。

## 代表工作

- [[StableWM]]: 提出并实现 World 抽象，包装 16 个环境

## 相关概念

- [[Gymnasium]]
- [[Factors of Variation]]
- [[Push-T]]
- [[TwoRoom]]
