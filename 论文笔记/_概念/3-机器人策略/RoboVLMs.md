---
type: concept
aliases: [RoboVLMs, Robot VLMs, RoboVLM Evaluation]
---

# RoboVLMs

## 定义

一个统一的机器人 VLA 评估框架（GitHub: Robot-VLAs/RoboVLMs），提供 CALVIN、LIBERO、SimplerEnv 等多个 benchmark 的标准化评估脚本，被多个 VLA 工作用作评估基础设施。

## 核心要点

1. **统一评估接口**: 对 CALVIN、LIBERO、SimplerEnv 提供标准评估脚本，便于公平对比
2. **多机器人支持**: 支持 Franka、WidowX 等多种机器人平台
3. **VLA 兼容性**: 支持 OpenVLA、UniVLA 等主流 VLA 模型的评估接入
4. **社区基础设施**: 被 UniVLA、FASTER 等论文作为评估代码基础使用

## 代表工作

- [[UniVLA]]: 使用 RoboVLMs 作为 CALVIN/LIBERO/SimplerEnv 评估框架
- [[OpenVLA]]: RoboVLMs 的兼容对象之一

## 相关概念

- [[CALVIN]]: RoboVLMs 支持的 benchmark 之一
- [[LIBERO]]: RoboVLMs 支持的 benchmark 之一
- [[SimplerEnv]]: RoboVLMs 支持的 benchmark 之一
- [[VLA]]: RoboVLMs 的评估对象
