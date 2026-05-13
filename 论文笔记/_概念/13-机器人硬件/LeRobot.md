---
type: concept
aliases: [LeRobot, lerobot]
---

# LeRobot

## 定义
HuggingFace 推出的开源机器人学习框架/库，统一了机器人数据集格式、遥操与数据采集、低成本硬件驱动（如 SO-100/SO-101）、以及策略训练与部署接口，旨在降低真机机器人学习的门槛。

## 核心要点
1. 提供标准化的 `LeRobotDataset` 格式，方便共享和复用真机数据。
2. 内置对低成本开源机械臂（SO-100 等）的支持，配套遥操数据采集流程。
3. 与 [[SmolVLA]] 等开源 VLA 模型形成生态闭环；也是 [[RIO]] 等跨本体框架对比/借鉴的对象。

## 代表工作
- HuggingFace, 2024：LeRobot 开源项目。
- [[RIO]]：在跨本体 robot stack 对比中将 LeRobot 列为参照。

## 相关概念
- [[SO-100]]
- [[SmolVLA]]
- [[RIO]]
- [[DROID 数据集]]
