---
type: concept
aliases: [RMA, Rapid Motor Adaptation, 快速运动适应]
---

# RMA

## 定义
腿式机器人 sim-to-real 迁移的经典 Teacher-Student 蒸馏框架：Teacher 策略在仿真中用特权信息（地形、摩擦系数等）训练，Student 策略在真实部署时仅用机载传感器信息，通过在线适应模块实时推断环境参数。

## 数学形式
$$a_t = \pi(o_t, \hat{e}_t), \quad \hat{e}_t = \text{AdaptModule}(o_{t-k:t})$$

其中 $\hat{e}_t$ 为 adaptation module 从历史观测推断的隐式环境嵌入，替代 Teacher 策略中的显式特权信息 $e_t$。

## 核心要点
1. 两阶段训练：Phase 1 用 PPO 训练 Teacher（含特权信息）；Phase 2 蒸馏，训练 Adaptation Module 预测环境嵌入
2. 在线适应：部署时 Adaptation Module 实时更新隐式环境参数，无需重新训练
3. 无需域随机化调参：环境参数由模块自动推断

## 代表工作
- [[RMA]]: Kumar et al., 2021 (Berkeley)，用于四足机器人户外地形行进
- [[DynaWM]]: 在双足轮腿楼梯运动中基于 RMA 框架，加入世界模型增强动力学感知

## 相关概念
- [[Sim-to-Real]] — 框架目标
- [[Knowledge Distillation]] — 蒸馏方法
- [[PPO]] — Teacher 策略训练算法
- [[locomotion]] — 应用场景
