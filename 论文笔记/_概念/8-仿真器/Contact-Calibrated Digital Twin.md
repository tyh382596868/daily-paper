---
type: concept
aliases: [接触校准数字孪生, 触觉数字孪生]
---

# Contact-Calibrated Digital Twin

## 定义
通过真实触觉传感器信号标定仿真器的接触参数，构建能重现真实触觉响应的数字孪生仿真环境。

## 数学形式
$$\min_{\theta_{sim}} \|T_{sim}(\theta_{sim}) - T_{real}\|_2^2$$
其中 $T$ 为触觉信号，$\theta_{sim}$ 为仿真接触参数。

## 核心要点
1. Real2Sim 阶段：用真实接触数据校准仿真器参数
2. 仿真中训练触觉策略
3. 零样本迁移到真实机器人

## 代表工作
- [[Blind Dexterous Grasping]]: 提出 Real2Sim 触觉校准流程

## 相关概念
