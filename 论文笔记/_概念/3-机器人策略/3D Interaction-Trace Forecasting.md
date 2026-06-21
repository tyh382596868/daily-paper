---
type: concept
aliases: [3D 交互轨迹预测]
---

# 3D Interaction-Trace Forecasting

## 定义
预测关键交互点（手、工具、物体、接触区域）在 3D 空间中的平滑运动轨迹，作为操作规划的中间表示。

## 数学形式
$$\hat{\tau}_{1:T} = f_{\theta}(o_t, \text{query points})$$
其中 $\hat{\tau}$ 为预测的 3D 轨迹序列。

## 核心要点
1. 只预测稀疏的关键点轨迹而非完整视频帧
2. 具身无关，同一轨迹可迁移到不同机器人
3. 比像素预测更高效，比关节级动作更可解释

## 代表工作
- [[mu0]]: 提出 3D interaction trace 作为 compact world model 表示
- [[SeeTraceAct]]: 2D 轨迹追踪先驱工作

## 相关概念
