---
type: concept
aliases: [BridgeV2, Bridge Dataset]
---

# BridgeData

## 定义
斯坦福/伯克利开源的大规模机器人操作数据集（BridgeData V2），包含 60,096 条真实机器人轨迹，覆盖厨房、桌面等场景，广泛用于 VLA 训练和评测。

## 核心要点
1. 约 60k 条 7-DoF WidowX 机械臂轨迹，24 个任务类别
2. 语言条件化：每条轨迹配自然语言指令
3. 是 Open X-Embodiment、RT-2、OpenVLA 的主要训练数据之一
4. 常作为 VLA 泛化性评测的 out-of-distribution 测试集

## 代表工作
- [[RT-2]]: 使用 BridgeData 训练
- [[OpenVLA]]: BridgeData 是重要数据组成部分
- [[ReflectiveVLA]]: 用 BridgeData 做 OOD 泛化评测

## 相关概念
- [[RoboMIND]]
- [[LIBERO]]
- [[Open X-Embodiment]]
