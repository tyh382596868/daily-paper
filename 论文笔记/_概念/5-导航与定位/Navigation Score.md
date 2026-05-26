---
type: concept
aliases: [Navigation Score, 导航得分]
---

# Navigation Score

## 定义

[[WBench]] 中用于评测视频世界模型导航能力的复合指标：用 [[MegaSaM]] 从生成视频反推相机轨迹，与目标轨迹比较 [[Absolute Trajectory Error|ATE]]，并加跨轮一致性项归一化得到。

## 数学形式

$$
\mathrm{Nav} = 1 - \frac{1}{N}\sum_{i=1}^N \min\!\left(\frac{\mathrm{ATE}_i}{\tau},\, 1\right) + \lambda \cdot \mathrm{ConsistencyTerm}
$$

## 核心要点

1. **范式无关**: 文本驱动 / 相机控制 / 动作条件模型都能统一评测
2. **复合指标**: 单轮 ATE + 跨轮起止点对接一致性
3. **关键发现**（[[WBench]]）:
   - Navigation 与其他维度（画质 / 一致性 / 物理）**相关性近零** (r≈-0.12)
   - 多轮退化最严重：第 1 轮 → 第 4+ 轮 跌 **~33 分**
   - 相机控制 / 世界模型在 Native Navigation 上显著领先文本模型

## 代表工作

- [[WBench]]: 提出该指标作为 I.1（Interaction Adherence #1）
- HY-World 1.5 在该指标下 87.5 领跑相机控制范式

## 相关概念

- [[Absolute Trajectory Error]]
- [[MegaSaM]]
- [[5-导航与定位]]
