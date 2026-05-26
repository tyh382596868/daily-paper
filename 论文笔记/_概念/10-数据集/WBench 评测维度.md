---
type: concept
aliases: [WBench 评测维度, Video Quality, Setting Adherence, Interaction Adherence, Consistency, Physical Compliance, WBench Dimensions]
---

# WBench 评测维度

## 定义

[[WBench]] 提出的 5 维 22 子指标评测体系，用于全面诊断 [[交互式世界模型]] 的各项能力。

## 五大维度

| 维度 | 子指标数 | 关键能力 |
|------|----------|----------|
| **Video Quality** | 6 | 美学、画质、闪烁、动态度、流畅度、HPSv3-Norm |
| **Setting Adherence** | 2 | 场景设定遵循、主体设定遵循（VLM 评分） |
| **Interaction Adherence** | 4 | [[Navigation Score|导航]]、事件编辑、主体动作、视角切换 |
| **Consistency** | 8 | [[Spatial Consistency|空间]] / [[Geometric Consistency|几何]] / [[Photometric Consistency|光度]] / 主体 / 背景 / 视角 / 片段连续性 |
| **Physical Compliance** | 2 | 因果保真度（VLM 两阶段）、视觉合理性（微调 VLM） |

## 核心要点

1. **维度互补**: 5 维之间相关性低，单一指标无法替代综合评测
2. **关键相关性**:
   - Physics ↔ Video Quality 强相关 (r=0.84)
   - Navigation ↔ 其他维度 **几乎正交** (r≈-0.12)
   - 相机控制能力 ↔ 视角一致性 **近零相关**
3. **多轮退化**: Navigation 第 1 → 第 4+ 轮跌 33 分，是当前世界模型最大短板
4. **人类对齐**: 所有 10 个评测方面 [[人类对齐 Spearman 相关|Spearman ρ]]≥0.94

## 代表工作

- [[WBench]]: 提出该评测体系，评测 20 个 SOTA 模型

## 相关概念

- [[VLM-as-Judge]]
- [[Navigation Score]]
- [[Spatial Consistency]]
- [[Geometric Consistency]]
- [[Photometric Consistency]]
- [[10-数据集]]
