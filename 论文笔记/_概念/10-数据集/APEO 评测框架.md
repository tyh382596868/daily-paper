---
type: concept
aliases: [APEO, APEO Rubric, APEO 评测框架, Adherence Physics Environment Outcome]
---

# APEO 评测框架

## 定义

由 [[WhatIfWorld]] 提出的视频[[世界模型]]四维评测框架，包含 **A**dherence（动作执行）、**P**hysics（物理合理性）、**E**nvironment（环境稳定性）、**O**utcome（因果末态）四个维度，每个维度都在**单视频模式**和**配对模式**下分别评分。

## 数学形式

$$
s_{\text{Avg}} = \mathrm{mean}(A_s, P_s, E_s) \qquad p_{\text{Avg}} = \mathrm{mean}(A_p, P_p, E_p, O_p)
$$

其中 $s$ 下标表示单视频评分（无 Outcome 维度，因为需要配对才能评 Outcome），$p$ 下标表示配对评分。

## 核心要点

1. **四个维度**:
   - **Adherence**: 视频是否执行了 prompt 描述的动作？
   - **Physics**: 运动是否符合基本物理（无瞬移/形变/幽灵力）？
   - **Environment**: 背景、相机、非目标物体是否保持稳定？
   - **Outcome**: 配对视频末态差异是否符合物理预测？
2. **单视频 vs 配对双模式**: 单视频评内部一致性，配对评因果一致性。两者差距体现 [[对照瓶颈]]。
3. **二分类问题驱动**: 每个维度对应**一个二分类问题**（pass/fail），避免 Likert 评分的位置偏置、冗长偏置、中心倾向偏置。
4. **[[VLM-as-Judge|VLM judge]] 实现**: 在 421 条人工标注上达成 82.30% 一致率，与人-人一致率 84.03% 相当。

## 代表工作

- [[WhatIfWorld]]: 首次提出 APEO 框架，用于因果干预响应评测。

## 相关概念

- [[VLM-as-Judge]]
- [[对照瓶颈]]
- [[因果推断]]
- [[VBench]]: 单视频质量评测的代表，APEO 在其基础上引入配对评测。
