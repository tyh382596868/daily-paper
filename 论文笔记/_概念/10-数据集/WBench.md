---
type: concept
aliases: [WBench, WBench Benchmark, Meituan WBench]
---

# WBench

## 定义

首个面向 [[交互式世界模型]] 的多轮评测基准。沿 **5 维 22 子指标**（Video Quality / Setting Adherence / Interaction Adherence / Consistency / Physical Compliance）评估模型，提供统一的导航接口（文本 ↔ 6-DoF pose ↔ 离散动作），使 [[文本驱动视频生成|文本驱动]]、[[Camera-Controlled 视频生成|相机控制]]、[[Action-Conditioned World Model|动作条件]] 三类模型可公平横向比较。

## 核心要点

1. **289 cases / 1058 turns**：多轮交互（平均 3.7 轮/case，最长 9 轮），覆盖第一/第三视角、6 类场景、4 类交互（Navigation / Subject Action / Event Editing / Perspective Switching）。
2. **22 子指标**：覆盖像素级、特征级、几何级、语义级；其中 [[VLM-as-Judge]] 用 [[Qwen3-VL]]-30B 评估语义层（事件触发、动作完成、视角切换、物理合理性）。
3. **统一动作接口**：文本指令、6-DoF 位姿、离散动作三种动作表征可双向转换，用 [[MegaSaM]] 反推实际轨迹做跨范式对比。
4. **人类对齐**：10 个评测方面 Spearman ρ≥0.94，其中 4 项达 ρ=1.00（事件编辑、主体动作、视角切换、空间一致性）。
5. **关键诊断结论**：没有模型 5 维全胜；**Navigation 多轮退化 33 分**远高于语义交互；Physics ↔ Video Quality 强相关（r=0.84），Navigation 与其他维度近零相关。

## 代表工作

- [[WBench]]（本基准论文）：评测 20 个 SOTA 系统（Seedance / Wan / Kling / LongCat-Video / Cosmos / Genie 3 / Matrix-Game / HY-World 等）
- 后续视频世界模型评测的事实标准（与 [[VBench]] / [[WorldArena]] 互补）
- [[WhatIfWorld]]: 与 WBench 互补——WBench 评多轮交互一致性，WhatIfWorld 评反事实因果干预响应。

## 相关概念

- [[VBench]]: 单轮视频质量评测前身
- [[WorldArena]]: 机器人仿真域交互评测前身
- [[交互式世界模型]]
- [[VLM-as-Judge]]
- [[10-数据集]]
