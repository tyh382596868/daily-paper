---
type: concept
aliases: [CALVIN Benchmark, Composing Actions from Language and Vision]
---

# CALVIN

## 定义

CALVIN（Composing Actions from Language and Vision）是 Mees 等人提出的语言条件长程操作 benchmark，共 4 个环境（A/B/C/D），任务由自然语言指令串接，评估"完成连续 5 个子任务"的链长。是 [[VLA]] 类方法最常用的仿真评测之一。

## 核心要点

1. **4 环境 × 34 子任务**：A/B/C/D 四种桌面物理布局，包含开关抽屉、推方块、按按钮、拿杯子等；
2. **两个主流 split**：
   - **ABCD→D**：4 环境数据训练 → D 上评估（in-distribution）；
   - **ABC→D**：ABC 训练 → D 评估（out-of-distribution，难度更高）；
3. **核心指标**：连续 5 个子任务的平均成功链长（最大 5.0）；
4. **数据规模**：约 24k+ 演示，附带语言标注。

## 代表工作

- **CALVIN 原论文** (Mees et al., 2022)
- [[NIAF]]：ABC→D = 4.47 / ABCD→D = 4.66（SOTA）
- [[FLOWER]] / [[BEAST]] / [[UniVLA]]：主要对比基线

## 相关概念

- [[LIBERO]]
- [[VLA]]
- [[Action Chunking]]
