---
type: concept
aliases: [物理合理性, Plausibility, Physical Realism]
---

# Physical Plausibility

## 定义

视频或仿真画面在重力、接触、刚体动力学等基本物理规律下"看起来合理"的程度，常作为视频生成模型的评测维度。

## 核心要点

1. 通常由 [[VLM-as-Judge|VLM judges]] 或人评打分
2. 与"任务可执行性"相关性可能极低——[[Dream-exe]] 报告 Spearman $r=-0.03$
3. 与 [[VBench]]、[[Physics IQ]] 等基准的核心指标重叠

## 代表工作
- [[Dream-exe]]：证伪 plausibility = world model
- [[Physics IQ]]
- [[VBench]]

## 相关概念
- [[VBench]]
- [[Physics IQ]]
- [[World Model]]
- [[Dream.exe Benchmark]]
