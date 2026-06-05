---
type: concept
aliases: [VLM-as-Judge, VLM 作为评测器, MLLM-as-Judge]
---

# VLM-as-Judge

## 定义

用一个 [[VLM|视觉语言模型]] 作为评测器对视频/图像生成结果打分的评测范式。常用于无法用传统视觉指标（PSNR / FID）评估的**语义层**任务，例如事件触发、动作完成度、视角切换正确性、物理合理性。

## 核心要点

1. **典型协议**: 5 点李克特量表 + 多子项打分（变化检测、事件发生、完成度、细节准确性、伪影存在性）
2. **优势**: 可评估单纯靠像素无法判断的语义维度；与人类判断高度对齐（[[WBench]] 报告 Spearman ρ≥0.94）
3. **劣势**:
   - 依赖外部模型，模型更新会导致分数漂移
   - VLM 自身的偏置（数据偏置 / 风格偏置 / 长度偏置）
   - 评估成本高，需要多次推理
4. **常用模型**: GPT-4V, Gemini, [[Qwen3-VL]]-30B, InternVL
5. **进阶用法**: **微调判官**——在伪影数据上微调 VLM 提升对几何畸变、穿模、不自然形变的敏感度

## 代表工作

- [[WBench]]: 用 Qwen3-VL-30B 实现 Interaction Adherence + Physical Compliance 的全自动评测
- [[Dream-exe]]: 用 Gemini 3 Pro + Qwen3-VL 双判定评估视频生成的可执行性
- VBench-2.0 / GenAI-Bench: 视频/图像生成评测中的 VLM 评判
- MM-Vet, LLaVA-Bench: 早期 VLM-as-Judge 实践

## 相关概念

- [[VLM]]
- [[Qwen3-VL]]
- [[11-深度学习基础]]
