---
type: concept
aliases: [动作捷径, 动作 Shortcut]
---

# Action Shortcut

## 定义
策略模型在训练中学到的**绕开真实因果链**的捷径——直接从语言/视觉的表面模式映射到动作，跳过中间应有的推理（如空间理解、深度感知、目标定位）。表现：在分布内任务上成功率高，但任何**分布偏移**（视角变化、新光照、新表面）都会导致急剧崩溃。

## 核心要点
1. **典型表征**: action head 仅依赖少量视觉 token + 任务关键词，不利用空间/深度通道
2. **触发因素**: prompt 模板触发模型"切换"到捷径路径（参见 [[Prompt-Induced Reasoning Gap]]）
3. **诊断方法**:
   - 投影空间相似度分析（[[3DThinkVLA]] Figure 5b）
   - 鲁棒性 benchmark（LIBERO-Plus、SimplerEnv）下的性能崩溃
4. **缓解方法**:
   - [[Latent Distillation|潜空间蒸馏]]强制 action 路径保留 reasoning 表征
   - 多模态/多任务 [[Co-training|协同训练]]增加 alternative 信号
   - Dropout 空间分支正则化（[[3DThinkVLA]] 在融合阶段用）

## 代表工作
- [[3DThinkVLA]]: 明确诊断并提出抑制方案
- [[Causal Confusion Paradox]]: 类似的因果混淆问题

## 相关概念
- [[Prompt-Induced Reasoning Gap]]
- [[Compounding Errors]]
- [[Causal Confusion Paradox]]
