---
type: concept
aliases: [MemoryVLA, MemoryVLA++]
---

# MemoryVLA / MemoryVLA++

## 定义

清华大学 & Dexmal 提出的 **Cognition-Memory-Action**（记忆增强 [[VLA]]）框架：用 **感知-认知双通道记忆库（PCMB）** 显式建模机器人操作任务的非马尔科夫时序依赖，在 ICLR 2026 以 MemoryVLA 发表，期刊扩展版 MemoryVLA++ 进一步引入 **Temporal Imagination**（时序想象）模块实现双向时序建模。

## 核心要点

1. **双通道 PCMB**：感知 Token（DINOv2+SigLIP）+ 认知 Token（LLaMA 隐层）分别存储历史，低层细节与高层语义互补
2. **门控融合**：$\tilde{x} = g_x \odot H_x + (1-g_x) \odot x$，自适应融合检索历史与当前工作记忆
3. **Token-merge Consolidation**：超容量时对余弦相似度最高的相邻条目取均值合并，防止记忆无限增长
4. **Temporal Imagination（MemoryVLA++ 新增）**：前向预测未来观测表征，与历史记忆共同为动作生成提供双向时序上下文
5. **Diffusion Action Expert**：记忆/想象增强 Token 条件化 DiT 预测动作块

## 代表工作

- **MemoryVLA** (arXiv:2508.19236, ICLR 2026)：提出 PCMB，SimplerEnv-Bridge 71.9%，LIBERO-5 96.5%，真实长程任务 +26pp vs CogACT
- **MemoryVLA++** (arXiv:2606.09827)：扩展 Imagination 模块，提出 LIBERO-Memory 基准
- [[RoboMemArena]] 中早期版本作为对比基线，平均 TSR 15.0%（此处指较简单的 token-level 工作记忆实现）
- [[EvoScene-VLA]] 相关工作对比

## 相关概念

- [[VLA]]
- [[感知-认知记忆库|PCMB]]
- [[关键帧记忆库]]
- [[Action Chunking]]
- [[SOMA]]: 显式空间 3D 记忆路线，与 MemoryVLA 的 token 时间记忆形成正交补充
- [[MemER]]
