---
type: concept
aliases: [SmolVLA, Smol VLA]
---

# SmolVLA

## 定义
一个紧凑、开源的 Vision-Language-Action 模型（由 HuggingFace 提出），把小型 VLM 骨干 + 轻量 action expert 结合，目标是在消费级算力上做可用的机器人操作策略，常被后续工作当作轻量 VLA baseline。

## 核心要点
1. 走"小而能用"路线：参数量远小于 [[OpenVLA]]、[[π₀]] 这类模型，便于在低成本机器人（如 SO-100）上部署。
2. 配合 [[LeRobot]] 生态使用，强调开源数据 + 开源训练流程。
3. 在多篇 VLA inference-time 增强工作（token 剪枝、动态修正、自适应 chunk）里作为被"打补丁"的冻结骨干出现。

## 代表工作
- HuggingFace, 2025：SmolVLA 原始工作。
- [[GridS]]、[[PPC]]、[[MCF-Proto]]：以 SmolVLA 等为骨干做效率/鲁棒性增强。

## 相关概念
- [[OpenVLA]]
- [[LeRobot]]
- [[VLA]]
- [[Action Chunking]]
