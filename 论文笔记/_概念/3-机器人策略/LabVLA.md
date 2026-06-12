---
type: concept
aliases: [Laboratory VLA]
---

# LabVLA

## 定义
浙大/上海 AI Lab 提出的科学实验室 VLA，基于 RoboGenesis 仿真引擎合成 LabEmbodied-Data，采用 FAST 预训练 + Flow Matching 后训练的两阶段 recipe，在 LabUtopia benchmark 实现 71.1%/70.0% ID/OOD 成功率。

## 核心要点
1. **三件套**: RoboGenesis 引擎 + LabEmbodied-Data 数据集 + LabUtopia benchmark
2. **两阶段训练**: FAST Action Tokenizer 预训练对齐动作 token；Flow Matching 后训练 + Knowledge Insulation 防遗忘
3. **架构**: Qwen3-VL-4B-Instruct backbone + 18 层 DiT 动作专家，共 ~4B 参数
4. **跨具身支持**: 16 种机器人平台（单臂/双臂/移动操作臂）
5. **真实机器人验证**: Franka 平台 4 类任务，最高 92% 成功率

## 数学形式

Knowledge Insulation 核心：

$$
\mathcal{L}_{\text{KI}} = \alpha \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{FAST}} + \sum_j \lambda_j \mathcal{L}_{\text{CE}}^{(j)}, \quad \alpha = 10
$$

## 代表工作
- [[LabVLA]]: arXiv 2606.13578（Zhejiang University / Shanghai AI Lab / HIT）

## 相关概念
- [[FAST Action Tokenizer]]
- [[Flow Matching]]
- [[Knowledge Insulation]]
- [[DiT]]
- [[VLA]]
- [[RoboGenesis]]
- [[Cross-Embodiment Learning]]
