---
type: concept
aliases: [Zero-shot Robustness, 零样本鲁棒性, OOD Robustness]
---

# Zero-shot Robustness

## 定义

Zero-shot Robustness 指模型在**训练时未见过的分布偏移**（视觉/几何/物理扰动、新场景、新指令）下，**无需任何微调**直接维持任务性能的能力。在[[世界模型]]评测中，常通过 [[Factors of Variation]] 扰动来量化：固定模型权重，仅改变环境的某个 FoV，测量成功率下降幅度。

## 数学形式

给定训练分布 $\mathcal{D}_{\mathrm{train}}$ 上学到的策略 $\pi$ 与扰动分布 $\mathcal{D}_v$：

$$
\mathrm{ZSR}(\pi, v) = \mathbb{E}_{\tau \sim \pi, \, \mathrm{env}(v)} \big[ \mathbb{1}\{\mathrm{success}(\tau)\} \big], \quad v \notin \mathrm{supp}(\mathcal{D}_{\mathrm{train}})
$$

鲁棒性间隙 (robustness gap)：

$$
\Delta = \mathrm{SuccessRate}_{\mathrm{ID}} - \mathrm{ZSR}(\pi, v)
$$

## 核心要点

1. **零微调**：与 transfer learning / few-shot 不同，禁止任何额外训练
2. **可控扰动**：用 FoV 等结构化扰动，避免"难度无法量化"的问题
3. **在 WM 中的关键性**：deployment 时遇到的环境总会偏离训练分布，鲁棒性是 deploy-ready 的硬指标
4. **典型失败模式**：模型学到了 task-irrelevant 特征（如背景颜色），扰动后 latent 漂移导致规划失败

## 代表工作

- [[StableWM]]: 用 FoV 系统性量化 ZSR，揭示 [[DINO-WM]] 在 Push-T 上 94% → 4-20%
- [[DINO-WM]]: 在 ID 上表现强，但 ZSR 严重不足
- VBench / WBench: 视频生成的鲁棒性评测但与 control 任务无关

## 相关概念

- [[Factors of Variation]]
- [[世界模型]]
- [[Continual Learning]]
- [[sim-to-real]]
