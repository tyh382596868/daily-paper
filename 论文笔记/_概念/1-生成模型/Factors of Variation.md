---
type: concept
aliases: [FoV, 变化因子, 因子变化]
---

# Factors of Variation

## 定义

Factors of Variation（FoV）指环境中可独立控制、可解耦的属性维度，常用于鲁棒性评测、表征学习、解耦表示（disentangled representation）。在 [[世界模型]] 评测中，FoV 用于生成可控的分布偏移以测试 OOD 泛化能力。

## 数学形式

给定环境 $\mathrm{env}(v)$，其中 $v \in \mathcal{V}$ 是 FoV 配置向量：

$$
\mathrm{Robustness}(\pi, \mathcal{V}) = \mathbb{E}_{v \sim p(\mathcal{V})} \big[ \mathrm{SuccessRate}(\pi, \mathrm{env}(v)) \big]
$$

在 train / test 时使用不同的 FoV 子集 $\mathcal{V}_{\mathrm{train}}, \mathcal{V}_{\mathrm{test}}$ 即可构造可控 OOD 评测。

## 核心要点

1. **维度可解耦**：颜色、尺寸、位置、物理参数等彼此独立
2. **声明式配置**：通过命名属性 + 配置字典控制，避免 hack 渲染器
3. **train/test 划分**：可在不同 FoV 子集上独立采样，天然支持 OOD
4. **覆盖三类属性**：视觉（颜色/纹理）、几何（尺寸/位置/形状）、物理（摩擦/质量/速度）

## 在 SWM 中的实现

[[StableWM]] 的每个环境暴露 6-17 个命名 FoV，例如 [[Push-T]]：
- `agent.color`、`agent.scale`、`agent.position`
- `block.color`、`block.scale`、`block.shape`
- `goal.position`、`goal.angle`
- 等共 16 个维度

用户通过 `options={"variation": ["agent", "block.color"]}` 在数据采集时选择扰动子集。

## 代表工作

- [[StableWM]]: 把 FoV 作为 WM 评测的核心抽象
- [[DINO-WM]]: 在 FoV 扰动下暴露脆弱性（94% → 4-20%）

## 相关概念

- [[世界模型]]
- [[Push-T]]
- [[DINO-WM]]
- [[sim-to-real]]
