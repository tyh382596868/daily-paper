---
title: "YoCausal: How Far is Video Generation from World Model? A Causality Perspective"
method_name: "YoCausal"
authors: [You-Zhe Xie, Yu-Hsuan Li, Jie-Ying Lee, Kaipeng Zhang, Yu-Lun Liu, Zhixiang Wang]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, benchmark, causality, evaluation, counterfactual, arrow-of-time]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2605.30346v1
created: 2026-05-30
---

# 论文笔记：YoCausal: How Far is Video Generation from World Model? A Causality Perspective

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National Yang Ming Chiao Tung University (NYCU) |
| 日期 | May 2026 |
| 项目主页 | https://www.youzhexie.me/papers/YoCausal/index.html |
| 代码仓库 | https://github.com/youzhe0305/YoCausal |
| 数据集 | https://huggingface.co/datasets/YouZhe/YoCausal-dataset |
| 对比基线 | [[VBench]]、[[WBench]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30346) / [Code](https://github.com/youzhe0305/YoCausal) |

---

## 一句话总结

> 用"反向播放真实视频作为反事实样本"来定量探测 [[Video Diffusion Model|视频扩散模型]] 是否具备 [[World Model|世界模型]] 该有的 [[Arrow of Time|时间方向]] 感知与 [[Counterfactual|反事实]] 因果推理能力。

---

## 核心贡献

1. **YoCausal Benchmark**: 首个把"video gen 是不是 world model"这个含糊问题，落到**可测量、双层级、基于反事实**的评测框架。不用合成数据，直接对真实视频做时间反转即得到反事实样本。
2. **[[Reverse Surprise Index|RSI 指标]]**: Level 1 指标，用 [[Denoising Loss|去噪损失]] 在正向/反向序列上的差异量化"模型是否感知到时间方向"。
3. **[[Causality Cognition Index|CCI 指标]]**: Level 2 指标，借助 [[Vision-Language Model|VLM]] 把数据集划分为因果子集 $\mathcal{D}_c$ 和非因果子集 $\mathcal{D}_{nc}$，用 $\mathrm{RSI}(\mathcal{D}_c) - \mathrm{RSI}(\mathcal{D}_{nc})$ 把"真正的因果理解"从"单纯的时间偏置"里剥离出来。
4. **13 个主流 VDM 的全面评测**: 覆盖 [[AnimateDiff]]、[[CogVideoX]]、[[Mochi-1]]、[[HunyuanVideo]]、[[Wan2.1]] / [[Wan2.2]] 系列、[[LTX-Video]]，给出与 human baseline 的差距分析、与模型规模/发布时间的相关性分析。

---

## 问题背景

### 要解决的问题

[[Video Diffusion Model|视频扩散模型]] 近年被反复"暗示"具备 [[World Model|世界模型]] 能力，但社区缺少一个**可量化的、基于因果性的**测试方法。现有评测（如 [[VBench]]）主要看视觉质量、美学、时间一致性，无法回答：模型究竟是学到了"事件如何按因果展开"，还是只记住了"训练分布中常见的时间统计规律"？

### 现有方法的局限

- **VBench 类指标**：聚焦感知质量与美学，与因果理解几乎正交（论文实测 [[Aesthetic Quality|美学]] 与 CCI 的 Kendall $\tau = 0.0000$）。
- **合成场景探针**：之前一些 WM 评测用合成 / 受控物理场景，与开放世界分布差距大，结论难以泛化。
- **隐式论证**："video gen ≈ world model" 的论证大多是定性的、靠 cherry-picked sample，缺乏数值证据。

### 本文的动机

借鉴认知科学里的 [[Violation of Expectation|Violation of Expectation 范式]]——把世界"反着放"立刻就违反了因果直觉。这给了一种**几乎零成本的反事实生成方法**：用 ffmpeg 把视频时间轴反转即得到分布外但视觉合理的样本。如果模型真的懂因果，正向去噪损失应显著低于反向；如果只学到统计，两者差异会消失。

---

## 方法详解

### 框架总览

YoCausal 采用 **双层级 probing benchmark** 架构：

- **输入**: 真实视频 $x^f$（forward）与其时间反转版 $x^r$（reverse）
- **被测模型**: 已训练好的 [[Video Diffusion Model|VDM]] $\theta$（黑盒，不更新参数）
- **核心机制**:
  - Level 1 用 [[Denoising Loss|去噪损失]] 比较 $x^f$ vs $x^r$ → [[Reverse Surprise Index|RSI]]
  - Level 2 用 [[Vision-Language Model|VLM]] 做因果分层 → [[Causality Cognition Index|CCI]]
- **输出**: 每个被测 VDM 在四个数据子集（General/Physics/Human/Animal）上的 RSI 和 CCI 分数

### 核心模块

#### 模块 1: 时间反转作为反事实生成

**设计动机**: 与合成 [[Counterfactual|反事实]] 数据不同，时间反转保持了像素分布、纹理、物体外观，只破坏了因果时序。这避免了"反事实样本本身分布外"导致的混淆。

**具体实现**:
- 4 个真实视频源 → 总计 1,232 条视频（见数据集表）
- 每条视频生成正反两版 $(x^f, x^r)$
- 不需要任何标注，扩展性强

#### 模块 2: Level 1 - RSI（时间方向感知）

**设计动机**: 一个真正学到了"世界如何演化"的模型，应该对反向视频感到"惊讶"——即其 [[Denoising Loss|去噪损失]] 应该更高。

**具体实现**:
- 对每个视频对 $(x^f_{ij}, x^r_{ij})$ 计算两个去噪损失
- 用 indicator function 统计"反向损失 > 正向损失"的比例
- 在 4 个子集上分别计算后取平均

#### 模块 3: Level 2 - CCI（真正的因果认知）

**设计动机**: 高 RSI 不等于懂因果——可能只是"反向播放的视频在像素层面更不自然，所以 loss 更高"。要剥离这种 spurious 信号，需要对比"强因果"与"弱因果"视频上的 RSI 差异。

**具体实现**:
- 用 **Gemini 3.0 Pro** 作为 [[VLM-as-Judge|VLM 判官]]，把每个子集划分为因果子集 $\mathcal{D}_c$（如"杯子掉下来摔碎"）与非因果子集 $\mathcal{D}_{nc}$（如"鸟在天上飞"，正反看都合理）
- 与人工标注的 [[Kendall Tau|Kendall $\tau$]] 一致性 = 0.7613（高一致）
- CCI = 因果子集 RSI − 非因果子集 RSI

#### 模块 4: 归一化与可比性

引入 Normalized CCI，把 human baseline 设为 100%，方便跨模型横向对比。

---

## 关键公式

### 公式 1: [[Reverse Surprise Index|RSI - 时间方向感知指标]]

$$
\mathrm{RSI}(\mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{\mathcal{D}_i \in \mathcal{D}} \frac{1}{|\mathcal{D}_i|} \sum_{x_{ij} \in \mathcal{D}_i} \mathbb{1}\!\left[ \mathcal{L}_{\mathrm{denoise}}(\theta; x^r_{ij}) > \mathcal{L}_{\mathrm{denoise}}(\theta; x^f_{ij}) \right]
$$

**含义**: 在每个子集 $\mathcal{D}_i$ 上统计"模型对反向视频比正向视频更困惑"的比例，跨子集取均值。取值范围 $[0, 1]$，越高代表 [[Arrow of Time|时间方向]] 感知越强。0.5 = 完全随机。

**符号说明**:
- $\mathcal{D}$: 整体数据集，由 4 个子集 $\{\mathcal{D}_{\text{General}}, \mathcal{D}_{\text{Physics}}, \mathcal{D}_{\text{Human}}, \mathcal{D}_{\text{Animal}}\}$ 组成
- $x^f_{ij}, x^r_{ij}$: 第 $j$ 个视频的正向版与时间反转版
- $\mathcal{L}_{\mathrm{denoise}}(\theta; x)$: VDM 参数 $\theta$ 在视频 $x$ 上的 [[Denoising Loss|扩散去噪损失]]（与训练目标相同）
- $\mathbb{1}[\cdot]$: 指示函数
- Human baseline = **79.08%**

### 公式 2: [[Causality Cognition Index|CCI - 因果认知指标]]

$$
\mathrm{CCI}(\mathcal{D}) = \mathrm{RSI}(\mathcal{D}_c) - \mathrm{RSI}(\mathcal{D}_{nc})
$$

**含义**: 同一个模型在**因果子集 $\mathcal{D}_c$** 与**非因果子集 $\mathcal{D}_{nc}$** 上的 RSI 差值。如果模型真的学到了因果（而不是浅层时序统计），那么因果视频上的"反向异常感"应该显著强于非因果视频。

**符号说明**:
- $\mathcal{D}_c$: 由 [[Vision-Language Model|VLM]]（Gemini 3.0 Pro）判定为"明确具备因果链"的视频（如打碎、跌倒、连锁反应）
- $\mathcal{D}_{nc}$: 非因果视频，正反播放都看似合理（如稳态飞行、鱼群游动）
- 取值范围 $[-1, 1]$，正值代表 **有因果感**，负值代表"模型反而在非因果视频上更困惑"——说明它依赖低层时序统计而非因果
- Human baseline = **+8.67%**

### 公式 3: Normalized CCI（归一化因果认知）

$$
\widetilde{\mathrm{CCI}}(\mathcal{D}) = \frac{\mathrm{CCI}(\mathcal{D})}{\mathrm{CCI}_{\mathrm{human}}(\mathcal{D})} \times 100\%
$$

**含义**: 把 human baseline 设为 100%，方便横向对比。如 Wan2.1-14B 的 Normalized CCI ≈ 68.17%，意为达到人类因果认知水平的 ~68%。

**符号说明**:
- $\mathrm{CCI}_{\mathrm{human}}(\mathcal{D})$: 人类标注者在同一数据集上的 CCI
- 负值意味着模型方向与人类相反（4 个被测模型出现了这种情况）

### 公式 4: Kendall 一致性（评测稳健性验证）

$$
\tau = \frac{\#\text{Concordant} - \#\text{Discordant}}{\binom{n}{2}}
$$

**含义**: 用 [[Kendall Tau|Kendall $\tau$]] 验证 YoCausal 排名与其他指标的一致性。结论：
- 与人类偏好 $\tau = 0.5111$（中等正相关）
- 与 [[LikePhys 物理理解]] $\tau = 0.5111$
- 与模型发布时间 $\tau = 0.5958$
- 与模型参数量 $\tau = 0.6880$（最强）
- 与 [[Aesthetic Quality|美学质量]] $\tau = 0.0000$（完全无关 — 强力证据：CCI 测的是与视觉美学正交的能力）

---

## 关键图表

### Figure 1: Teaser / 总览

![Figure 1](https://arxiv.org/html/2605.30346v1/x1.png)

**说明**: YoCausal 的核心 motivation。展示"正向播放的真实视频" vs "时间反转视频"作为反事实样本的对照。一个真正具备 [[World Model|世界模型]] 能力的 VDM 应该对反向视频显著更"困惑"。图示典型例子：杯子摔碎（强因果）vs 鸟在飞（弱因果）—— 前者反向播放明显违反因果，后者反向看也合理。

### Figure 2: 框架总览

**说明**: 三段式框架——
1. **数据集构建**：对 4 个真实视频源做时间反转，得到 $(x^f, x^r)$ 配对。
2. **Level 1 (RSI)**：把正反样本喂给被测 VDM，比较 [[Denoising Loss|去噪损失]]。
3. **Level 2 (CCI)**：用 [[Vision-Language Model|VLM]] 把数据分层为 $\mathcal{D}_c$ / $\mathcal{D}_{nc}$，对比 RSI 差异。

### Table 1: 数据集组成

| 子集 | 来源 | 视频数 | 时长 | 特征 |
|------|------|--------|------|------|
| $\mathcal{D}_{\text{General}}$ | [[Moments in Time]] | 500 | 3s | 日常通用场景 |
| $\mathcal{D}_{\text{Physics}}$ | [[Physics IQ]] | 132 | 5s | 物理因果强 |
| $\mathcal{D}_{\text{Human}}$ | [[Kinetics-400]] | 400 | 3s | 人类动作 |
| $\mathcal{D}_{\text{Animal}}$ | [[Animal Kingdom]] | 200 | 3s | 动物行为 |
| **总计** | — | **1,232** | — | 可扩展 |

**说明**: 故意选择四类异质场景，让 RSI/CCI 能在不同因果密度的数据上做对比。Physics 子集因果密度最高（91.7% human RSI），Animal 最低（72.0%）。

### Table 2: Level 1 RSI 结果（按 Overall 升序）

| Model | General | Physics | Human | Animal | **Overall** |
|-------|---------|---------|--------|--------|-------------|
| AnimateDiff-SDXL | 27.80% | 41.67% | 48.73% | 46.50% | 41.18% |
| CogVideoX-2B | 33.20% | 40.15% | 56.64% | 36.00% | 41.50% |
| Wan2.1-T2V-1.3B | 29.80% | 59.09% | 59.15% | 34.00% | 45.51% |
| AnimateDiff-SD-1.5 | 33.00% | 32.58% | 61.68% | 55.50% | 45.69% |
| CogVideoX1.5-5B | 28.80% | 62.12% | 62.91% | 33.50% | 46.83% |
| Mochi-1-preview | 37.80% | 43.18% | 76.50% | 39.00% | 49.12% |
| CogVideoX-5B | 31.10% | 67.42% | 63.16% | 38.00% | 49.92% |
| Wan2.2-TI2V-5B | 34.40% | 71.97% | 63.75% | 37.50% | 51.91% |
| HunyuanVideo | 25.80% | 64.39% | 86.50% | 31.50% | 52.05% |
| Wan2.1-T2V-14B | 37.60% | 70.45% | 66.92% | 38.00% | 53.24% |
| Wan2.2-T2V-A14B | 36.80% | 77.27% | 66.17% | 36.50% | 54.19% |
| LTX-Video-13B | 61.20% | 47.73% | 47.50% | 69.50% | 56.48% |
| **LTX-Video-2B** | **58.60%** | 57.58% | 54.25% | **65.00%** | **58.86%** |
| 🧠 **Human Baseline** | **76.60%** | **91.70%** | **76.00%** | **72.00%** | **79.08%** |

**关键发现**:
- 最强模型 LTX-Video-2B 仍比 human 落后 **20.22%**，离"视频生成 = 世界模型"还有相当距离。
- Physics 子集上 human 达 91.7%，多数模型 < 70%，物理因果是 VDM 的弱项。
- Wan 系列在 Physics 上表现亮眼（A14B = 77.27%），说明模型规模在物理因果上有效。
- LTX-Video 在 General/Animal 上反常领先 —— 论文怀疑与其训练数据组成有关。

## 实验结果

### Table 3: Level 2 CCI 结果（按 CCI 升序，揭示因果方向）

| Model | RSI($\mathcal{D}_c$) | RSI($\mathcal{D}_{nc}$) | **CCI** | Normalized CCI |
|-------|------|------|------|----------------|
| AnimateDiff-SD-1.5 | 43.40% | 48.61% | **−5.21%** | −60.09% |
| AnimateDiff-SDXL | 38.93% | 44.00% | **−5.07%** | −58.48% |
| LTX-Video-13B | 54.65% | 58.97% | **−4.32%** | −49.83% |
| Wan2.2-TI2V-5B | 50.90% | 53.02% | **−2.12%** | −24.45% |
| HunyuanVideo | 51.15% | 51.44% | −0.29% | −3.34% |
| LTX-Video-2B | 57.95% | 58.15% | −0.20% | −2.31% |
| CogVideoX-2B | 41.11% | 40.18% | +0.93% | 10.73% |
| Mochi-1-preview | 49.11% | 45.26% | +3.85% | 44.41% |
| CogVideoX1.5-5B | 48.46% | 43.61% | +4.85% | 55.94% |
| CogVideoX-5B | 51.36% | 46.27% | +5.09% | 58.71% |
| Wan2.1-T2V-1.3B | 46.92% | 41.56% | +5.36% | 61.82% |
| Wan2.2-T2V-A14B | 55.73% | 50.22% | +5.51% | 63.55% |
| **Wan2.1-T2V-14B** | 54.80% | 48.89% | **+5.91%** | **68.17%** |
| 🧠 **Human Baseline** | **85.09%** | **76.42%** | **+8.67%** | **100.00%** |

**关键发现**（论文最具冲击力的结论）:
1. **"RSI 强 ≠ CCI 强"**: LTX-Video-2B 拿到 RSI 第一（58.86%），但 CCI 接近 0（−0.20%）—— 它感知到时间方向，但完全没区分"因果" vs "非因果"。
2. **CCI 负值出现了 4 个模型**: AnimateDiff 系列、LTX-13B、Wan2.2-TI2V-5B 的 CCI 是**负的**，意味着这些模型在非因果视频上反而表现更"困惑"——典型的浅层时序统计学习者。
3. **Wan2.1-T2V-14B 是 CCI 冠军**（68.17% normalized），但仍远未触及 human 水平（100%）。
4. **结论一句话**：**"Perceiving the arrow of time does not imply understanding causality."**

### Table 4: 与其他评测指标的 Kendall 一致性

| 对比维度 | $\tau$ | 解释 |
|----------|--------|------|
| 模型参数量 | 0.6880 | 强相关 — 规模有助于因果认知 |
| 模型发布时间 | 0.5958 | 显著 — 新模型更好（[[Scaling Law]]） |
| 人类偏好 | 0.5111 | 中等 — 与"好看"有一定关联 |
| LikePhys 物理理解 | 0.5111 | 中等 — 与物理 prior 部分重合 |
| **美学质量** | **0.0000** | **零相关 — CCI 测的是与美学完全正交的能力** |

**关键发现**: 美学相关性为零，是 YoCausal 最强力的存在证明——它确实在测一个不同维度的能力，而不是 VBench 的换皮。

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Moments in Time]] | 500 → 1000 (正反对) | 通用日常 | General 子集 |
| [[Physics IQ]] | 132 → 264 | 物理强因果 | Physics 子集 |
| [[Kinetics-400]] | 400 → 800 | 人类动作 | Human 子集 |
| [[Animal Kingdom]] | 200 → 400 | 动物行为 | Animal 子集 |

### 实现细节

- **被测模型**: 13 个开源 VDM（[[AnimateDiff]] x2 / [[CogVideoX]] x3 / [[Mochi-1]] / [[HunyuanVideo]] / [[Wan2.1]] x2 / [[Wan2.2]] x2 / [[LTX-Video]] x2）
- **去噪损失计算**: 与每个模型自身训练目标对齐（DDPM / [[Flow Matching]] 等不同 schedule）
- **VLM 因果分层**: Gemini 3.0 Pro，τ = 0.7613 与人工标注一致
- **Human baseline**: 通过让人工标注者执行同样的"判断哪个是正向"任务来估计

### 可视化结果

- **散点图**: RSI vs CCI，揭示两者的弱相关性（最佳 RSI 模型 LTX-2B 的 CCI 几乎为 0）
- **柱状图**: 13 个模型在 4 个子集上的 RSI 分布，Physics 子集差距最大
- **Failure case**: AnimateDiff 在因果视频上反而 loss 更低（CCI 负），说明它学的是"AnimateDiff 训练分布"，不是世界

---

## 批判性思考

### 优点

1. **方法论极简**：时间反转 = 反事实，零标注成本，可任意扩展数据。
2. **双层级设计干净**：RSI 先验测时间方向，CCI 再剥离因果，避免单一指标的歧义。
3. **强力的负向证据**：CCI 出现负值 + 与美学 $\tau = 0$ 是非常清晰的"模型没真学到因果"的实证。
4. **覆盖面广**：13 个 VDM 横扫，包含最新的 Wan2.2、Mochi、LTX，结论有时效性。

### 局限性

1. **闭源模型缺位**：Sora、Veo、Kling 都没测，结论的天花板不知道在哪。可能 [[Scaling Law]] 在闭源那一档表现完全不同。
2. **依赖训练分布**：RSI 基于去噪损失差，而去噪损失对训练数据分布高度敏感——LTX-Video 在 General/Animal 上的异常领先很可能是数据偏好，而非真因果能力。**跨模型可比性需要谨慎**。
3. **VLM 分层有偏**：因果 vs 非因果由 Gemini 判断，虽然 $\tau=0.7613$ 一致性高，但 VLM 自身可能继承训练偏置，影响 CCI 公平性。
4. **Human baseline 估计粗糙**：人工"判断哪个是正向"任务的设计细节、标注者数量、agreement 等可复现性不足。

### 潜在改进方向

1. **替换损失为模型不可知信号**：如用统一的 perceptual / classifier-based score，减少训练分布敏感性。
2. **加入 [[Action-Conditioned World Model|动作条件 WM]] 评测**：YoCausal 当前只测 T2V，但 WM 真正的用途在 robotics 上是 action-conditioned 的，应该有对应版本。
3. **细粒度因果类型**: 把因果子集拆成"物理因果 / 社会因果 / 意图因果"，看模型在哪一类先突破。
4. **加入更长视频**：当前 3-5 秒的视频可能不足以承载复杂因果链。

### 可复现性评估

- [x] 代码开源（GitHub）
- [x] 数据集开源（HuggingFace）
- [ ] 评测脚本细节（loss 实现是否完全对齐各模型训练目标？需要确认）
- [x] 数据集来源公开

---

## 关联笔记

### 基于

- [[Video Diffusion Model]]: 被测对象的基础类
- [[World Model]]: 评测目标 — 想知道 VDM 离 WM 多远
- [[Counterfactual]]: 评测的核心方法论 — 时间反转 = 反事实
- [[Violation of Expectation]]: 借鉴的认知科学范式

### 对比

- [[VBench]]: 主流 VDM 评测，测视觉质量；YoCausal 与之 $\tau \approx 0$ 在美学维度，证明测的是正交能力
- [[WBench]]: World model 评测套件，更偏 control 视角
- [[Gamma-World]]: 同期 WM 评测工作
- [[WAM-Survey]]: 同期世界模型综述

### 方法相关

- [[Reverse Surprise Index]]: 本文 Level 1 指标
- [[Causality Cognition Index]]: 本文 Level 2 指标
- [[Arrow of Time]]: RSI 测的核心概念
- [[Denoising Loss]]: 计算 RSI 的底层信号

### 被测模型

- [[AnimateDiff]]、[[CogVideoX]]、[[Mochi-1]]、[[HunyuanVideo]]、[[Wan2.1]]、[[Wan2.2]]、[[LTX-Video]]

### 数据相关

- [[Moments in Time]]、[[Physics IQ]]、[[Kinetics-400]]、[[Animal Kingdom]]

### 工具相关

- [[Vision-Language Model]]: Gemini 3.0 Pro 作为因果分层判官
- [[VLM-as-Judge]]: VLM 评测范式
- [[Kendall Tau]]: 一致性度量

---

## 速查卡片

> [!summary] YoCausal
> - **核心**: 用时间反转作为反事实，双层级探针测 VDM 的因果理解能力
> - **方法**: RSI（去噪损失差测时间方向）+ CCI（因果/非因果子集 RSI 差测真因果）
> - **结果**: 最强 VDM CCI 仅达 human 的 68%，4 个模型 CCI 为负；与美学 $\tau=0$
> - **关键结论**: "Perceiving the arrow of time does not imply understanding causality"
> - **代码**: https://github.com/youzhe0305/YoCausal

---

*笔记创建时间: 2026-05-30*
