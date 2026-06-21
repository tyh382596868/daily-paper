---
title: "BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression"
method_name: "BiWM"
authors: [Shaohao Rui, Xiaofeng Mao, Zhanyu Zhang, Peijia Lin, Yansong Zhu, Yibo Zhang, Haibin Wan, Weijie Ma]
year: 2026
venue: arXiv
tags: [interactive-world-model, bidirectional-autoregression, video-diffusion, diffusion-distillation, camera-control, world-model, open-source]
zotero_collection: 1-生成模型
image_source: pending
arxiv_html: https://arxiv.org/html/2606.10135
created: 2026-06-21
---

# 论文笔记：BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression

> ⚠️ **注意**：论文提交后，作者发现部分可视化结果因运行时配置错误而产生，已标注待撤回并会提交修正版本。定量数字结论整体成立，视觉对比结果需谨慎参考。

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | LynnReal AI / 上海创新研究院 / 上海交通大学 / 复旦大学 |
| 日期 | June 2026（arXiv:2606.10135，提交于 2026-06-09） |
| 项目主页 | 暂未公开 |
| 对比基线 | [[minWM]]、[[Yume]]（Yume-1.5）、Matrix-Game-3.0 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.10135) / [HTML](https://arxiv.org/html/2606.10135) |

---

## 一句话总结

> BiWM 是首个面向[[交互式世界模型]]的双向自回归开源全栈框架，仅需两个训练阶段（相机控制微调 + [[DMD|Distribution Matching Distillation]] 蒸馏），同时逼近闭源双向模型的生成质量并保持低延迟自回归推理。

---

## 核心贡献

1. **首个双向自回归开源全栈框架**: 填补了开源[[交互式世界模型]]在[[Bidirectional Autoregressive Diffusion|双向自回归范式]]下的空白（此前 [[minWM]] 仅支持因果模型）。
2. **精简两阶段流水线**: 相比 [[minWM]] 的四阶段，压缩至"相机/动作控制微调 → 少步自回归 [[DMD]] 蒸馏"两阶段，仅需在 8×H200 GPU 上数百优化步即可收敛。
3. **抗退化 DMD 目标**: 针对 [[DMD]] 的 mode-seeking 退化问题，引入 GAN 对抗精化项 + 前向 KL 正则化锚点（mass-covering），有效保留场景运动多样性。
4. **可插拔历史压缩机制**: 同时集成 FramePack 式（近密远疏压缩）和 PackForcing 式两种历史压缩方案，降低长时序推理的内存与计算开销。
5. **跨骨干泛化能力**: 支持 Wan2.1-T2V-1.3B、Wan2.2-TI2V-5B、HunyuanVideo-1.5-TI2V-8B、LTX-2.3-22B 四种主流视频扩散骨干，及对 Yume-1.5 / Matrix-Game-3.0 等已有双向模型进行二次微调。

---

## 问题背景

### 要解决的问题

把高质量的**双向视频扩散模型**转化为能够实时交互的[[交互式世界模型]]，同时保持生成质量并实现可控推理。

### 现有方法的局限

**因果自回归方案（如 [[minWM]]）的问题**：
- 流水线复杂（四个阶段）：相机控制微调 → [[自回归扩散训练]] → 因果初始化 → 少步蒸馏
- 因果模型固有误差累积：帧 $t$ 的表示在生成后锁定，错误不可修正，长时序稳定性差
- 真实世界相机控制场景下易失去可控性

**双向模型（如 Yume-1.5、Matrix-Game-3.0）的优势**：
- 窗口内帧之间双向注意力，早期帧表示可根据后续帧修正（self-correcting）
- 生成质量更高，长时序探索更稳定

**开源缺口**：现有开源框架（如 [[minWM]]）仅支持因果模型，双向自回归范式缺乏开源全栈实现。

### 本文的动机

双向模型的优势来自**窗口内全注意力的自我纠错机制**，自回归代价只在窗口之间跨越。BiWM 认为可以直接基于预训练双向视频骨干，通过极简两阶段训练实现交互化，无需先转化为因果架构。

---

## 方法详解

### 模型架构

BiWM 采用 **双向自回归扩散**（[[Bidirectional Autoregressive Diffusion]]）架构：

- **输入**: 初始帧 $I_0$ + 相机轨迹 / 动作控制信号 $a_t$ + 历史压缩上下文 $h_{<w}$
- **Backbone**: 预训练双向视频扩散 Transformer（Wan2.1 / Wan2.2 / HunyuanVideo-1.5 / LTX-2.3）
- **注意力机制**: 窗口内保持骨干原生[[全注意力]]（Full Attention），窗口间通过历史压缩机制实现跨窗口条件化
- **输出**: 当前窗口内的视频帧潜变量 $z_{t:t+W}$（$W$ 为窗口大小）
- **推理模式**: 自回归逐窗口生成，支持少步推理（few-step DMD 蒸馏后）

### 核心模块

#### 模块 1: 相机/动作控制微调（Stage 1）

**设计动机**: 为双向扩散骨干注入相机轨迹或动作控制能力，同时保留视频生成质量。

**具体实现**:
- 保持骨干全双向注意力权重不变
- 新增控制信号编码器，将相机姿态 / Plücker 坐标 / 动作向量映射到与骨干特征同维度的条件嵌入
- 监督目标与骨干预训练一致（[[扩散损失|Flow Matching / DDPM 损失]]）

#### 模块 2: 双向自回归 DMD 蒸馏（Stage 2）

**设计动机**: 将 Stage 1 的多步双向模型压缩为少步（few-step）自回归推理，同时通过[[DMD|Distribution Matching Distillation]]对齐生成分布。

**具体实现**:
- 以 Stage 1 模型为教师（teacher），蒸馏为学生（student）少步生成器
- 使用**自回归自展 DMD**（self-rollout DMD）：学生在自回归推理的上下文中展开，对齐窗口级分布
- **抗退化目标**（见下方公式）：对抗 DMD 的 mode-seeking 问题

#### 模块 3: 可插拔历史压缩机制

**设计动机**: 长时序推理时，历史窗口的 KV 缓存随时间线性增长，需压缩。

两种方案（可独立切换）:
- **FramePack 式**: 近距历史高分辨率保留，远距历史按时间距离指数级下采样（最远约 32× 压缩），基于 [[Yume]] 的 FramePack 设计
- **PackForcing 式**: 低码率分支编码历史，作为上下文条件，来源于 [[Causal Forcing|PackForcing]] 机制

---

## 关键公式

### 公式 1: [[DMD|标准 DMD 目标]]

$$
\mathcal{L}_{\mathrm{DMD}} = \mathbb{E}_{t, x_0 \sim p_\theta} \left[ w(t) \left\| \nabla_{x_t} \log p_\phi(x_t) - \nabla_{x_t} \log q(x_t) \right\|^2 \right]
$$

**含义**: 使学生生成分布 $p_\theta$ 与真实数据分布 $q$ 的得分函数差异最小化，其中通过预训练扩散教师 $p_\phi$ 估计 $q$ 的得分。

**符号说明**:
- $p_\theta$: 学生（少步生成器）的分布
- $p_\phi$: 教师扩散模型（Stage 1 的双向相机控制模型）
- $q(x_t)$: 真实视频数据在噪声水平 $t$ 下的边际分布
- $w(t)$: 时间步权重函数

### 公式 2: [[DMD|抗退化 DMD 总目标]]

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{DMD}} + \lambda_{\mathrm{GAN}} \mathcal{L}_{\mathrm{GAN}} + \lambda_{\mathrm{KL}} \mathcal{L}_{\mathrm{forward\text{-}KL}}
$$

**含义**: 在标准 DMD 目标上叠加 GAN 对抗精化项和前向 KL 正则化锚点，联合解决 mode-seeking 退化和运动多样性坍塌。

**符号说明**:
- $\mathcal{L}_{\mathrm{GAN}}$: GAN 判别器损失，对抗精化生成细节
- $\mathcal{L}_{\mathrm{forward\text{-}KL}}$: 前向 KL 散度（mass-covering），惩罚学生忽略真实分布中的高概率区域，防止动态场景坍塌为静态均值
- $\lambda_{\mathrm{GAN}}, \lambda_{\mathrm{KL}}$: 权重系数

### 公式 3: [[Bidirectional Autoregressive Diffusion|双向自回归条件分解]]

$$
p(x_{1:T}) = \prod_{w=0}^{W_{\max}} p_\theta\!\left(x_{(w \cdot L):(w+1) \cdot L} \;\Big|\; h_{<w}\right)
$$

**含义**: 视频帧序列按窗口 $L$ 分组自回归生成，每个窗口内帧保持双向注意力（完整窗口内互相可见），窗口间仅依赖历史压缩上下文 $h_{<w}$。

**符号说明**:
- $L$: 窗口大小（帧数）
- $h_{<w}$: 前 $w$ 个窗口的历史压缩表示（FramePack 或 PackForcing 格式）
- $W_{\max}$: 总窗口数

---

## 关键图表

> 🖼️ **Figure 1: 流水线对比概览** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: 对比 [[minWM]] 四阶段因果流水线与 BiWM 两阶段双向自回归流水线的训练步骤差异。BiWM 省去"因果自回归训练"和"因果初始化"两步，直接在双向骨干上执行相机控制微调后做 DMD 蒸馏。

> 🖼️ **Figure 2: 双向 vs 因果注意力示意图** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: 展示窗口内双向注意力（早期帧可被后续帧修正）vs 因果注意力（帧一旦生成表示锁定）的区别，直观说明误差累积的来源。

> 🖼️ **Figure 3: 历史压缩机制对比（FramePack vs PackForcing）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: 对比两种可插拔历史压缩方案在内存占用和长时序一致性上的权衡。

> 🖼️ **Figure 4: 抗退化效果消融** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: 展示纯 DMD、DMD+GAN、DMD+GAN+前向KL 三种目标下生成视频的动态多样性对比，验证抗退化模块的必要性。

> 🖼️ **Figure 5: 多骨干生成效果展示** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: BiWM 在 Wan2.1、Wan2.2、HunyuanVideo-1.5、LTX-2.3 四种骨干上的相机/动作可控视频生成效果。

> 🖼️ **Figure 6: 相机可控性对比（BiWM vs minWM）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.10135)）

**说明**: 在真实世界相机轨迹控制场景下，[[minWM]] 出现可控性失效，而 BiWM 维持了准确的相机跟随。

### Table 1: 与主流世界模型的定量对比

> ⚠️ 注意：作者已告知部分可视化结果因配置错误产生，量化结果可靠性有待修正版确认。

| 方法 | 范式 | 训练阶段数 | 骨干 | 开源 | 相机控制 | 备注 |
|------|------|-----------|------|------|---------|------|
| [[Yume]]-1.5 | 双向自回归 | — | 私有 | ❌ | ✅ | 闭源 |
| Matrix-Game-3.0 | 双向自回归 | — | 私有 | ❌ | ✅ | 闭源 |
| [[minWM]] | 因果自回归 | 4 | Wan2.1/HY1.5 | ✅ | ✅ | 开源，仅因果 |
| **BiWM（本文）** | **双向自回归** | **2** | Wan2.1/Wan2.2/HY1.5/LTX-2.3 | **✅** | **✅** | **首个开源双向全栈** |

**关键发现**: BiWM 是目前唯一同时满足"双向自回归 + 全栈开源 + 多骨干支持"的[[交互式世界模型]]框架。

### Table 2: 训练效率对比

| 方法 | 阶段数 | 收敛步数 | 硬件 |
|------|--------|---------|------|
| [[minWM]] | 4 | 较长（各阶段分别训练） | 未公开 |
| **BiWM** | **2** | **数百步**（两阶段联合收敛） | **8×H200** |

**关键发现**: BiWM 大幅降低训练成本，两个阶段均保持双向注意力，无需中间态架构转换。

---

## 实验

### 支持的骨干模型

| 骨干模型 | 参数量 | 类型 | 特点 |
|---------|--------|------|------|
| [[Wan2.1]]-T2V-1.3B | 1.3B | T2V | 轻量，适合快速验证 |
| Wan2.2-TI2V-5B | 5B | TI2V | 图像条件，中量级 |
| HunyuanVideo-1.5-TI2V-8B | 8B | TI2V | 高质量，重量级 |
| LTX-2.3-22B | 22B | T2V/TI2V | 最大规模骨干 |

### 实现细节

- **训练硬件**: 8×H200 GPU
- **Stage 1（相机控制微调）**: 数百优化步，保持骨干全双向注意力
- **Stage 2（DMD 蒸馏）**: 数百优化步，直接在双向骨干上执行 self-rollout DMD
- **历史压缩**: FramePack 和 PackForcing 均作为可插拔模块，可独立配置

### 支持的场景

- **真实世界相机控制**: 给定相机姿态轨迹（Plücker 坐标或旋转矩阵）控制视角
- **动作可控**: 支持机器人/游戏等动作信号条件化生成
- **二次微调**: 对已有双向闭源模型（Yume-1.5, Matrix-Game-3.0）进行新数据分布下的迁移训练

### 可视化结果

- BiWM 在真实相机轨迹控制任务上显著优于 [[minWM]]（后者常出现可控性失效）
- 长时序展开中，双向自回归的自纠错机制有效抑制帧漂移
- 支持从 1.3B 到 22B 规模的跨骨干生成，质量随模型规模提升

---

## 批判性思考

### 优点

1. **范式创新且填补空白**: 首次将双向自回归世界模型开源，推动社区跟进闭源 Yume/Matrix-Game 的技术路线。
2. **工程精简**: 两阶段方案大幅降低复现门槛，数百步即可收敛对开源社区友好。
3. **抗退化设计有针对性**: GAN + 前向KL 组合直接解决了[[DMD]]已知的 mode-seeking 痛点，思路清晰。
4. **多骨干泛化**: 四种主流骨干的统一接口验证了方法的架构无关性。

### 局限性

1. **可视化结果存在配置错误问题**: 作者已标注撤回并提交修正版，视觉对比结论的可信度需等待修正版确认。
2. **定量指标不完整**: 搜索到的信息中，针对 WorldMark 等标准 benchmark 的具体数值未能获取，对比基线限于定性描述。
3. **训练数据未完整披露**: 用于相机控制微调的数据集构建细节（如 [[minWM]] 对 DL3DV / WorldPlay 的详细记录）暂不明确。

### 潜在改进方向

1. 引入更强的时序一致性约束（如光流监督）进一步提升双向窗口边界处的帧连续性
2. 探索窗口大小 $L$ 的自适应调整，针对动态场景扩大窗口，静态场景压缩
3. 将[[扩散模型|扩散]]骨干替换为离散 [[VQ-VAE]] tokenizer 路线，探索与自回归 LLM 的统一架构

### 可复现性评估

- [x] 代码开源（论文声明开源）
- [ ] 预训练模型（待确认）
- [x] 训练细节完整（两阶段、8×H200、数百步）
- [x] 数据集可获取（使用公开骨干模型）

---

## 关联笔记

### 基于

- [[Wan2.1]]: 主要使用的轻量骨干（1.3B T2V）
- [[Wan2.2]]: 图像条件骨干（5B TI2V）
- [[DMD]]: Stage 2 蒸馏的核心技术，[[少步蒸馏]]的实现方式
- [[Bidirectional Autoregressive Diffusion]]: 本文采用的核心生成范式
- [[FramePack]]: 历史压缩机制来源之一（Yume 提出）

### 对比

- [[minWM]]: 直接竞品，BiWM 将四阶段因果方案替换为两阶段双向方案
- [[Yume]]: 闭源双向自回归模型，BiWM 的开源对标目标，支持对 Yume-1.5 进行二次微调
- [[Matrix-Game]]: 另一闭源双向模型，同样可作为 BiWM 的二次微调基座

### 方法相关

- [[交互式世界模型]]: 本文的应用场景定义
- [[相机可控微调]]: Stage 1 的核心操作
- [[少步蒸馏]]: Stage 2 DMD 蒸馏实现少步推理
- [[视频扩散模型]]: BiWM 所基于的底层技术类别

### 硬件/数据相关

- HunyuanVideo-1.5-TI2V-8B: 支持的最高质量骨干
- LTX-2.3-22B: 支持的最大规模骨干

---

## 速查卡片

> [!summary] BiWM: Bidirectional Autoregressive Video World Model
> - **核心**: 首个双向自回归交互世界模型开源全栈框架，两阶段即可从预训练骨干构建相机/动作可控世界模型
> - **方法**: Stage 1 相机控制微调 + Stage 2 抗退化 DMD 蒸馏（GAN + 前向KL），保持全双向注意力
> - **结果**: 支持 Wan2.1/Wan2.2/HunyuanVideo-1.5/LTX-2.3 四种骨干，8×H200 数百步收敛
> - **注意**: 部分可视化结果存在配置错误，修正版待发布

---

*笔记创建时间: 2026-06-21*
