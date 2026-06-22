---
title: "Kairos: A Native World Model Stack for Physical AI"
method_name: "Kairos"
authors: [Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu]
year: 2026
venue: arXiv
tags: [world-model, physical-ai, video-generation, hybrid-attention, embodied-ai, robot-learning, flow-matching]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.16533
created: 2026-06-22
---

# 论文笔记：Kairos: A Native World Model Stack for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Kairos Team (SenseNova) |
| 日期 | June 2026 |
| 项目主页 | https://github.com/kairos-agi/kairos-sensenova |
| 对比基线 | [[Cosmos3]]、[[V-JEPA 2]]、[[HY-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.16533) / [Code](https://github.com/kairos-agi/kairos-sensenova) / [HuggingFace](https://huggingface.co/kairos-agi) |

---

## 一句话总结

> Kairos 是首个面向 Physical AI 的原生世界模型栈，通过跨体态数据课程、混合线性时序注意力和部署感知协同设计，实现从视频生成到具身控制的统一基础设施。

---

## 核心贡献

1. **跨体态数据课程（CEDC）**: 将异构经验（开放世界视频 → 人类行为数据 → 机器人交互数据）组织为三阶段递进式训练，替代平铺式数据扩展
2. **混合多尺度时序记忆架构**: 统一 [[滑动窗口注意力|SWA]]（局部）+ [[扩张滑动窗口注意力|DSWA]]（中程）+ [[门控线性注意力|GLA]]（全局持久记忆）三种机制，并提供理论误差界证明
3. **部署感知系统协同设计**: 通过时间步蒸馏、权重量化、Token 流式传输，实现在消费级硬件上的实时闭环推理

---

## 问题背景

### 要解决的问题

现有世界模型主要作为"被动视觉生成器"——只能生成视频，无法真正支撑 Physical AI 的决策-行动-反馈闭环。

### 现有方法的局限

- **纯局部注意力**（如 SWA）：对长时序状态跟踪能力不足，无法记忆超出上下文窗口的事件（Theorem 1 给出了理论证明）
- **平铺数据扩展**：混合视频 + 机器人数据不加区分，缺乏从通用物理知识到具身技能的层次结构
- **生成模型与控制解耦**：世界模型和动作预测模块分开设计，推理效率低、联合优化困难

### 本文的动机

世界模型正在从"演示工具"转变为"可操作基础设施"。Kairos 的定位是：让世界模型同时支撑理解（VLM）、生成（Video DiT）和预测（World-Action Model），并以统一架构跑在实际硬件上。

---

## 方法详解

### 模型架构

Kairos 采用三模块统一架构：

- **世界理解模块（World Understanding）**：基于 Qwen 系列 [[VLM]] 提取跨模态语义表示
- **世界生成模块（World Generation）**：高压缩视频 [[VAE]] + 多模态条件编码器 + 带混合注意力的 [[DiT]] 主干
  - 支持 T2V、I2V、TI2V 三种生成模式
- **世界预测模块（World-Action Model）**：[[混合专家 Transformer|MoT]] 架构，Video DiT + Action DiT 联合建模
  - Action DiT 约为 Video DiT 的 1/5 规模，高效推理

总体上是一个"理解-生成-预测"统一栈（Native World Model Stack）。

### 核心模块

#### 模块 1: 混合线性时序注意力（Hybrid Linear Temporal Attention）

**设计动机**：单一注意力机制无法同时建模局部动态、中程依赖和全局长时记忆。利用 [[线性注意力]] 的 O(L) 复杂度实现高效长序列建模。

三种机制互补组合：

**1. [[滑动窗口注意力|SWA]]（局部动态）**

在时间维度上每帧长度 $L$ 个 token 为基本单位，窗口大小 $w = L \times \text{window\_size}$：

$$
\mathrm{SWA}(Q, K, V)_i = \sum_{j \in [i - w/2,\, i + w/2]} \mathrm{Softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) V_j
$$

常数复杂度，专门捕获帧内及相邻帧的局部运动。

**2. [[扩张滑动窗口注意力|DSWA]]（中程依赖）**

以扩张因子 $d$ 沿时间维度稀疏采样，将 $(B, F \cdot L, D)$ reshape 为 $(B \cdot d,\; F/d \cdot L,\; D)$，再在新维度上做 SWA，从而用常数成本捕获中程跳帧关联。

**3. [[门控线性注意力|GLA]] / GatedDeltaNet（全局持久记忆）**

首先提取特征：

$$
q_t = W_Q x_t, \quad k_t = W_K x_t, \quad v_t = W_V x_t, \quad \beta_t = \sigma(W_\beta x_t)
$$

内存检索与插值（Delta 机制）：

$$
v_t^{\mathrm{old}} = S_{t-1} k_t, \qquad v_t^{\mathrm{new}} = \beta_t v_t + (1 - \beta_t) v_t^{\mathrm{old}}
$$

带衰减门的 Delta 状态更新：

$$
\alpha_t = \sigma(W_\alpha x_t), \qquad S_t = \alpha_t S_{t-1} - v_t^{\mathrm{old}} k_t^\top + v_t^{\mathrm{new}} k_t^\top
$$

其中 $\alpha_t < 1$ 保证收缩性，使单步扰动随时间严格衰减而非放大。

#### 模块 2: 世界-动作模型（World-Action Model）

**设计动机**：联合建模未来视频帧和未来动作 Token，使控制策略可从世界模型的内部表示中直接预测动作。

**具体实现**：

- 联合 Token 组：历史视频 Token → 未来视频 Token + 未来动作 Token
- 未来视频 Token：稀疏时空注意力
- 未来动作 Token：全注意力（保证规划连贯性）
- ActionDiT 权重由 VideoDiT 插值初始化，加速收敛
- 推理时动作预测可脱离视频生成独立运行，大幅降低延迟

**联合损失**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{video}} + \lambda \cdot \mathcal{L}_{\mathrm{action}}
$$

---

## 关键公式

### 公式 1: [[Flow Matching|Flow Matching 训练目标]]

插值参数化：

$$
z_\sigma = (1 - \sigma) z_0 + \sigma \varepsilon
$$

**真实速度场**：

$$
u_\sigma = \frac{d z_\sigma}{d\sigma} = \varepsilon - z_0
$$

**Flow Matching 损失**：

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{z_0, \varepsilon, \sigma, c}\!\left[\| V_\theta(z_\sigma, \sigma, c) - u_\sigma \|_2^2\right]
$$

**含义**：令网络预测的速度场与插值真实速度对齐，从噪声 $\varepsilon$ 到干净帧 $z_0$ 的直线轨迹。

**符号说明**：
- $z_0$: 干净视频潜码
- $\varepsilon$: 高斯噪声
- $\sigma \in [0,1]$: 插值/噪声比例
- $c$: 条件（文本、图像等）
- $V_\theta$: 网络预测的速度场

### 公式 2: [[时间步分布偏移|形状感知指数时间步偏移]]

**指数偏移变换**：

$$
\tilde{\sigma}_i = \frac{s \cdot \sigma_i^{(0)}}{1 + (s-1)\sigma_i^{(0)}}
$$

等价 logit 空间形式：

$$
\tilde{\sigma}_i = \mathrm{sigmoid}\!\left(\mathrm{logit}(\sigma_i^{(0)}) + \log s\right)
$$

**偏移强度（形状感知）**：

$$
s = \exp(f(L)) \cdot \sqrt{F}
$$

其中线性映射 $f(L) = mL + b$，参数为：

$$
m = \frac{r_{\max} - r_{\min}}{L_{\max} - L_{\min}}, \qquad b = r_{\min} - m \cdot L_{\min}
$$

**含义**：根据视频分辨率 $L$（每帧 token 数）和帧数 $F$ 动态调整时间步采样分布，使训练更关注高噪声区段，提升长视频和高分辨率训练稳定性。

**符号说明**：
- $\sigma_i^{(0)}$: 原始时间步
- $\tilde{\sigma}_i$: 变换后时间步
- $s$: 偏移强度
- $L$: 每帧 token 数
- $F$: 总帧数
- $r_{\min}, r_{\max}$: 偏移强度范围

### 公式 3: [[世界模型理论界|长时序超额风险上界]]（Theorem 2）

设混合预测器以误差 $\varepsilon$ 近似各组件，GLA 使用收缩因子 $\rho < 1$ 的门控 Delta 更新：

$$
\mathcal{R}_t(\hat{\mu}_t) - \mathcal{R}_t^\star \;\leq\; \left(L\varepsilon + \frac{L_G \bar{\xi}}{1 - \rho}\right)^{\!2}, \quad t \to \infty
$$

**含义**：混合架构的长时序超额风险有**有限上界**——收缩性保证单步误差不随时间放大，而是以几何级数衰减至稳态。

**符号说明**：
- $\mathcal{R}_t^\star$: 贝叶斯最优风险
- $L, L_G$: 解码器 Lipschitz 常数
- $\varepsilon$: 近似误差
- $\bar{\xi}$: 平均扰动幅度
- $\rho$: 收缩因子（$\rho < 1$）

### 公式 4: [[局部注意力下界|纯局部模型超额风险下界]]（Corollary 1）

$$
\mathcal{R}_w^\star - \mathcal{R}_{\mathrm{full}}^\star \;\geq\; \mathbb{P}(E)\, \alpha(1-\alpha)(\mu_1 - \mu_2)^2
$$

**含义**：当存在超出窗口 $w$ 的长程依赖事件 $E$ 时，纯局部模型（SWA only）的风险严格大于全局模型，差距由事件概率和均值差距下界决定。

**符号说明**：
- $w$: 注意力窗口大小
- $E$: 超出窗口的长程依赖事件
- $\alpha$: 条件概率
- $\mu_1, \mu_2$: 两种状态的条件期望

---

## 关键图表

### Figure 1: Kairos 动机框架

![Figure 1: Motivation of Kairos](https://arxiv.org/html/2606.16533v2/figures/motivation_v1.png)

**说明**：展示 Kairos 的核心定位——从"被动视觉生成器"转变为 Physical AI 的可操作基础设施。世界模型需要同时支撑感知理解、视频生成和具身控制三项能力。

### Figure 2: Kairos 完整框架

![Figure 2: Framework of Kairos](https://arxiv.org/html/2606.16533v2/figures/framework_new_v1.png)

**说明**：完整的 Kairos 框架图，展示世界理解（VLM）、世界生成（Video DiT with Hybrid Attention）、世界预测（MoT: Video DiT + Action DiT）三模块的统一结构，以及 [[CEDC|跨体态数据课程]] 的三阶段数据流。

### Figure 3: 性能对比图

![Figure 3a](https://arxiv.org/html/2606.16533v2/x1.png)

![Figure 3b](https://arxiv.org/html/2606.16533v2/x2.png)

![Figure 3c](https://arxiv.org/html/2606.16533v2/x3.png)

![Figure 3d](https://arxiv.org/html/2606.16533v2/x4.png)

![Figure 3e](https://arxiv.org/html/2606.16533v2/x5.png)

![Figure 3f](https://arxiv.org/html/2606.16533v2/x6.png)

**说明**：(a)(b) 具身世界模型 Benchmark 性能对比（WorldModelBench、DreamGen Bench）；(c)(d) 世界-动作模型性能对比（LIBERO-plus、RoboTwin 2.0）；(e)(f) 推理时间线性扩展曲线，展示 Kairos 在不同序列长度下推理时间的线性增长（得益于线性注意力）。

### Figure 4: 模型架构详图

![Figure 4: Model Architecture of Kairos](https://arxiv.org/html/2606.16533v2/figures/kairos_arch1.jpg)

**说明**：详细展示 DiT block 结构，包括 [[混合线性时序注意力]] 的三种注意力机制如何在 block 内部组合，以及视频 Token 与动作 Token 的联合处理流程。

### Figure 5: DiT Block 混合注意力结构

![Figure 5: DiT block with hybrid linear attention](https://arxiv.org/html/2606.16533v2/x7.png)

**说明**：单个 DiT block 的内部结构，SWA（局部）、DSWA（中程扩张）和 GLA（全局）三个注意力分支并行作用于同一序列，各自捕获不同时间尺度的依赖关系。

### Figure 6: GatedDeltaNet（GLA 实现）架构

![Figure 6: GatedDeltaNet architecture](https://arxiv.org/html/2606.16533v2/x8.png)

**说明**：[[门控线性注意力|GLA]] 的具体实现架构，展示 Delta 状态更新机制：通过 $\beta_t$ 控制新旧值插值、$\alpha_t$ 门控实现收缩性，使全局记忆状态 $S_t$ 在无限长序列上保持有界。

### Figure 7: 跨体态数据课程（CEDC）可视化

![Figure 7: Cross-Embodiment Data Curriculum](https://arxiv.org/html/2606.16533v2/x9.png)

**说明**：三阶段递进式训练数据流。Phase I（数百万小时开放世界视频）→ Phase II（~10 万小时人类操作演示）→ Phase III（目标体态机器人交互数据），每阶段在前一阶段学到的物理/行为先验基础上继续细化。

### Figure 8: 形状感知时间步分布偏移曲线

![Figure 8: Timestep shifting curves](https://arxiv.org/html/2606.16533v2/x10.png)

**说明**：不同训练阶段（分辨率 256P→720P，帧数 1→241）下时间步分布的动态变化。随着分辨率和帧数提升，偏移强度 $s$ 增大，训练更多关注高噪声时间步，提升长视频生成稳定性。

### Figure 9: 跨体态生成样例

![Figure 9: Kairos cross-embodiment samples](https://arxiv.org/html/2606.16533v2/figures/kairos_2_compress.jpg)

**说明**：Kairos 在四类体态上的生成结果：单臂机械臂、双臂机器人、灵巧手、人形机器人。展示统一世界模型跨越不同具身形态的泛化能力。

### Table 1: 渐进式物理预训练阶段配置

| 任务类型 | 阶段 | 分辨率 | 最大帧数 |
|----------|------|--------|---------|
| T2I/N2I | 图像预训练 | 256P | 1 |
| T2V/TI2V/I2V/N2V | 预训练（低分辨率） | 256P | 81 |
| T2V/TI2V/I2V/N2V | 预训练（中分辨率） | 480P | 81 |
| T2V/TI2V/I2V/N2V | 预训练（高分辨率） | 720P | 81 |
| T2V/TI2V/I2V/N2V | 持续训练（长序列） | 720P | 241 |
| T2V/TI2V/I2V/N2V | 领域 SFT + 模型融合 | 720P | 241 |
| T2V/TI2V/I2V/N2V | RL 精调 | 720P | 241 |

**说明**：分辨率和帧数的渐进式提升避免了直接在高分辨率长序列上训练的不稳定性；每阶段对应独立的学习率与 weight decay 配置（详见正文）。

---

## 实验

### 数据集

| 数据集 | 来源类型 | 特点 | 用途 |
|--------|---------|------|------|
| Koala-36M | 开源 | 大规模通用视频 | Phase I 预训练 |
| Openhumanvid | 开源 | 人类行为视频 | Phase II 人类中心训练 |
| VidGen | 开源 | 视频生成数据 | Phase I 预训练 |
| AgiBotWorld-Beta | 开源 | 机器人操作数据 | Phase III 具身训练 |
| Droid | 开源 | 机器人数据集 | Phase III 具身训练 |
| 自有互联网数据 | 专有 | 4 大领域分类（人类/机器人/通用场景/物理现象）| 全阶段预训练 |
| 自有第一人称数据 | 专有 | 人类操作 ego-centric 数据 | Phase II/III |

**总规模**：数百万小时有效原始视频

### 实现细节

- **生成骨干**: 时间可扩展 [[DiT]]（混合线性注意力版本）
- **视觉编码器**: 高压缩视频 [[VAE]]
- **语言理解**: Qwen 系列 [[VLM]]
- **优化器**: AdamW，学习率 $1 \times 10^{-6}$ (RL) ~ $5 \times 10^{-5}$ (图像预训练)
- **训练策略**: [[Flow Matching]] + [[DPO]] 精调 + 模型融合（Model Soup, CART, TIES, DARE, WUDI-Merging）
- **推理加速**: 时间步蒸馏 + 权重量化 + Token 流式传输 + CUDA 专用核

### Benchmark 评估

- **具身世界模型**: WorldModelBench、DreamGen Bench（物体持久性、物理预测、动作条件动态）
- **世界-动作模型**: LIBERO-plus、RoboTwin 2.0（轨迹预测、策略学习、操作任务）
- **推理效率**: 线性时间扩展验证（与序列长度正比，对比二次方复杂度方法）

---

## 批判性思考

### 优点

1. **理论严谨性**：Theorem 1 & 2 给出了局部注意力不足的下界证明和混合架构的有限误差上界，不只是工程 trick
2. **统一架构**：理解、生成、预测三模块共享主干，避免多系统拼接带来的分布偏移问题
3. **部署闭环**：从训练到推理到消费级硬件量化全链路考虑，是少数明确针对实际部署场景的世界模型工作
4. **数据课程设计**：CEDC 的三阶段递进符合从通用到专用的迁移学习直觉，Phase II 的 ~10 万小时人类演示是罕见的大规模人类行为先验注入

### 局限性

1. **专有数据依赖**：大量训练数据为自有数据，可复现性受限
2. **Benchmark 数值未完整披露**：论文主要以图表形式展示性能对比，未提供完整数值表格，难以精确对比
3. **MoT 架构复杂性**：Video DiT + Action DiT 的联合训练需要精细的权重初始化策略（插值），对工程实现要求高

### 潜在改进方向

1. 将 CEDC 课程策略推广到其他领域（自动驾驶、手术机器人）
2. 研究不同 $\rho$ 收缩因子对不同任务长度的影响，动态调整
3. 探索 GLA 与 Mamba/SSM 架构的对比，是否能进一步降低内存占用

### 可复现性评估

- [x] 代码开源（GitHub: kairos-agi/kairos-sensenova）
- [x] 预训练模型（HuggingFace: kairos-agi）
- [ ] 训练细节完整（部分超参数公开，但专有数据集构成不透明）
- [ ] 数据集可获取（开源子集可获取，专有数据不可获取）

---

## 关联笔记

### 基于

- [[Flow Matching]]: 视频生成的训练目标
- [[DiT]]: 主干扩散 Transformer 架构
- [[门控线性注意力|GatedDeltaNet]]: 全局记忆的核心算子
- [[VLM]]: 世界理解模块基础

### 对比

- [[Cosmos3]]: NVIDIA 世界模型，同为 Physical AI 定位
- [[V-JEPA 2]]: Meta 的联合嵌入预测架构世界模型
- [[HY-World]]: HunyuanVideo 系世界模型

### 方法相关

- [[混合线性时序注意力]]: 核心架构创新
- [[跨体态数据课程|CEDC]]: 训练范式创新
- [[时间步分布偏移]]: 训练稳定性技巧
- [[混合专家 Transformer|MoT]]: 世界-动作联合建模架构

### 硬件/数据相关

- [[AgiBotWorld-Beta]]: 机器人具身训练数据集
- [[Droid]]: 开源机器人操作数据集

---

## 速查卡片

> [!summary] Kairos: A Native World Model Stack for Physical AI
> - **核心**: 将世界模型从视频生成工具升级为 Physical AI 可操作基础设施
> - **方法**: 混合线性时序注意力（SWA+DSWA+GLA）+ 跨体态数据课程 + MoT 联合世界-动作建模
> - **结果**: WorldModelBench/DreamGen/LIBERO/RoboTwin 2.0 多项 SOTA，推理时间线性扩展
> - **代码**: https://github.com/kairos-agi/kairos-sensenova

---

*笔记创建时间: 2026-06-22*
