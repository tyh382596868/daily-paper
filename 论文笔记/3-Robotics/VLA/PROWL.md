---
title: "PROWL: Prioritized Regret-Driven Optimization for World Model Learning"
method_name: "PROWL"
authors: [Ahmet H. Güzel, Jenny Seidenschwarz, Benjamin Graham, Jonathan Sadeghi, Jeffrey Hawke, Jack Parker-Holder, Ilija Bogunovic]
year: 2026
venue: arXiv
tags: [world-model, adversarial-curriculum, video-diffusion, reinforcement-learning, curriculum-learning, minerl]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.18803v1
created: 2026-05-20
---

# 论文笔记：PROWL: Prioritized Regret-Driven Optimization for World Model Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University College London AI Centre; Odyssey |
| 日期 | May 2026 |
| 项目主页 | https://odyssey.ml/introducing-prowl |
| 对比基线 | [[VPT]]-frozen 行为参考策略 / Phase 1 被动预训练世界模型 |
| 链接 | [arXiv](https://arxiv.org/abs/2605.18803) / [Code](无公开仓库) |

---

## 一句话总结

> 用 KL 约束的对抗策略主动暴露[[视频扩散模型|视频世界模型]]的高误差轨迹，并通过优先级回放缓冲把这些失败转成由易到难的课程，从而修复被动数据采样不到的交互关键转移。

---

## 核心贡献

1. **约束式对抗课程 (Constrained Adversarial Curriculum)**: 把协同进化的 min–max 优化与行为正则结合，让对抗策略只在行为分布附近寻找失败案例，避免世界模型被推向[[分布外泛化|分布外]]区域而产生「[[奖励攻击|reward hacking]]」。
2. **优先级对抗轨迹缓冲 (PAT Buffer)**: 用预测误差、动作保真度、学习进度三项指标对发现的轨迹重排序，世界模型解决简单失败后自动降权、把更难的未解决轨迹拉进训练。
3. **协同进化发现交互关键动作组合**: 对抗策略发现了 27 个被动数据集（BASALT、[[VPT]]-frozen）中完全缺失的新型复合动作模式。
4. **行为锚定的实证分析**: 系统性地表明有效的对抗世界模型训练「必须」依赖显式行为正则来平衡探索性失败发现，否则会退化为相机抖动式的奖励攻击。

---

## 问题背景

### 要解决的问题

[[视频扩散模型|动作条件视频世界模型]]通常在被动收集的人类演示数据上训练。但被动演示数据系统性地**欠采样了罕见、交互关键 (interaction-critical) 的转移**——这些转移恰恰是世界模型最容易预测错误、却也最重要的部分（如复杂动作组合下的几何/运动/视觉一致性）。

### 现有方法的局限

- **被动预训练 ([[passive 世界模型|passive world model]])**: 只能见到演示分布内的转移，对动作密集、组合复杂的场景泛化差。
- **主动数据采集（[[Plan2Explore]]、Ready Policy One 等）**: 多用于状态空间[[基于模型的强化学习|model-based RL]]，未针对高保真视频世界模型的失败发现设计。
- **无约束对抗训练**: 直接 min–max 对抗会让策略发现「策略空间外」的退化解（如疯狂转相机），世界模型被迫拟合这些非真实转移，损害真实分布上的性能。

### 本文的动机

作者认为：世界模型自己的**预测失败本身就是最有信息量的训练信号**。关键是要有一个机制——既能主动「钓出」这些失败（对抗策略），又能约束这些失败不偏离真实行为分布（[[KL 散度|KL]] 正则），还能让训练始终聚焦于「尚未解决的最难失败」（优先级缓冲）。这三者构成 PROWL。

---

## 方法详解

### 模型架构

PROWL 采用 **两阶段（Phase 1 预训练 + Phase 2 对抗课程）协同进化** 架构：

- **输入**: 上下文观测帧序列 + 动作条件 $c$（[[VPT]] 风格的离散键鼠动作）
- **世界模型 Backbone**: [[Wan2.2|Wan2.1]] 1.3B 参数[[扩散变换器]]，[[VAE]] 与文本编码器冻结
- **核心模块**: [[对抗策略]] $\pi_\phi$ 用 [[PPO]] 训练去发现高误差轨迹；[[PAT Buffer]] 对发现的轨迹重排序
- **训练范式**: [[Diffusion Forcing]]（每帧独立加噪），按 $K=3$ 的[[Action Chunking|动作块]]推进
- **输出**: 给定动作条件下的未来潜变量帧序列 $z_{t:t+H}$
- **总参数**: 世界模型 1.3B（Phase 2 只训练 cross-attention 与 adapter）

两阶段流程：

- **Phase 1（预训练）**: 在 BASALT *FindCave* 人类演示上用 [[Diffusion Forcing]] 训练世界模型，得到被动基线 $\theta_0$。
- **Phase 2（对抗课程）**: $\pi_\phi$ 与世界模型 $\theta$ 协同进化——策略找失败、世界模型修失败、[[PAT Buffer]] 维护难度递增的课程。

### 核心模块

#### 模块1: 约束式对抗策略 $\pi_\phi$

**设计动机**: 利用 [[强化学习]] 让一个智能体主动在环境里探索，专门去触发世界模型的预测崩溃；但用[[KL 散度]]约束把它锚定在冻结的 [[VPT]] 行为参考 $\pi_{ref}$ 附近，防止它发现不真实的退化解。

**具体实现**:

- 策略以世界模型的预测误差为奖励信号（见公式 7 的 `score`），通过 [[PPO]] 最大化期望分数。
- 对每个状态施加前向 [[KL 散度]] $\mathrm{KL}(\pi_\phi \| \pi_{ref})$ 惩罚，权重 $c_{kl} \in \{0.5, 1.0, 1.5\}$。
- $c_{kl}=0.5$（无锚定）时策略退化为「相机抖动 (camera-thrashing)」式[[奖励攻击]]；$c_{kl}\ge1.0$ 时探索稳定。

#### 模块2: 优先级对抗轨迹缓冲 (PAT Buffer)

**设计动机**: 把对抗策略发现的失败组织成一个**由易到难、动态演化的课程**——世界模型解决了某些失败后，这些轨迹的信息量下降，应当降权；未解决的难轨迹应当上浮。

**具体实现**:

- 容量 256 条轨迹，每条轨迹按[[复合轨迹分数|复合分数]] `score`（公式 7）排序。
- 采样分布混合「分数优先」与「陈旧度优先」两项（公式 2），用 $\rho$ 平衡。
- 训练时按混合比 PAT:passive = 0.5:0.5 与被动数据交替喂给世界模型。
- 学习进度项 $\Delta\ell_{regret}$ 跟踪世界模型对每条轨迹误差的下降，自动实现「解决即降权」。

#### 模块3: 世界模型微调目标

**设计动机**: 世界模型既要修复对抗发现的失败，又不能遗忘被动分布。

**具体实现**:

- 用混合系数 $r$ 加权「被动数据的 [[Flow Matching|Flow-Matching]] 损失」与「PAT 缓冲数据的 Flow-Matching 损失」（公式 4 上半）。
- Phase 2 中冻结时空 Backbone，仅训练 [[交叉注意力|cross-attention]] 与 adapter，降低灾难性遗忘风险。

---

## 关键公式

### 公式1: [[Diffusion Forcing|扩散强迫损失]]

$$
\mathcal{L}_{DF} = \sum_i \left\| v_\theta(z_i^{\sigma_i}, \sigma_i, c) - (\varepsilon_i - z_i) \right\|_2^2
$$

**含义**: Phase 1 世界模型的预训练目标。每帧 $i$ 独立采样噪声等级 $\sigma_i$，网络 $v_\theta$ 预测速度场（[[Flow Matching]] 形式），使模型可在不同帧加不同噪声、支持自回归式滚动预测。

**符号说明**:
- $z_i^{\sigma_i}$: 第 $i$ 帧在噪声等级 $\sigma_i$ 下的含噪潜变量
- $\sigma_i$: 第 $i$ 帧独立采样的噪声等级（[[Diffusion Forcing]] 的核心——逐帧异步加噪）
- $c$: 动作条件
- $v_\theta$: 待训练的速度预测网络
- $\varepsilon_i - z_i$: Flow-Matching 的速度目标（噪声减干净潜变量）

### 公式2: [[PAT Buffer|优先级回放分布]]

$$
P(i) = (1-\rho)\, P_{score}(i) + \rho\, P_{stale}(i)
$$

**含义**: PAT 缓冲中第 $i$ 条轨迹被采样的概率，混合「分数优先」与「陈旧度优先」，保证既聚焦高误差轨迹、又不让旧轨迹饿死。

**符号说明**:
- $P_{score}(i)$: 正比于轨迹复合分数的采样概率（聚焦难例）
- $P_{stale}(i)$: 正比于轨迹未被采样时长的概率（防止陈旧）
- $\rho \in [0,1]$: 两者的混合权重

### 公式3: [[Prediction Regret|潜在遗憾 (Latent Regret)]]

$$
\ell_{regret}(\tau) = \sqrt{\frac{1}{H \cdot C \cdot N_{lat}} \sum_{t=S}^{S+H-1} \left\| z_t^{pred} - z_t^{real} \right\|_2^2}
$$

**含义**: 衡量世界模型在轨迹 $\tau$ 上的预测误差——预测潜变量与真实潜变量的归一化 RMSE，是对抗策略奖励与 PAT 排序的核心信号。

**符号说明**:
- $z_t^{pred}, z_t^{real}$: 第 $t$ 帧的预测/真实潜变量
- $H$: 评估时域长度（帧数）
- $C$: 潜变量通道数
- $N_{lat}$: 每帧潜变量空间元素数
- $S$: 评估起始帧

### 公式4: [[Action-Follow Score|动作跟随分数 (AFS)]]

$$
\ell_{AFS}(\tau) = \frac{1}{N \cdot H_{raw}} \sum_{t=0}^{N-1} \sum_{i=1}^{H_{raw}} \left\| F_{t,i}^{pred} - F_{t,i}^{real} \right\|_2
$$

**含义**: 用[[光流]]度量「动作是否被正确执行」。比较预测视频与真实视频的光流向量，捕捉外观指标（[[感知图像相似度|LPIPS]]、[[结构相似性|SSIM]]）看不见的运动保真度失败。光流由 SEA-RAFT 计算。

**符号说明**:
- $F_{t,i}^{pred}, F_{t,i}^{real}$: 第 $t$ 帧第 $i$ 个像素位置的预测/真实光流向量
- $N$: 帧对数量
- $H_{raw}$: 原始像素帧的空间元素数

### 公式5: [[复合轨迹分数]]

$$
\mathrm{score}(\tau) = z_{\mathcal{B}}(\ell_{regret}) + \lambda_{AFS}\cdot z_{\mathcal{B}}(\ell_{AFS}) + \Delta\ell_{regret}
$$

**含义**: PAT 缓冲对每条轨迹的综合评分，同时考虑潜在误差、动作保真误差与学习进度，决定其在课程中的优先级。

**符号说明**:
- $z_{\mathcal{B}}(\cdot)$: 在 PAT 缓冲 $\mathcal{B}$ 内的 z-score 归一化
- $\lambda_{AFS}$: AFS 项权重（$\in \{0.10, 0.25\}$）
- $\Delta\ell_{regret}$: 学习进度项，跟踪潜在遗憾随训练的下降量（解决即降权）

### 公式6: 非对称 min–max 协同进化目标

$$
\begin{aligned}
\theta^* &= \arg\min_\theta\; (1-r)\,\mathbb{E}_{\tau\sim p_{passive}}\!\left[\mathcal{L}_{FM}(\tau;\theta)\right] + r\cdot\mathbb{E}_{\tau\sim p_{\mathcal{B}}}\!\left[\mathcal{L}_{FM}(\tau;\theta)\right] \\
\phi^* &= \arg\max_\phi\; \mathbb{E}_{\tau\sim\pi_\phi}\!\left[\mathrm{score}(\tau;\theta)\right] - c_{kl}\cdot\mathbb{E}_{s\sim d^{\pi_\phi}}\!\left[\mathrm{KL}\!\left(\pi_\phi(\cdot|s)\,\|\,\pi_{ref}(\cdot|s)\right)\right]
\end{aligned}
$$

**含义**: PROWL 的核心对抗博弈。世界模型 $\theta$ 最小化被动数据与 PAT 数据的混合 [[Flow Matching|Flow-Matching]] 损失；策略 $\phi$ 最大化世界模型预测分数，但被前向 [[KL 散度]] 锚定在冻结行为参考附近。两者「非对称」——世界模型见混合数据、策略受 KL 约束。

**符号说明**:
- $r$: PAT 数据与被动数据的混合系数
- $p_{passive}, p_{\mathcal{B}}$: 被动数据分布 / PAT 缓冲分布
- $\mathcal{L}_{FM}$: Flow-Matching 损失
- $d^{\pi_\phi}$: 策略 $\pi_\phi$ 诱导的状态访问分布
- $\pi_{ref}$: 冻结的 [[VPT]] 行为参考策略
- $c_{kl}$: KL 约束强度

### 公式7: [[PPO|KL 正则化 PPO]] 损失

$$
\mathcal{L}_{PPO} = \mathcal{L}_{clip} + c_v\,\mathcal{L}_{value} - c_e\,\mathcal{H}(\pi_\phi) + c_{kl}\cdot\mathbb{E}_{s\sim d^{\pi_\phi}}\!\left[\mathrm{KL}\!\left(\pi_\phi(\cdot|s)\,\|\,\pi_{ref}(\cdot|s)\right)\right]
$$

**含义**: 对抗策略的实际优化目标。在标准 [[PPO]] 的剪裁项、价值项、熵项之外，加入对 [[VPT]] 行为参考的 KL 惩罚，把探索锚定在真实行为流形附近。

**符号说明**:
- $\mathcal{L}_{clip}$: PPO 剪裁的策略梯度项
- $\mathcal{L}_{value}$: 价值函数回归损失，权重 $c_v$
- $\mathcal{H}(\pi_\phi)$: 策略熵，鼓励探索，权重 $c_e$
- $c_{kl}$: KL 行为锚定权重

---

## 关键图表

### Figure 1: PROWL 框架总览

> 🖼️ **Figure 1** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: PROWL 两阶段流程。Phase 1 在被动人类演示数据上预训练世界模型；Phase 2 运行对抗课程——[[对抗策略]]、世界模型、[[PAT Buffer]] 三者之间的数据流闭环：策略发现高误差轨迹 → 写入 PAT 缓冲 → 缓冲按分数重排序 → 世界模型在混合数据上微调 → 误差信号反哺策略奖励。

### Figure 2: 对抗策略动力学

> 🖼️ **Figure 2** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: (a) 回合回报（仅作合理性检查，按构造非平稳）；(b) 到冻结 [[VPT]] 参考的 [[KL 散度]]——弱约束探索与稳定对抗探索明显分离；(c) 相机速度作为[[奖励攻击]]的「触发线」——$c_{kl}=0.5$ 时相机速度飙升；(d) 相机抖动 (camera-thrashing) 的退化案例可视化。结论：**弱 KL 约束导致奖励攻击，强约束在维持 KL 控制的同时抑制它**。

### Figure 3: 课程形成与新型交互发现

> 🖼️ **Figure 3** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: (a) [[PAT Buffer]] 平均潜在遗憾随时间变化——成功的配置「先升后降」：先发现难例使遗憾上升，再随世界模型学习而下降；(b) 严格新型复合动作模式的发现计数——PROWL 发现了 27 个 BASALT 与 [[VPT]]-frozen 缓冲中完全没有的复合动作模式。

### Figure 4: 留出与新型场景下的定性对比

> 🖼️ **Figure 4a** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

> 🖼️ **Figure 4b** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: 留出与新型动作序列上的滚动预测对比。上图为来自 [[VPT]] 参考策略的高误差留出轨迹（旋转+前进+平移组合）；下图为对抗策略发现的新型复合动作（旋转+前进+跳跃+冲刺+平移）。PROWL 在这些动作密集组合下生成更连贯的滚动预测。

### Figure 5: AFS-EPE 捕捉外观指标看不见的运动失败

> 🖼️ **Figure 5** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: 与 Figure 6 同一条轨迹。黄色带标注动作密集帧。[[结构相似性|SSIM]] / [[感知图像相似度|LPIPS]] / 潜在遗憾等外观指标在动作密集帧上几乎不变，但 [[Action-Follow Score|AFS-EPE]]（光流端点误差）在这些帧出现明显尖峰——说明运动保真度失败对标准外观指标「不可见」，必须用 AFS 才能捕捉。

### Figure 6: VPT Base 与 PROWL 在相同动作下的对比

> 🖼️ **Figure 6** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18803)）

**说明**: 留出 VPT 参考轨迹第 22–32 帧。相同动作条件下，VPT Base 在动作密集帧出现[[光流]]不连贯（运动方向混乱），PROWL 生成的光流场连贯一致，验证 PROWL 显著改善了动作条件下的运动建模。

### Table 1: 留出 BASALT 任务的零样本泛化（lam010 配置）

| 指标 | vs Phase 1 | vs VPT-frozen 基线 |
|------|-----------|-------------------|
| 潜在遗憾 (整体) | −3.5% | −2.5% |
| AFS-EPE (整体) | −12.6% | −3.9% |
| 潜在遗憾 (最难 Top 10%) | — | −8.7% |
| AFS-EPE (最难 Top 10%) | — | −20.9% |

**说明**: 三个留出任务（*MakeWaterfall*、*BuildVillageHouse*、*CreateVillageAnimalPen*，共 300 段、每任务 100 段）。PROWL 在整体上小幅改进，但**在最难子集上差距显著拉大**——说明对抗课程的收益集中在硬例上。

### Table 2: 聚焦专家配置（kl150：$c_{kl}=1.5$, $\lambda_{AFS}=0.25$）

| 评估设定 | 潜在遗憾 | AFS-EPE | 其他 |
|----------|---------|---------|------|
| 跨缓冲对抗迁移 (384 条) | −5.8% | −8.9% | — |
| 分布内稳定性 (FindCave, n=64) | −3.5% | −7.4% | — |
| 长时域累积误差 (第 200 帧, 18 chunks) | — | — | LPIPS −6.7%, SSIM +4.3% |

**说明**: kl150 配置在硬分布内案例与长时域稳定性上表现最佳。长时域评估超出训练时域，仍能降低累积误差。

### Table 3: 课程动力学与新型模式（lam010 配置）

| 指标 | vs Phase 1 | vs 匹配计算量基线 |
|------|-----------|------------------|
| 27 个新型模式的平均潜在遗憾 | −5.5% | −4.5% |
| 27 个新型模式的平均 AFS-EPE | −11.5% | −9.1% |

**说明**: 在对抗策略发现的 27 个「严格新型」复合动作模式上，PROWL 相对 Phase 1 与「匹配计算量基线」（同等算力但不用对抗课程）都有改进——说明收益来自课程本身而非额外算力。

### Table 4: 关键超参数

| 项目 | 配置 |
|------|------|
| 世界模型 | Wan2.1 1.3B；Phase 2 仅训 cross-attention + adapter |
| VAE 压缩比 | 时间/空间 (4, 8, 8) |
| PAT 缓冲容量 | 256 条轨迹 |
| 混合比 (PAT:passive) | 0.5 : 0.5 |
| 动作块大小 $K$ | 3（20 fps 下 12 像素帧 = 0.6 s） |
| 上下文窗口 | 21 个潜变量帧 |
| PPO 学习率 | $3\times10^{-5}$ |
| KL 权重 $c_{kl}$ | $\{0.5, 1.0, 1.5\}$ |

**关键发现**: Phase 2 冻结主干、只训交叉注意力与 adapter，是抑制灾难性遗忘与控制计算量的关键设计。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| BASALT *FindCave* 演示 | 人类演示 | 被动收集 | Phase 1 训练 / 分布内评估（64 段） |
| BASALT 留出任务 | 300 段（3 任务 ×100） | *MakeWaterfall* / *BuildVillageHouse* / *CreateVillageAnimalPen* | 零样本泛化测试 |
| 对抗发现轨迹 (PAT) | 缓冲 256 条 | Phase 2 在线生成 | Phase 2 微调 |
| 跨缓冲对抗集 | 384 条 | 来自不同配置的对抗轨迹 | 对抗迁移评估 |
| 长时域片段 | 30 段 | 评估超出训练时域 | 累积误差评估 |

### 实现细节

- **Backbone**: [[Wan2.2|Wan2.1]] 1.3B 参数[[扩散变换器]]，[[VAE]] 与文本编码器冻结
- **训练范式**: [[Diffusion Forcing]]（逐帧异步加噪）
- **对抗策略优化器**: [[PPO]]，学习率 $3\times10^{-5}$
- **环境**: MineRL / BASALT（Minecraft）
- **行为参考**: 冻结的 [[VPT]] 策略
- **光流计算**: SEA-RAFT
- **配置谱系**: 沿 $(c_{kl}, \lambda_{AFS})$ 两轴探索三种区制

### 配置谱系（三种区制）

1. **无锚定 (Unanchored)**: $c_{kl}=0.5$ → 通过相机抖动进行[[奖励攻击]]，作为反面对照。
2. **广探索 (Broad-exploration, lam010)**: $c_{kl}=1.0, \lambda_{AFS}=0.10$ → 动作空间覆盖广，擅长留出任务迁移。
3. **聚焦专家 (Focused-specialist, kl150)**: $c_{kl}=1.5, \lambda_{AFS}=0.25$ → 擅长硬分布内案例与长时域稳定性。

### 可视化结果

定性上（Figure 4/6），PROWL 在动作密集的复合动作组合下生成的滚动预测明显更连贯，[[光流]]场一致性显著优于 VPT Base；标准外观指标却看不出差异，凸显 AFS 作为评估指标的必要性。

---

## 批判性思考

### 优点

1. **问题切中要害**: 准确指出被动演示数据「欠采样交互关键转移」是视频世界模型泛化的根本瓶颈，并给出一个可操作的闭环解法。
2. **行为锚定的实证洞见有普适价值**: Figure 2 清晰展示了无约束对抗 → 奖励攻击的失败模式，KL 锚定的必要性可迁移到其他对抗式自改进系统。
3. **AFS 指标设计巧妙**: 用光流端点误差捕捉「动作没被正确执行」这一外观指标盲区，对动作条件视频生成的评估有方法论贡献。
4. **课程自演化机制干净**: PAT 缓冲用学习进度项 $\Delta\ell_{regret}$ 实现「解决即降权」，无需手工调度难度。

### 局限性

1. **单种子运行**: 每个配置只跑单种子（作者自述为算力权衡），改进幅度多在 2–13% 区间，缺乏方差估计，结论稳健性存疑。
2. **环境单一**: 只在 MineRL/BASALT 上验证，跨环境（如真实机器人、自动驾驶）泛化性完全未测。
3. **长时域评估在训练时域外**: 长时域累积误差评估超出训练时域，属外推，解释力有限。
4. **冻结主干可能限制适配**: Phase 2 冻结时空 Backbone 虽抑制遗忘，但也可能限制世界模型对真正新型动力学的适配能力。
5. **改进幅度偏小**: 整体指标提升多为个位数百分比，仅在「最难 Top 10%」子集上才有 20% 量级收益，实用价值取决于下游对硬例的敏感度。

### 潜在改进方向

1. 多种子 + 置信区间，确认改进显著性。
2. 把对抗课程迁移到真实机器人视频世界模型（如 [[Dreamer 4]] / [[Cosmos-Policy]] 风格的设定）验证跨域有效性。
3. Phase 2 部分解冻主干并配合更强正则，探索适配能力与遗忘的平衡。
4. 把 PAT 课程与下游策略学习耦合——直接评估「更好的世界模型」是否带来「更好的策略」。

### 可复现性评估

- [ ] 代码开源（未提供公开仓库）
- [x] 预训练模型基于公开的 Wan2.1 1.3B
- [x] 训练细节较完整（Table 4 给出关键超参）
- [x] 数据集可获取（BASALT / MineRL 公开）

---

## 关联笔记

### 基于

- [[Diffusion Forcing]]: Phase 1 世界模型的预训练范式（逐帧异步加噪）
- [[Wan2.2|Wan2.1]]: 世界模型的 1.3B 扩散变换器主干
- [[VPT]]: 提供冻结的行为参考策略 $\pi_{ref}$ 与对照基线
- [[PPO]]: 对抗策略的强化学习优化器

### 对比

- [[passive 世界模型]]: PROWL 要解决的「被动训练欠采样交互关键转移」的对象
- [[PLR]]: 同属优先级课程/UED 思路，PROWL 把它从环境层面迁移到世界模型轨迹层面

### 方法相关

- [[对抗策略]]: 核心——主动发现世界模型失败
- [[PAT Buffer]]: 核心——把失败组织成由易到难的课程
- [[Prediction Regret]]: 世界模型预测误差的度量
- [[Action-Follow Score]]: 基于光流的运动保真度度量
- [[复合轨迹分数]]: PAT 缓冲的综合排序信号
- [[奖励攻击]]: 无 KL 锚定时对抗训练的失败模式
- [[KL 散度]]: 行为锚定的约束机制
- [[课程学习]]: PROWL 的整体范式
- [[光流]]: AFS 指标的计算基础

### 硬件/数据相关

- MineRL / BASALT: 实验环境与数据来源

---

## 速查卡片

> [!summary] PROWL: Prioritized Regret-Driven Optimization for World Model Learning
> - **核心**: 用 KL 约束的对抗策略主动暴露视频世界模型的高误差轨迹，转成由易到难的课程
> - **方法**: 两阶段——Phase 1 被动预训练 + Phase 2 对抗策略 / PAT 优先级缓冲 / 世界模型协同进化
> - **结果**: MineRL 上对留出任务、长时域稳定性、新型动作模式均有改进；最难 Top 10% 子集 AFS 提升约 21%
> - **关键洞见**: 有效的对抗世界模型训练「必须」靠 KL 行为锚定，否则退化为相机抖动式奖励攻击
> - **代码**: 无公开仓库（项目页 odyssey.ml/introducing-prowl）

---

*笔记创建时间: 2026-05-20*
