---
title: "Rethinking VLM Representation for VLA Initialization"
method_name: "RethinkVLAInit"
authors: [Weifeng Lin, Siyuan Huang, Hao Li, Tingwei Chen, Ruichuan An, Xinyu Wei, Jianbo Liu, Hongsheng Li]
year: 2026
venue: arXiv
tags: [vla, vlm-initialization, embodied-vqa, lora, robot-pretraining, representation-analysis]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.25802v1
created: 2026-05-27
---

# 论文笔记：Rethinking VLM Representation for VLA Initialization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CUHK / PolyU / Peking University / ACE Robotics |
| 日期 | May 2026 |
| 项目主页 | https://github.com/AFeng-x/Rethink_VLA_Initialization |
| 对比基线 | [[OpenVLA-OFT]] / [[π₀]] / [[Qwen3-VL]] / [[PaliGemma]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.25802) / [HTML](https://arxiv.org/html/2605.25802v1) / [Code](https://github.com/AFeng-x/Rethink_VLA_Initialization) |

---

## 一句话总结

> 系统拆解 [[VLM]] → [[VLA]] 初始化的三个轴（[[Embodied VQA]] 监督 / 参数更新策略 / 机器人数据预训练），结论是**保住预训练表征**最关键 —— [[LoRA]] 优于 [[Full Finetune|全量微调]]，"Grounding+Egocentric Understanding" 是最佳 VQA 组合，分阶段 LoRA 推到 RoboCasa 55.2%。

---

## 核心贡献

1. **首个系统化 VLA 初始化研究**: 横跨 3 个 [[基座 VLM]]（[[Qwen3-VL]]-4B/2B、[[PaliGemma]]2-3B）、2 种 [[Action Head]]（[[OpenVLA-OFT|MLP head]] 与 [[π₀]] 风格的 [[Diffusion Expert]]）、3 个 benchmark（[[LIBERO|Libero-10]]、[[SimplerEnv|SimplerBridge]]、[[RoboCasa]] GR1），把"何种 VLM 表征对 VLA 有用"的问题量化。
2. **七个 [[Embodied VQA]] 域的因果分析**: 把具身 VQA 拆成 [[Spatial Understanding|Spatial]]、[[Grounding]]、[[Plan & Reasoning]]、[[Camera Prediction]]、[[Egocentric Understanding]]、[[Temporal Understanding]]、[[Action-NTP]] 七类，用 [[单域消融|single-domain ablation]] 与 [[多域组合|domain composition]] 定位收益来源。
3. **参数更新策略的明确结论**: [[LoRA]]（rank 16 / α 32）系统性优于 [[Full Finetune|全量微调]]，并通过 [[LoRA Merge Strength|λ-merge 强度扫描]] 证明 **"过度重塑预训练表征会削弱 VLA 初始化"** 这一核心观点。
4. **机器人数据预训练的正确打开方式**: 用 [[AgiBot-World-Beta]] 做 [[Action-NTP]] 预训练前先做 [[Grounding]]+[[Egocentric Understanding]] VQA 适配，再用 LoRA（rank 64 / α 128）做机器人数据预训练，[[RoboCasa]] GR1 上从 49.5% → 55.2% (+5.7%)。

---

## 问题背景

### 要解决的问题

[[VLA]] 在工程上几乎都是"拿一个 [[VLM]] 当初始化、续训机器人动作"，但社区从未系统回答过：

- **哪些 [[VLM]] 能力**（空间、定位、规划……）真的对动作学习有用？
- **更新多少参数**（[[LoRA]] vs. [[Full Finetune]]）才不毁掉预训练表征？
- **机器人轨迹数据**该在 [[VLM]] 阶段、[[Action Head]] 阶段还是中间阶段注入？

### 现有方法的局限

- 主流 [[VLA]]（[[OpenVLA]]、[[π₀]]、[[GR00T N1.7]]）默认对 VLM 做 [[Full Finetune|全量微调]]，但是否破坏了预训练 [[VLM]] 表征从未量化。
- 现有 [[Embodied VQA]] benchmark（[[EmbodiedScan]]、[[RoboVQA]]）和 [[VLM-as-Judge|VLM 评测]] 关注 VQA 准确率本身，不直接对接 [[VLA]] 下游性能。
- "[[Knowledge Insulation]]" 类工作只论证"动作监督会损伤 VLM"，但没系统分析哪些 [[VLM]] 能力 *本来就有用*。

### 本文的动机

作者的中心观察是：

> *"原始预训练 VLM 表征本身就是 VLA 的最大性能来源 —— 从头训练的策略在所有 benchmark 上掉 20%+。"*

因此问题不是"怎么改造 VLM"，而是 **"怎么在尽量保住预训练表征的前提下注入动作相关信号"**。

---

## 方法详解

### 模型架构

<!-- 使用 [[概念]] 内联链接所有技术术语 -->

本文不提出新的 [[VLA]] 模型，而是设计一套**两阶段对照实验框架**：

- **输入**: 当前帧 $o_t$ + 语言指令 $\ell$
- **Stage 1 — VLM 适配**: 在 [[Embodied VQA]] 数据上用 [[LoRA]] 或 [[Full Finetune]] 适配 [[基座 VLM]]
- **Stage 2 — VLA 训练**: 把适配后的 VLM 接 [[Action Head]]，在机器人轨迹上训练
- **Action Head A**: [[OpenVLA-OFT]] 风格的**轻量 MLP**（对 VLM 表征**敏感**，主用）
- **Action Head B**: [[π₀]] 风格的 [[Diffusion Expert]]（高容量解码器，验证趋势是否仍然成立）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$

### 七个 [[Embodied VQA]] 域

| 域 | 含义 | 代表数据集 |
|----|------|----------|
| [[Spatial Understanding|Spatial]] | 物体间相对/绝对距离、朝向 | [[SpaceLLaVA]] / [[SpatialThinker]] / [[SpatialRGPT]] / [[SenseNova-SI]] |
| [[Grounding]] | 语言到像素/动作区域的定位 | [[PixMo Points]] / [[RoboPoint]] / [[RoboRefer]] / [[RoboAfford]] |
| [[Plan & Reasoning]] | 目标 → 子目标的任务分解 | [[VLM-3R]] / [[EO-Data]] |
| [[Camera Prediction]] | 相机内外参 / 视角变化估计 | (论文未单列大数据集) |
| [[Egocentric Understanding]] | 第一人称下的手部状态、握持物 | [[Robo2VLM]] / [[EgoThinker]] / [[EgoTaskQA]] / [[ShareRobot]] |
| [[Temporal Understanding]] | 视频事件顺序与因果关系 | (论文未单列大数据集) |
| [[Action-NTP]] | 把动作轨迹当 token 做自回归预测 | [[OXE\|OpenX-Embodiment]] / [[AgiBot-World-Beta]] |

### 两阶段训练管线

#### Stage 1: [[Embodied VQA]] 适配
- **数据预算**: 800K 样本 / 1 个 epoch / batch 128 / AdamW
- **更新策略对照**: [[LoRA]] (r=16, α=32, 保留预训练表征) vs. [[Full Finetune]]
- **单域**: 一次只用一个域的数据
- **多域组合**: 固定总样本量，按比例混合 2 域 / 3 域 / 全 7 域

#### Stage 2: [[VLA]] 动作训练
- **训练步数**: 50K–80K 步（按 benchmark 不同）/ batch 128 / bf16
- **下游 recipe 完全统一**：所有初始化方案用相同动作训练超参数，把差异归因到 Stage 1

#### 可选: 机器人数据预训练
- 在 Stage 1 与 Stage 2 之间插入 [[AgiBot-World-Beta]] 上的 [[Action-NTP]] LoRA（r=64, α=128, 100K 步, 16×A800）。

---

## 关键公式

### 公式 1: [[LoRA Merge Strength|LoRA 合并强度]]

$$
W_\lambda = W_0 + \lambda \cdot \Delta W_{\text{LoRA}}, \quad \Delta W_{\text{LoRA}} = \tfrac{\alpha}{r} \, B A
$$

**含义**: 把 [[LoRA]] 更新量按可调系数 $\lambda$ 合并回原权重，用来定量评估"偏离预训练表征多远"对下游 [[VLA]] 性能的影响。

**符号说明**:
- $W_0 \in \mathbb{R}^{d \times d}$: 预训练 [[VLM]] 的原始权重
- $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$: 学得的低秩矩阵
- $r$: [[LoRA]] 秩，Stage 1 默认 16
- $\alpha$: 缩放因子，Stage 1 默认 32
- $\lambda$: 合并强度，$\lambda=1$ 对应标准 LoRA 合并；论文扫描 $\lambda \in \{0, 0.5, 1.0, 1.5, 2.0\}$

**实验观察 (Appendix B)**: 成功率随 $\lambda$ 呈 **非单调** 曲线，峰值出现在 $\lambda = 1.0$（95.6%），$\lambda = 1.5$ 降到 95.0%，$\lambda = 2.0$ 进一步降到 92.7% —— 直接为"过度重塑预训练表征会削弱迁移"提供因果证据。

---

## 关键图表

### Figure 1: 研究设计总览

![Figure 1](https://arxiv.org/html/2605.25802v1/x1.png)

**说明**: 论文的研究设计骨架。横轴是三个研究维度（[[Embodied VQA]] 监督 / 参数更新策略 / 机器人数据预训练），纵轴是评测面（多种基座 [[VLM]] × 多种 [[Action Head]] × 多种仿真 benchmark）—— 整篇论文都是这张表格里某些格子的"系统填空"。

### Figure 2: 七个 [[Embodied VQA]] 域与两种 [[VLA]] 架构

![Figure 2](https://arxiv.org/html/2605.25802v1/x2.png)

**说明**: 上半部分给出 [[Spatial Understanding|Spatial]] / [[Grounding]] / [[Plan & Reasoning]] / [[Camera Prediction]] / [[Egocentric Understanding]] / [[Temporal Understanding]] / [[Action-NTP]] 七个具身 VQA 域的典型样例（例如 Grounding 是"点出红色杯子的位置"）；下半部分对比两条评测管线 —— [[OpenVLA-OFT]] 风格的 MLP head 与 [[π₀]] 风格的 [[Diffusion Expert]]。

### Figure 3: [[LoRA]] vs. [[Full Finetune]] 的对比

![Figure 3](https://arxiv.org/html/2605.25802v1/x3.png)

**说明**: 在所有 (VLM, VQA 域, benchmark) 组合下，[[LoRA]] 的成功率分布**系统性高于** [[Full Finetune]]。作者的解读是：全量微调会把预训练知识"洗掉"，而 LoRA 的低秩更新更接近"加法注入"，因此对动作学习的下游分布更友好。

### Figure 4: 基座 [[VLM]] 影响

![Figure 4](https://arxiv.org/html/2605.25802v1/x4.png)

**说明**: (a) 相对基线的平均成功率变化（不同 [[VQA]] 域 → 对每个基座 VLM 的提升/退步）；(b) 详细成功率数字。同一域对不同基座的影响方向**不一致** —— 比如 Spatial 在 [[Qwen3-VL]]-4B 上为负，但在 [[PaliGemma]]2-3B 上为正，说明 VQA 收益**强依赖**于基座本身的弱点。

### Table 1: 单域 VQA 适配成功率（%）

| 域 \ Benchmark | Libero-10 (MLP) | SimplerBridge (MLP) | RoboCasa (MLP) | Libero-10 (Diff) | RoboCasa (Diff) |
|--------------|-----|-----|-----|-----|-----|
| Baseline (无 VQA 适配) | 92.4 | ~45.8 | ~49.5 | ~91.0 | ~51.7 |
| Spatial | 90.x | – | 47.x | – | 50.x |
| **Grounding** | **+3.2 (95.6)** | – | – | – | 53.x |
| Plan & Reasoning | 中性 | – | – | – | – |
| Camera Prediction | 偏负 | – | – | – | – |
| Egocentric Understanding | 中性偏正 | – | – | – | – |
| Temporal Understanding | 中性 | – | – | – | – |
| Action-NTP | 偏负 (单独) | 显著负 | 偏负 | – | – |

**关键发现**:
- 单域 [[Grounding]] 在 [[LIBERO|Libero-10]] (MLP) 上拿到全表最大正增益 +3.2%
- [[SimplerEnv|SimplerBridge]] 上几乎所有单域适配都是**负迁移**，说明该 benchmark 的瓶颈不在 VQA 能力
- 从零训练（无 [[VLM]] 初始化）在所有 benchmark 上掉 20%+，这是论文最响亮的数字

### Table 2: 多域组合（domain composition）

| 组合 | Libero-10 (MLP) | RoboCasa (Diff) |
|------|-----|-----|
| Grounding only | 95.6 | 53.0 |
| Egocentric only | 94.x | 52.x |
| **Grounding + Egocentric** | **95.7** | **53.5** |
| 3 域混合 | 94.9 | 51.x |
| 全 7 域 | 94.2 | 49.1 |

**关键发现**: 最佳组合是 **{[[Grounding]] + [[Egocentric Understanding]]}** —— "知道东西在哪儿 + 知道自己手在哪儿"恰好对应执行抓取所需的两类**互补**信号。但**继续加域反而退步**：作者把这解读为"额外域稀释了有用监督，甚至引入干扰"。

### Table 3: [[AgiBot-World-Beta|AgiBot]] 机器人数据预训练（RoboCasa GR1，Diffusion head）

| 配置 | 成功率 |
|------|-----|
| Base（仅 Stage 2 动作训练） | 49.5 |
| Robot-only LoRA（只做 Action-NTP 预训练） | 54.0 (+4.5) |
| **Sequential: G+E VQA → Robot LoRA** | **55.2 (+5.7)** |

**关键发现**: VQA 预训练与机器人数据预训练**互不替代**且**有序最优** —— 先种 VQA 表征，再灌机器人轨迹，效果优于任意单独路径。

### Table 4 / Table 5: 数据与训练配置（Appendix）

- **Table 4**: Stage 1 VQA 数据池按 7 域统计样本数（总量 ~800K）
- **Table 5**: Stage 2 各 benchmark 的训练步数（50K–80K）、动作维度、相机配置

### Tables 6–9: 附录诊断

- **Table 6**: [[LoRA Merge Strength|λ-merge]] 强度扫描结果（即上述非单调曲线）
- **Table 7**: [[VLM]] 通用能力保留（适配前后在通用 VQA benchmark 上的退化幅度）
- **Table 8**: 冻结 backbone 的 probing 实验
- **Table 9**: 不同 [[LoRA]] rank/α 的敏感性

---

## 实验

### 数据集与 benchmark

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Stage 1 VQA 池 | ~800K | 跨 7 个 [[Embodied VQA]] 域 | VLM 适配 |
| [[AgiBot-World-Beta]] | 大规模真实机器人轨迹 | 用作 [[Action-NTP]] 预训练 | 中间阶段 |
| [[LIBERO|Libero-10]] | 10 长程任务 | 单臂桌面，控制难度低 | VLA 评测 |
| [[SimplerEnv|SimplerBridge]] | WidowX 风格 4 种变体 | 较强视觉/控制偏移 | VLA 评测 |
| [[RoboCasa]] GR1 Tabletop | 24 任务（含双臂、articulated） | 类人形操作，最难 | VLA 评测 |

### 实现细节

- **Stage 1 (VQA)**: AdamW / lr scheduled / batch 128 / 1 epoch / 800K samples / [[LoRA]] r=16 α=32 vs. [[Full Finetune]]
- **Stage 2 (VLA)**: 50K–80K 步 / batch 128 / bf16
- **机器人数据预训练**: [[LoRA]] r=64 α=128 / 100K 步 / 16×A800 GPUs
- **基座**: [[Qwen3-VL]]-4B（主）、[[Qwen3-VL]]-2B、[[PaliGemma]]2-3B
- **动作头**: [[OpenVLA-OFT]] 的轻量 MLP（主）+ [[π₀]] 风格 [[Diffusion Expert]]

### 关键定量结论

1. **预训练 [[VLM]] 表征 = 20%+ 性能** — 从头训练比初始化掉 20%+
2. **[[LoRA]] > [[Full Finetune]]** — 在 (VLM, 域, benchmark) 组合上几乎全胜
3. **最佳 VQA 组合 = [[Grounding]] + [[Egocentric Understanding]]** — RoboCasa Diffusion 53.5
4. **最佳整体配方 = Sequential 阶梯式** — Base 49.5 → +Robot LoRA 54.0 → +前置 G+E VQA 55.2
5. **过度合并 LoRA ($\lambda > 1$) 反而退步** — 95.6 → 92.7（λ=2）

---

## 批判性思考

### 优点
1. **罕见的"系统性消融"工作**: 真的把 7 域 × 多 VLM × 多 Action Head × 多 benchmark 跑了一遍，社区终于有可以引用的对照表
2. **结论可操作**: "Grounding + Egocentric + LoRA + Sequential" 是一条非常具体的食谱，社区可以直接照搬
3. **λ-merge 强度实验**: 把"是否破坏预训练表征"从直觉量化成单调可扫描的标量

### 局限性
1. **机制解释偏弱**: 为什么是 [[Grounding]] + [[Egocentric Understanding]]、为什么不是别的对？论文只给"互补信号"的直觉，缺一个表征几何或 [[probing]] 层面的证据
2. **VLA 架构覆盖窄**: 主要是 [[OpenVLA-OFT]] 风格的轻量 MLP + [[π₀]] [[Diffusion Expert]]，[[流匹配|Flow-Matching]] 头、自回归动作 token（如 [[OpenVLA]]、[[GR00T N1.5]]）未在主表
3. **Benchmark 仍是仿真**: [[LIBERO]] / [[SimplerEnv|SimplerBridge]] / [[RoboCasa]] 都是仿真，真实 [[sim-to-real]] 表现可能完全不同
4. **未与 [[Knowledge Insulation]] 等"动作侧解决方案"对照**: 现有"如何防止 VLM 被动作监督污染"的工作（如 KI、IntentVLA 的 frozen backbone）跟"如何用 VQA 加强 VLM"是互补关系，但论文没做联合实验

### 潜在改进方向
1. 用 [[CKA]] / 表征相似度量化"预训练表征保留度"，给 $\lambda$ 曲线一个机制层面的解释
2. 把 G+E 组合迁移到 [[Flow Matching|流匹配]]/AR token 等更新的 [[Action Head]] 上验证可推广性
3. 接入真实机器人验证 Sequential 配方是否仍是最优

### 可复现性评估
- [x] 代码开源（GitHub 公开）
- [ ] 预训练模型（README 显示尚未释放权重）
- [x] 训练细节完整（LoRA rank/α、步数、batch size 都给了）
- [x] 数据集可获取（VQA 数据池全为公开数据集）

---

## 关联笔记

### 基于
- [[OpenVLA-OFT]]: 主用 MLP action head 来源
- [[π₀]]: Diffusion Expert action head 来源
- [[LoRA]]: 论文核心更新策略
- [[AgiBot-World-Beta]]: 中间阶段机器人数据预训练

### 对比
- [[OpenVLA]]: 默认 Full Finetune VLM，本文证明这不是最优
- [[Knowledge Insulation]]: 另一条思路 —— 不让动作监督污染 VLM
- [[VLM4VLA]] / [[VLASER]]: 同期讨论 VLM→VLA 转换的工作

### 方法相关
- [[Embodied VQA]]: 论文研究的供给侧
- [[Grounding]] / [[Egocentric Understanding]]: 实验找出的最佳 VQA 对
- [[LoRA Merge Strength]]: 量化"偏离预训练表征"的工具

### 硬件/数据相关
- [[LIBERO]] / [[SimplerEnv|SimplerBridge]] / [[RoboCasa]]: 三大评测仿真器
- [[Qwen3-VL]] / [[PaliGemma]]: 测试的基座 VLM

---

## 速查卡片

> [!summary] Rethinking VLM Representation for VLA Initialization
> - **核心**: 预训练 [[VLM]] 表征本身 = 20%+ 性能，"少改" 比 "多改" 重要
> - **方法**: 7 域 × 2 策略 × 3 benchmark 的系统消融
> - **最佳配方**: [[Grounding]] + [[Egocentric Understanding]] VQA → [[LoRA]] r=16 → [[AgiBot-World-Beta\|AgiBot]] [[Action-NTP]] LoRA r=64
> - **结果**: RoboCasa GR1 49.5 → 55.2 (+5.7)
> - **代码**: https://github.com/AFeng-x/Rethink_VLA_Initialization

---

*笔记创建时间: 2026-05-27*
