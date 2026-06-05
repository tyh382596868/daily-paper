---
title: "World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis"
method_name: "WLA"
authors: [Yi Yang, Zhihong Liu, Siqi Kou, Yiyang Chen, Yanzhe Hu, Jianbo Zhou, Boyuan Zhao, Zhijie Wei, Xiao Xia, Xueqi Li, Pengfei Liu, Zhijie Deng]
year: 2026
venue: arXiv
tags: [vla, world-model, embodied-foundation-model, flow-matching, test-time-scaling, robot-manipulation, long-horizon]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.05979v1
created: 2026-06-05
---

# 论文笔记：World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 上海交通大学 Deng Zhijie 组；南京邮电大学 |
| 日期 | June 2026 |
| 项目主页 | https://github.com/SJTU-DENG-Lab/WLA |
| 对比基线 | [[GR00T N1.7]]、[[π0]]、[[CoT-VLA]]、[[DreamVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05979) / [Code](https://github.com/SJTU-DENG-Lab/WLA) |

---

## 一句话总结

> WLA 用单一自回归骨干同时输出子任务文本、未来子目标图像、机器人动作三种模态，把 [[World Action Model]] 与 [[Vision-Language-Action Model|VLA]] 的语义推理能力合并，并通过推理时仅启用动作专家实现 40 ms 的实时控制。

---

## 核心贡献

1. **统一三模态生成范式**: 提出 [[World-Language-Action Model|WLA]] 架构，把高层 [[Language Reasoning|语言子任务推理]]、低层 [[Action-Conditioned World Model|未来观测预测]] 与 [[Action Chunking|动作块]] 生成耦合到同一个 AR Transformer 中，三者共享 backbone 表征但通过不同 expert head 解码。
2. **隐式世界-动作耦合**: 训练时 [[World Expert]] 与 [[Action Expert]] 共用 backbone 参数，使语义动力学先验注入动作策略；推理时可直接丢弃 World Expert，保留 2B 活跃参数与 ~40 ms 的低延迟。
3. **可选 [[Test-Time Scaling|测试时扩展]]**: 推理时采样 $K$ 个候选动作块，用 World Expert 预测对应未来图像，再用 [[Value Model|价值模型]] 评分，挑选最优，显著提升长程任务表现。
4. **大规模 SOTA 验证**: 在 [[RoboTwin 2.0]] 50 个双臂任务上达到 92.94%（Clean）/ 90.02%（Randomized），[[LIBERO]] 平均 98.6%，[[RMBench]] 长程记忆基准 56.5%（vs Fast-WAM 13.3%）。

---

## 问题背景

### 要解决的问题

当前具身基础模型走向两个极端：
- **[[Vision-Language-Action Model|VLA]]** 走 VLM 路线，擅长语言指令理解和短程动作，但缺乏对未来视觉状态的显式建模，长程任务和 OOD 场景下脆弱；
- **[[World Action Model]] (WAM)** 显式建模未来帧（如 [[DreamVLA]]、[[Cosmos-Policy]]），但完整视频生成开销大，且缺少语言层级的子任务规划，难以做语义推理。

如何把这两条路线**统一**且保持推理时**可实时部署**，是 WLA 的核心问题。

### 现有方法的局限

1. 纯 VLA（如 [[OpenVLA]]、[[π0]]）将世界知识完全隐式压入策略权重，缺乏可解释的中间表示；
2. 纯 WAM（如 [[Vid2World]]、[[DreamVLA]]）预测完整视频帧，推理慢（>200 ms），难以在线控制；
3. 两阶段流水线（如 [[GR-2]]、[[CoT-VLA]] 把规划与控制割裂）训练不联合优化，存在分布偏移。

### 本文的动机

作者认为 **语言子任务 + 单帧未来 + 动作** 三者是机器人决策的最小充分表征：语言提供长程语义、单帧未来提供物理动力学先验、动作直接落地控制。三者在训练时联合优化、在推理时按需启用，可同时拿到性能和速度。

---

## 方法详解

### 模型架构

[[World-Language-Action Model|WLA]] 采用 **三专家 AR Transformer** 架构：

- **输入**: 语言指令 $\ell$ + 历史观测 $o_{t-h}$ + 当前观测 $o_t$ + 机器人状态 $q_t$ + 记忆 $\mathcal{M}$
- **Backbone**: [[RynnBrain]]-2B（2.1B 参数，AR Transformer，初始化自 VLM 预训练）
- **核心三专家**:
  - [[Language Expert]]: 自回归解码子任务文本 $\mathcal{S}_t$（如 "grasp the red cup"）
  - [[World Expert]]: [[SANA]]-600M 扩散 head，预测未来 [[VAE]] 潜变量 $o_{t+n}$
  - [[Action Expert]]: [[Flow Matching|flow-matching]] head（390M），输出 [[Action Chunking|动作块]] $a_{t:t+n}$
- **共享 [[Meta Query|Meta-Query]]**: 用因果注意力聚合 backbone 输出到固定长度隐藏态 $h_t$，供 World/Action Expert 共用
- **总参数**: 3.4B；推理活跃 2B（关闭 World Expert）

### 核心模块

#### 模块 1: Language Expert（语言子任务推理）

**设计动机**: 在控制信号之前显式生成自然语言子目标，让模型"先想后做"，对应于 [[Chain-of-Thought|CoT]] 风格的[[Embodied Reasoning|具身推理]]。

**具体实现**:
- 输入 $o_{t-h}, o_t, \ell, \mathcal{M}$，输出文本子任务 $\mathcal{S}_t$（公式 3.1）
- 用 backbone 自回归解码 token，加入到后续 World/Action Expert 的上下文
- 训练时与人工标注子任务对齐，损失项 $\mathcal{L}_{lang}$，权重 $\beta = 0.005$

#### 模块 2: World Expert（未来子目标图像）

**设计动机**: 把"看一帧 $n$ 步后的目标画面"作为物理动力学的紧凑监督信号，比生成完整视频便宜得多，又比纯潜变量更可解释。

**具体实现**:
- 仅预测**单帧** $o_{t+n}$（典型 $n=16$），不生成视频序列
- 用 [[SANA]] 扩散头在 [[VAE]] 潜空间生成（公式 3.3），训练目标 $\mathcal{L}_{vm}$
- 关键技巧：World Expert 的梯度通过 backbone 反传，**强制 Action Expert 共享同一份动力学先验**
- 推理时可关闭，但训练好的 backbone 已经携带"未来感"

#### 模块 3: Action Expert（动作 flow matching）

**设计动机**: 用 [[Flow Matching|flow matching]] 替代传统离散动作 token 解码，连续动作空间更自然、多模态分布更好建模（参考 [[π0]]、[[FLOWER]]）。

**具体实现**:
- 输入 $h_t, q_t$，输出 $H$ 步动作块 $a_{t:t+H}$（公式 3.4）
- flow matching 在 $\tau \sim \mathcal{U}(0,1)$ 时间步上回归速度场
- [[Action Chunking|动作块]] 长度按任务调整（LIBERO H=10，RoboTwin H=20）

#### 模块 4: Test-Time Scaling（可选推理增强）

**设计动机**: 长程或高随机性任务时，单条动作链不够鲁棒。借助已经训练好的 World Expert，可以"先想象再执行"。

**具体实现**:
1. Action Expert 采样 $K$ 个候选动作块 $\{a^{(k)}\}_{k=1}^K$
2. 对每个候选用 World Expert 想象对应未来帧 $o^{(k)}_{t+n}$
3. [[Value Model|价值模型]] $V(o^{(k)}_{t+n}, \ell)$ 评分
4. 选 $\arg\max_k V$ 执行

#### 模块 5: 推理加速

为了把 116 ms 压到 40 ms，作者用了三组工程优化：
- **[[CUDA Graph]] capture**: 消除 Python dispatch 开销
- **算子融合**: 自定义 Triton kernel 融合 RMSNorm + RoPE + Attention
- **预计算与缓存**: 语言 embedding、RoPE table、KV cache 一次算完

---

## 关键公式

### 公式 1: [[Language Expert|子任务推理]]

$$
\mathcal{S}_t = f_{\text{lang}}\big(o_{t-h},\, o_t,\, \ell,\, \mathcal{M}\big)
$$

**含义**: 给定历史/当前观测、语言指令和记忆，AR 解码当前子任务文本 $\mathcal{S}_t$。

**符号说明**:
- $\ell$: 顶层语言指令（如 "clean the table"）
- $\mathcal{M}$: 任务记忆（已完成子任务列表）
- $\mathcal{S}_t$: 当前 step 的子任务文本（如 "pick up the sponge"）

### 公式 2: [[Meta Query|物理动力学隐变量]]

$$
h_t = f_{\text{dyn}}\big(o_{t-h},\, o_t,\, \ell,\, \mathcal{M},\, \mathcal{S}_t,\, Q\big)
$$

**含义**: 把所有上下文连同可学习的 meta-query $Q$ 一起送入 backbone，输出共享隐藏态 $h_t$，作为 World/Action Expert 的输入桥梁。

**符号说明**:
- $Q$: 一组可学习 meta-query token，用因果注意力压缩上下文
- $h_t$: 共享隐藏态（World/Action Expert 共用）

### 公式 3: [[Action-Conditioned World Model|未来帧预测]]

$$
o_{t+n} = f_{\text{wm}}\big(h_t,\, o_t\big)
$$

**含义**: World Expert 基于 $h_t$ 和当前观测，在 [[VAE]] 潜空间生成 $n$ 步之后的子目标帧。

### 公式 4: [[Flow Matching|动作块生成]]

$$
a_{t:t+n} = f_{\text{act}}\big(h_t,\, q_t\big)
$$

**含义**: Action Expert 基于共享隐藏态和当前关节状态 $q_t$，flow-matching 解码得到未来 $n$ 步动作块。

### 公式 5: 联合训练损失

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + \alpha \cdot \mathcal{L}_{\text{vm}} + \beta \cdot \mathcal{L}_{\text{lang}}
$$

**含义**: 动作损失为主项，加权融合未来帧扩散损失与语言子任务损失。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: [[Flow Matching|flow matching]] 速度场回归损失
- $\mathcal{L}_{\text{vm}}$: World Expert 的扩散损失（VAE 潜空间）
- $\mathcal{L}_{\text{lang}}$: 子任务文本的 next-token CE 损失
- $\alpha = 0.1,\ \beta = 0.005$: 经验权重，作者强调 $\alpha$ 不能过大否则压制动作精度

### 公式 6: Test-Time Scaling 动作选择

$$
a^\star_{t:t+n} = \arg\max_{k \in \{1,\dots,K\}} V\big(f_{\text{wm}}(h_t, o_t \mid a^{(k)}),\ \ell\big)
$$

**含义**: 对 $K$ 个候选动作块，分别用 World Expert 想象未来帧，再用价值模型 $V$ 评分，取最高分作为执行动作。

---

## 关键图表

### Figure 1: VLA / WAM / WLA 范式对比 + WLA-0 总览

![Figure 1](https://arxiv.org/html/2606.05979v1/x1.png)

**说明**: 上半部分对比三种范式：纯 [[Vision-Language-Action Model|VLA]] 缺少未来建模、纯 [[World Action Model|WAM]] 缺少语义子任务、WLA 同时输出 (text, image, action) 三模态。下半部分汇报 WLA-0 在 RoboTwin 2.0 / LIBERO / RMBench 上的表现以及 40 ms 推理延迟。

### Figure 2: WLA 架构总览 + Test-Time Scaling

![Figure 2](https://arxiv.org/html/2606.05979v1/x2.png)

**说明**: 左边展示 [[RynnBrain]] backbone + 三专家（Language / World / Action）与共享 [[Meta Query]]；右边展示 [[Test-Time Scaling]] 模式：动作专家采样 $K$ 条 → World Expert 想象 → 价值模型选最优。注意 World Expert 在效率模式下可整体丢弃。

### Figure 3: 真实世界任务 + OOD 场景 + 延迟对比

![Figure 3](https://arxiv.org/html/2606.05979v1/x3.png)

**说明**: 在 Agilex Piper 双臂平台上展示 4 个真机任务（Unscrew Cap / Pack Object / Stack Cup / Dispose Trash）的标准、OOD 物体、OOD 场景三种设置，配以延迟柱状图，WLA 40 ms 相比 [[Motus]] >1600 ms 快约 40×。

### Figure 4: Beat Block Hammer 三阶段执行

![Figure 4](https://arxiv.org/html/2606.05979v1/x4.png)

**说明**: 展示 RoboTwin 2.0 的 Beat Block Hammer 任务在 Clean / Randomized / Hard 三档设置下的执行轨迹，可视化 WLA 在复杂随机化布局下仍能完成 pick-place-hit 长程子任务序列。

### Figure 5: 视频学习未见任务

![Figure 5](https://arxiv.org/html/2606.05979v1/x5.png)

**说明**: WLA 通过观看人类第一视角视频或跨本体视频，迁移到 5 个 RoboTwin 2.0 未见任务上的实验。同本体视频迁移效果显著（13.0%→34.4%），人类视频迁移因领域差距未成功（作者列为局限）。

### Table 1: RoboTwin 2.0 主结果

| 方法 | Clean | Randomized |
|---|---|---|
| [[π0]] | 78.1% | 71.4% |
| [[GR00T N1.7]] | 85.3% | 80.2% |
| WLA (w/o $\mathcal{L}_{vm}$) | 90.98% | 89.34% |
| **WLA-0** | **92.94%** | **90.02%** |

**说明**: 50 个双臂任务、2,500 clean + 25,000 randomized 轨迹。$\mathcal{L}_{vm}$ 隐式动力学监督带来约 +2 pt 提升，证明 World Expert 即使推理时丢弃也对策略有正面影响。

### Table 2: LIBERO 全套件

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| [[OpenVLA]] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[π0]] | 96.4 | 96.8 | 88.5 | 85.2 | 91.7 |
| **WLA-0** | **99.4** | **99.6** | **98.8** | **96.6** | **98.6** |
| WLA-0 + TTS | 99.6 | 99.8 | 99.0 | 97.2 | **98.9** |

**说明**: 平均 98.6%，长程 Long 套件提升尤其明显（vs π0 +11.4 pt），印证语言子任务对长程任务的关键作用。

### Table 3: RMBench 长程记忆基准

| 方法 | Battery Try | Blocks Ranking | Cover Blocks | Press Button | 平均 |
|---|---|---|---|---|---|
| Fast-WAM | 6% | 4% | 18% | 25% | 13.3% |
| [[DreamVLA]] | 22% | 11% | 47% | 41% | 30.3% |
| **WLA-0** | **45%** | **23%** | **84%** | **74%** | **56.5%** |

**说明**: RMBench 测试需要长时记忆的任务，WLA 相对最强 baseline 提升 +26 pt，说明显式语言子任务 + 记忆 $\mathcal{M}$ 设计对长程信息留存关键。

### Table 4: 真机 OOD 评估（10 trials/setting）

| Task | Standard | OOD Object | OOD Scenario |
|---|---|---|---|
| Unscrew Cap | 7/10 | 3/10 | 6/10 |
| Pack Object | 7/10 | 5/10 | 4/10 |
| Stack Cup | 10/10 | 9/10 | 7/10 |
| Dispose Trash | 6/10 | 4/10 | 2/10 |

**说明**: 真实 Piper 双臂平台测试，OOD 物体一般下降 ~20-30 pt，OOD 场景下降更多，符合预期。Stack Cup 几何稳定任务在 OOD 物体上几乎不下降。

### Table 5: 视频学习消融

| 配置 | Clean | Randomized |
|---|---|---|
| Seen-Action baseline | 13.0% | 11.6% |
| + Same-Embodiment Video | **34.4%** | **30.0%** |
| + Cross-Embodiment Video | 28.8% | 27.4% |
| + Human Egocentric Video | 13.4% | 12.1% |

**关键发现**: 同本体视频带来 +21 pt，跨本体也能带来 +16 pt，但人类第一视角视频几乎无效，提示当前世界模型对人-机器人形态差距仍敏感。

---

## 实验结果

### 数据集与基准

| 数据集/Benchmark | 规模 | 用途 |
|---|---|---|
| [[RoboTwin 2.0]] | 50 任务，2.5k clean + 25k randomized 轨迹 | 双臂仿真训练/测试 |
| [[LIBERO]] | 4 suites × 10 任务 | 通用 VLA 基准 |
| [[RMBench]] | 4 个长程记忆任务 | 长时记忆评估 |
| Real Piper 平台 | 4 任务 × 3 设置 × 10 trials | 真机泛化 |
| Egocentric Video | ~50h 人类视频 | 视频迁移学习 |

### 实现细节

- **Backbone**: [[RynnBrain]]-2B（VLM 预训练初始化）
- **优化器**: AdamW，学习率 $5\times10^{-5}$（base）/ $5\times10^{-6}$（min），weight decay $1\times10^{-8}$
- **Warmup**: 1,000 steps；梯度裁剪 1.0
- **Batch Size**: 256–448（按 benchmark 调整）
- **训练步数**: 30k–100k
- **硬件**: 8× H100 训练；推理 1× RTX 5090

### 关键定量结论

1. **仿真 SOTA**: RoboTwin 2.0 Clean 92.94%、LIBERO 98.6%、RMBench 56.5%，全部刷新当前最佳。
2. **推理实时**: 关闭 World Expert 后 ~40 ms / 步，相比同等性能 [[Motus]]（>1600 ms）快约 **40×**。
3. **隐式监督有效**: 即使推理不用 World Expert，训练时加入 $\mathcal{L}_{vm}$ 带来 +2 pt（92.94 vs 90.98），证明动力学梯度有正向迁移。
4. **TTS 收益边际递减**: LIBERO 上 TTS 仅 +0.3 pt（98.6→98.9），但作者声称在 OOD 真机上收益更大。
5. **真机泛化合理**: Standard 平均 30/40，OOD Object 21/40，OOD Scenario 19/40，符合一般 VLA 的衰减曲线。
6. **视频迁移**: 同本体视频带来巨大增益（+21 pt），但人类视频几乎无效，是未来方向。

### 工程加速

| 优化 | 延迟（ms） |
|---|---|
| 原始 PyTorch | ~116 |
| + [[CUDA Graph]] | ~78 |
| + Triton 算子融合 | ~52 |
| + 预计算 / KV cache | **~40** |

---

## 批判性思考

### 优点

1. **范式清晰**: "language + future image + action" 三模态最小充分集，比纯 VLA 多语义、比纯 WAM 多速度。
2. **训练-推理解耦**: 训练时世界模型监督，推理时按需启用，工程友好。
3. **大规模验证**: 三大公开基准 + 真机 + 视频学习全覆盖，结果可信。
4. **加速工程扎实**: 116→40 ms 的工程细节披露完整，可复现。

### 局限性

1. **单一硬件平台**: 真机仅 Agilex Piper 双臂，泛化性证据不充分。
2. **TTS 收益有限**: LIBERO 上 +0.3 pt，回报与额外延迟（K 倍 World Expert 推理）不成比例。
3. **人类视频迁移失败**: 作者承认 ego-video 域差距未解决，限制了从互联网视频中扩展数据的潜力。
4. **子任务标注成本**: $\mathcal{L}_{lang}$ 依赖人工子任务标注，规模化困难。
5. **单帧未来过于稀疏**: 只预测 $n$ 步后单帧，对快速变化场景（如倒水）可能丢失中间动力学。

### 潜在改进方向

1. 用自动子任务挖掘（如 VLM zero-shot）替代人工标注；
2. 多帧未来或 latent video 监督，弥补单帧稀疏问题；
3. Test-Time Scaling 用更小的代理价值模型（如 [[CLIP]] 评分）降低开销；
4. 引入 [[Cross-Embodiment Training]] 减少同本体依赖。

### 可复现性评估

- [x] 代码开源（https://github.com/SJTU-DENG-Lab/WLA）
- [ ] 预训练模型（暂未释出）
- [x] 训练细节完整（优化器、batch、step 均披露）
- [x] 数据集可获取（RoboTwin 2.0 / LIBERO / RMBench 均公开）

---

## 关联笔记

### 基于
- [[π0]]: flow matching 动作头的直接前身
- [[DreamVLA]]: 显式未来帧预测的 WAM 路线代表
- [[CoT-VLA]]: 语言子任务推理的早期尝试
- [[RynnBrain]]: backbone 来源

### 对比
- [[GR00T N1.7]]: 纯 VLA SOTA，本文主要比较对象
- [[Motus]]: 同样三模态但推理慢 40× 的对照
- [[Vid2World]]: 完整视频生成的 WAM 路线

### 方法相关
- [[World-Language-Action Model]]: 本文提出的核心范式
- [[Flow Matching]]: 动作生成方法
- [[SANA]]: World Expert 扩散 backbone
- [[Action Chunking]]: 动作输出粒度
- [[Test-Time Scaling]]: 推理增强机制
- [[Meta Query]]: backbone-expert 桥梁
- [[CUDA Graph]]: 推理加速

### 硬件/数据相关
- [[Agilex Piper]]: 真机平台
- [[RoboTwin 2.0]]: 主仿真基准
- [[LIBERO]]: 通用 VLA 基准
- [[RMBench]]: 长程记忆基准

---

## 速查卡片

> [!summary] WLA: World-Language-Action Model
> - **核心**: 单 AR backbone + 三专家（Language / World / Action），训练时联合优化，推理仅启用 Action Expert
> - **方法**: 子任务文本 + 单帧未来潜变量 + flow-matching 动作块；可选 Test-Time Scaling
> - **结果**: RoboTwin 2.0 92.94% / LIBERO 98.6% / RMBench 56.5% / 40 ms 推理
> - **代码**: https://github.com/SJTU-DENG-Lab/WLA

---

*笔记创建时间: 2026-06-05*
