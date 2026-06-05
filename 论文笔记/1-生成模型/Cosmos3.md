---
title: "Cosmos 3: Omnimodal World Models for Physical AI"
method_name: "Cosmos3"
authors: [NVIDIA Cosmos Team]
year: 2026
venue: arXiv
tags: [omnimodal-foundation-model, world-model, mixture-of-transformers, vision-language-action, video-generation, physical-ai]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/abs/2606.02800
created: 2026-06-05
---

# 论文笔记：Cosmos 3: Omnimodal World Models for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | June 2026 |
| 项目主页 | https://research.nvidia.com/labs/cosmos-lab/cosmos3 |
| 代码 | https://github.com/nvidia/cosmos |
| 权重 | https://huggingface.co/collections/nvidia/cosmos3 |
| License | OpenMDW-1.1 |
| 对比基线 | [[Cosmos-Policy]] / [[BAGEL]] / [[Chameleon]] / [[Qwen3-VL]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02800) / [Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) |

---

## 一句话总结

> Cosmos 3 用一个 [[MoT|Mixture-of-Transformers]] 双塔架构（AR 推理塔 + 扩散生成塔）统一了 [[VLM]]、视频生成、世界模拟、[[VLA]] 四类 Physical AI 任务，是首个全开源的 omni-modal Physical AI 基座。

---

## 核心贡献

1. **统一架构**: 提出 [[Two-Tower MoT]] 两塔混合 Transformer，AR 塔做离散 token 推理、扩散塔做连续 token 生成，通过 [[JointSelfAttention|joint attention]] 单向连接（推理 → 生成），单个模型同时胜任 VLM / video generator / world simulator / [[VLA]] 四类工作。
2. **Omni-modality**: 原生支持语言 / 图像 / 视频 / 音频 / [[Action Chunking|动作]] 五种模态的输入与输出，通过 [[Modality-Specific Encoder|模态特化编码器]]（[[ViT]] / [[3D Causal VAE]] / 动作 embedding）投到共享表征空间，再用 [[3D mRoPE|3D 多模态旋转位置编码]] 统一时空对齐。
3. **大规模 Physical AI 数据**: 1.3B 数据点 / 393 个数据集，覆盖 9 种 [[Embodiment Anchoring|具身形态]]（Franka / UR / Agibot / Google Robot / WidowX / UMI / 自车 / 头戴相机 / 通用相机），开源 6 个 SDG 数据集 + checkpoints + 训练脚本 + NIM 微服务部署。
4. **SOTA 性能**: 开源 SOTA on [[R-Bench]] / PAI-Bench / [[Physics IQ|Physics-IQ]] / RoboLab / RoboArena / Artificial Analysis text-to-image & image-to-video / VANTAGE-Bench / Traffic Anomaly Reasoning。

---

## 问题背景

### 要解决的问题

Physical AI 需要同时具备 **物理推理**（理解物体交互、空间关系、动力学）、**世界生成**（合成符合物理的视频以作 SDG）、**动作生成**（输出可执行控制）三类能力。但既往工作把这三件事拆成完全独立的模型族：

- [[VLM]] 只做理解（如 [[Qwen3-VL]]）
- [[Video Diffusion Model|视频扩散模型]] 只做生成（如 [[CogVideoX]] / [[WAN]]）
- [[VLA]] 只做控制（如 [[Cosmos-Policy]] / [[OpenVLA]]）

每个模态家族各自训练，参数无法复用，跨任务知识不能迁移。

### 现有方法的局限

1. **模态割裂**: 同一团队需要维护 3-4 套基座，部署成本高
2. **AR 与 Diffusion 难统一**: AR 适合离散文本/动作 token、Diffusion 适合连续视觉/音频信号，简单拼接（如 [[Chameleon]] 把所有东西 tokenize 成离散 token）会损失生成质量
3. **跨模态 Scaling Law 不清**: 缺乏统一架构来研究多模态联合预训练的规模效应

### 本文的动机

把"推理"与"生成"作为两条独立的 [[Mixture of Contrastive Experts|参数路径]] 并行共存于同一个 Transformer 栈内部：每层都有 AR-expert FFN/attention 和 Diffusion-expert FFN/attention，token 走哪条专家由模态决定；两条专家通过共享的 attention QKV 空间彼此可见，从而在一个 forward pass 内完成"看 → 想 → 画 / 动"。

---

## 方法详解

### 模型架构

Cosmos 3 采用 **Two-Tower [[MoT|Mixture-of-Transformers]]** 架构：

- **输入**: 文本 $x_\text{text}$ + 图像 $x_\text{img}$ + 视频 $x_\text{vid}$ + 音频 $x_\text{aud}$ + 动作 $x_\text{act}$ + 噪声 latent $z_\tau$
- **Modality Encoders**:
  - 视觉理解侧: [[ViT|Vision Transformer]] 提取语义 token
  - 视觉/音频生成侧: [[3D Causal VAE]] 压缩为连续 latent
  - 动作侧: 按 [[Embodiment Anchoring|embodiment]] 划分的可学习 projection（9D 相机 / 10D 单臂 / 20D 双臂 / 29D Agibot / 57D 头戴）
- **Backbone**: 两座 Transformer 塔（AR 推理塔 + Diffusion 生成塔），均从 [[Qwen3-VL]] 预训练权重初始化
- **共享机制**:
  - 共享 [[3D mRoPE|3D 多模态 RoPE]]（时间 / 高 / 宽 三个维度独立编码，动作 token 沿时间轴对齐到对应视频帧）
  - 共享 attention QKV 投影空间 → joint attention
- **输出**: 文本 token（AR head）/ 视频 latent（diffusion head）/ 音频 latent（diffusion head）/ 动作序列 $a_{t:t+k}$（diffusion head）

### 核心模块

#### 模块 1: Two-Tower MoT 路由

**设计动机**: AR 与 Diffusion 各有所长，强行用 single tower（如 [[Chameleon]] 全离散）会牺牲生成保真度，而 cascaded model（如 [[BAGEL]] 视觉部分 frozen）则限制端到端学习。Cosmos 3 让二者**参数解耦、attention 共享**。

**具体实现**:

每个 Transformer 层把 token 分成两个子序列 $\mathcal{S}_\text{AR}$ 和 $\mathcal{S}_\text{DM}$，按模态指派：

- 文本 / 离散动作 token → AR 子序列，走 AR expert FFN，使用 [[Causal Self-Attention|causal mask]]
- 视觉 / 音频 latent / 连续动作 token → DM 子序列，走 Diffusion expert FFN，使用 [[Bi-Directional Attention|bidirectional mask]]

两个子序列在每层进入同一组 attention，所有 token 都能看到 AR 子序列（即 AR 提供 condition），但**只有 DM 子序列内部能互相看到 DM tokens**（推理 → 生成是单向流）。这样 AR 塔可以独立运行（生成塔被砍掉时不影响推理），DM 塔必须依赖 AR 塔。

#### 模块 2: Modality-Specific Encoding 与 3D mRoPE

**设计动机**: 不同模态的时空结构差异巨大（文本只有 1D 序列，图像 2D，视频/音频 3D，动作沿时间 1D），需统一到一个 attention 可索引的位置空间。

**具体实现**:

- 每个 visual token 拿到 $(t, h, w)$ 三元坐标，每个动作 token 拿到 $(t, 0, 0)$ 与对应视频帧对齐
- [[3D mRoPE]] 把 hidden 维度切成三段，分别对 $t, h, w$ 做 [[RoPE]] 旋转
- 文本 token 复用 1D RoPE，但与视觉 token 的 $t$ 维度对齐到 prompt 开始时刻

#### 模块 3: Efficient Video Sampling (EVS)

**设计动机**: 视频 token 数随分辨率与帧数平方增长，720p × 189 帧推理时 token 数破百万。

**具体实现**: 在 self-attention 层用基于运动幅度的 token pruning，把静态区域的冗余 token 合并，保留高运动信息密度的 token。配合 vLLM 服务框架与 [[NVFP4]] 量化，Nano 在 RTX PRO 6000 上端到端推理可达亚秒级。

---

## 关键公式

### 公式1: [[Mixture-of-Transformers Routing|MoT 路由]] 与 Joint Attention

每层把输入 $X = [X_\text{AR}; X_\text{DM}]$ 路由到两组 expert 参数 $\theta_\text{AR}, \theta_\text{DM}$：

$$
\text{FFN}(x_i) = \begin{cases} \text{FFN}_{\theta_\text{AR}}(x_i), & i \in \mathcal{S}_\text{AR} \\ \text{FFN}_{\theta_\text{DM}}(x_i), & i \in \mathcal{S}_\text{DM} \end{cases}
$$

而 attention 则在两个子序列之间联合计算，但带模态相关掩码 $M$：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}} + M\right) V
$$

**含义**: FFN 参数按模态选择专家以避免互相干扰，attention 则共享以实现跨模态 grounding。掩码 $M$ 强制 DM 子序列只能看到 AR 子序列（单向 condition），AR 子序列内部 causal、DM 子序列内部 bidirectional。

**符号说明**:
- $\mathcal{S}_\text{AR}, \mathcal{S}_\text{DM}$: AR 与 Diffusion 子序列的 token 集合
- $\theta_\text{AR}, \theta_\text{DM}$: 两个 expert 的独立参数（FFN + attention 投影都翻倍）
- $M$: 模态条件掩码，$M_{ij} = -\infty$ 表示 $i$ 不可见 $j$
- $d$: attention head 维度

### 公式2: [[Flow Matching]] 视觉生成损失

DM 塔采用 [[Rectified Flow]] 训练范式，对干净 latent $z_0$ 与噪声 $\epsilon \sim \mathcal{N}(0, I)$ 线性插值：

$$
z_\tau = (1 - \tau) z_0 + \tau \epsilon, \quad \tau \sim \mathcal{U}(0, 1)
$$

模型 $v_\theta$ 预测速度场，损失为：

$$
\mathcal{L}_\text{flow} = \mathbb{E}_{z_0, \epsilon, \tau, c}\big[\, \big\| v_\theta(z_\tau, \tau, c_\text{AR}) - (\epsilon - z_0) \big\|^2 \,\big]
$$

**含义**: 让生成塔学习从噪声指向干净 latent 的速度场，条件 $c_\text{AR}$ 来自 AR 塔输出的隐藏状态（推理 → 生成单向流）。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 均匀采样的噪声时间步
- $z_0$: 来自 [[3D Causal VAE]] 编码的视频/音频/动作 latent
- $\epsilon$: 标准高斯噪声
- $c_\text{AR}$: AR 塔输出作为条件（文本指令 / 图像 / 状态历史）
- $v_\theta$: DM 塔预测的速度场

### 公式3: 联合训练目标

AR 塔的 [[Next-Token Prediction|下一 token 预测]] 损失与 DM 塔的 flow 损失加权求和：

$$
\mathcal{L}_\text{total} = \lambda_\text{AR} \cdot \mathcal{L}_\text{AR} + \lambda_\text{DM} \cdot \mathcal{L}_\text{flow}
$$

其中

$$
\mathcal{L}_\text{AR} = - \sum_{i \in \mathcal{S}_\text{AR}} \log p_\theta(x_i \mid x_{<i})
$$

**含义**: 两塔在同一个 batch 中联合优化，模态 mask 决定每个 token 走哪个损失分支。论文中 $\lambda_\text{AR}, \lambda_\text{DM}$ 按数据集类型动态调度（纯文本 batch $\lambda_\text{DM}=0$，纯视觉生成 batch $\lambda_\text{AR}$ 取小值）。

**符号说明**:
- $\lambda_\text{AR}, \lambda_\text{DM}$: 模态相关权重
- $x_i$: AR 子序列中第 $i$ 个 token
- $p_\theta$: AR 塔的 softmax 概率

### 公式4: 推理时的 [[Classifier-Free Guidance]]

视觉/动作生成采样使用 CFG，引导强度 $w$：

$$
\hat{v}_\theta(z_\tau, \tau, c) = (1 + w) \, v_\theta(z_\tau, \tau, c) - w \, v_\theta(z_\tau, \tau, \varnothing)
$$

**含义**: 通过随机丢弃条件 $c$ 训练 unconditional 分支，推理时外推到更强的条件依赖以提升与文本/图像 prompt 的对齐度。

**符号说明**:
- $w$: guidance scale，论文典型值 7.5（图像）/ 5.0（视频）
- $\varnothing$: 空条件（dropout 训练时随机替换）

---

## 关键图表

### Figure 1: Cosmos 3 Model Architecture / 模型架构总览

![Figure 1](https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/cosmos3-model-architecture.png)

**说明**: Cosmos 3 的 Two-Tower [[MoT|Mixture-of-Transformers]] 架构图。左侧 AR Reasoner 塔接收文本 + 视觉理解 token（[[ViT]] 编码），右侧 Diffusion Generator 塔接收带噪 latent（[[3D Causal VAE]] 编码）。两塔每层共享 attention QKV 空间但 FFN 参数独立，AR 输出通过 joint attention 单向流向 DM 塔作为生成条件。所有 token 经 [[3D mRoPE]] 对齐到统一时空坐标。

### Figure 2: Unified Capabilities Overview / 统一能力总览

![Figure 2](https://www.engineering.com/wp-content/uploads/2026/06/nvidia-cosmos-3-1.jpg)

**说明**: Cosmos 3 在单一权重下覆盖的下游任务示意：[[VLM]]（物理推理 / 视频 QA / 异常检测）、文生图与图生视频、[[Action-Conditioned World Model|动作条件世界模拟]]（forward dynamics + inverse dynamics）、机器人 [[VLA|策略]]（Cosmos3-Nano-Policy-DROID）。展示同一基座可通过 prompt 切换"看 / 想 / 画 / 动"四种工作模式。

### Table 1: Model Family / 模型家族

| 模型 | 总参数 | Backbone (Qwen3-VL) | 配置 | 目标硬件 |
|------|--------|---------------------|------|----------|
| Cosmos 3 Edge | 4B | 2B | 单塔精简版 | 边缘设备（规划中）|
| Cosmos 3 Nano | 16B | 8B | 8B reasoner + 8B generator | RTX PRO 6000 workstation |
| Cosmos 3 Super | 64B | 32B | 32B reasoner + 32B generator | Hopper / Blackwell datacenter |
| Cosmos3-Super-Text2Image | 64B | 32B | 文生图专用微调 | datacenter |
| Cosmos3-Super-Image2Video | 64B | 32B | 图生视频专用微调 | datacenter |
| Cosmos3-Nano-Policy-DROID | 16B | 8B | [[VLA]] 微调（DROID）| workstation |

**说明**: 全系列分 Edge / Nano / Super 三档。Nano 与 Super 因为双塔，总参数约等于两倍 backbone，但通过 MoT 路由实际激活参数仅约一倍（参数共享 attention 部分）。

### Table 2: Supported Modalities & IO / 模态与 IO 规格

| 维度 | 规格 |
|------|------|
| 文本输入 | 最长 4096 token |
| 图像输入/输出 | 256p / 480p / 720p；16:9, 4:3, 1:1, 3:4, 9:16 |
| 视频输出 | 5–400 帧；10/16/24/30 FPS；默认 189 帧 ≈ 7.9 秒 |
| 音频输出 | AAC stereo, 48 kHz, 最长 0.5 秒（Nano）|
| Action 输出 | JSON 序列，9 种 embodiment：9D camera / 9D vehicle / 57D egocentric / 10D Franka / 20D dual-Franka / 29D Agibot / 10D UR / 10D Google Robot / 10D WidowX / 9D UMI |

**说明**: Embodiment 维度通过专属 [[Embodiment Anchoring]] projection 映射到统一 hidden size，使单一模型可跨多种机器人形态共享主干。

### Table 3: 训练数据规模

| 数据类型 | 来源 | 规模 |
|----------|------|------|
| 图像 | OpenImage | 1.2M |
| 网络图文 | Coyo700M | 100M |
| 视频 | YouTube | 340M |
| 机器人数据 | AgiBot / DROID / 自采集 | 私有 |
| 自车场景 | Nexar | 私有 |
| 合成数据 (SDG) | 6 个新发布数据集（embodied/physics/spatial/digital human/AV/warehouse）| — |
| **合计** | 393 dataset entries | **1.3B 数据点** |

**说明**: 这是迄今为止最大规模的多模态 Physical AI 训练集合之一，6 个 SDG 数据集随论文同步开源。

---

## 实验结果

### 数据集

- **理解评测**: VANTAGE-Bench、Traffic Anomaly Reasoning（AI City Challenge 2026）
- **生成评测**: [[R-Bench]]、PAI-Bench、[[Physics IQ|Physics-IQ]]、Artificial Analysis text-to-image & image-to-video leaderboards
- **策略评测**: RoboLab、RoboArena
- **自研评测**: **HUE framework** — 把生成视频拆解成 yes/no 事实问题，覆盖 4 个维度（语义对齐 / 物理定律 / 几何推理 / 视觉完整性）× 7 个领域

### 实现细节

- **初始化**: Backbone 来自 [[Qwen3-VL]] 8B / 32B 预训练权重
- **训练范式**: 两阶段 — (1) Pretraining 联合优化 $\mathcal{L}_\text{AR} + \mathcal{L}_\text{flow}$；(2) Post-training — SFT + Action post-training（DROID / 9 embodiment）
- **生成范式**: [[Rectified Flow]] / [[Flow Matching]] for vision & audio
- **位置编码**: [[3D mRoPE]]
- **量化部署**: BF16 / FP8 / [[NVFP4]]（2× 加速）
- **服务**: vLLM + NVIDIA NIM 微服务

### 主要结果

**理解类**：
- VANTAGE-Bench：Nano 与 Super 各自规模档第一
- Traffic Anomaly Reasoning：登顶 AI City Challenge 2026 leaderboard

**生成类**：
- [[R-Bench]]：开源 SOTA
- PAI-Bench / [[Physics IQ|Physics-IQ]]：领先所有开源模型
- Artificial Analysis text-to-image & image-to-video：开源 leaderboard 第一

**Physical AI 控制类**：
- RoboLab / RoboArena：登顶开源 leaderboard
- Cosmos3-Nano-Policy-[[DROID 数据集|DROID]]：作为 [[VLA]] 微调版本，作者声称在工作站上达到亚秒级推理 + SOTA 操作成功率

**消融观察**（来自架构讨论）：
- 单塔（全部走 AR）：视觉生成 FID 显著退化，验证 MoT 路由的必要性
- 共享 RoPE 1D（不用 3D mRoPE）：视频时序一致性下降
- 移除 EVS：720p 视频推理时延翻倍以上

> 注：技术报告 PDF 含完整数值表，本笔记因 fetch size 限制未抓取所有 row，原始数字以 [Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) 为准。

---

## 批判性思考

### 优点

1. **真正统一的多任务基座**: 一份权重切多种任务模式，省下维护多个独立模型的成本
2. **AR + Diffusion 解耦设计**: MoT 路由让推理与生成参数各自优化，又通过共享 attention 实现 grounding，比 Chameleon 全离散方案保真度更高，又比 BAGEL 类 cascade 方案更端到端
3. **完整开源 + 工业级部署**: 开 6 个 SDG 数据集 + 全 checkpoint + NIM 微服务 + FP8 / NVFP4 量化，对 Physical AI 社区是巨大推动
4. **Embodiment 普适**: 9 种具身形态共享一套基座，比单形态 VLA（如 OpenVLA 仅 7-DoF）更接近"通用机器人基座"愿景

### 局限性

1. **训练成本极高**: 64B Super 双塔 + 1.3B 多模态样本，复现门槛只有大厂能负担
2. **物理保真仍非严格**: 官方文档承认存在"temporal inconsistency, unstable camera or object motion, imprecise physical interactions"，缺乏显式 [[Physics Simulator]] 约束
3. **MoT 双塔参数翻倍**: 名义 16B/64B 几乎都是双塔总和，实际"激活参数"约为 backbone 单塔水平，命名上易混淆
4. **Action 头是 diffusion**：生成动作时无法像 AR 头那样实时流式输出（需多步 sampling），尽管 EVS + 量化部分弥补，但延迟仍是 robotics 实时控制的潜在瓶颈
5. **评测 leak 风险**: 自研 HUE framework 与训练数据来源有重叠（YouTube），跨家公司复现性存疑

### 潜在改进方向

1. **MoE 化的 Diffusion 塔**: 用 sparse MoE 替换 dense generator，进一步降低推理激活成本
2. **统一连续 token**: 把动作也走 AR 头（如 [[FAST]] tokenizer），减少 diffusion sampling 延迟
3. **跨形态在线适配**: 当前 9 种 embodiment 是 fixed projection，引入 hyper-network 让新机器人零样本插入
4. **加入显式物理先验**: 与 [[OrbiSim]] / [[Genesis]] 等仿真器闭环，把物理约束作为可微 loss

### 可复现性评估

- [x] 代码开源（GitHub: nvidia/cosmos, License OpenMDW-1.1）
- [x] 预训练模型（HuggingFace 全系列）
- [x] 训练细节公开（技术报告）
- [x] 数据集部分可获取（6 个 SDG 数据集开源，机器人私有数据未释放）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: 双塔 backbone 初始化权重
- [[Flow Matching]] / [[Rectified Flow]]: DM 塔训练范式
- [[3D Causal VAE]]: 视觉/音频 latent 编码器
- [[ViT]]: 视觉理解侧编码器
- [[3D mRoPE]]: 跨模态时空位置编码
- [[MoT|Mixture-of-Transformers]]: 整体路由架构

### 对比

- [[Chameleon]]: 全离散 token 的单塔 omni-modal，Cosmos 3 用 MoT 双塔在生成质量上更强
- [[BAGEL]]: 视觉 frozen 的 cascade 方案，Cosmos 3 端到端联合训练
- [[Cosmos-Policy]]: 上一代 NVIDIA 机器人策略，Cosmos 3 把它收编为单一基座的一个 mode
- [[OpenVLA]]: 经典 [[VLA]] 基座，Cosmos 3 提供更通用的跨 embodiment 替代

### 方法相关

- [[VLM]]: AR 塔单独运行时退化为 VLM
- [[VLA]]: 加 action head 与 SFT 即得 VLA（Cosmos3-Nano-Policy-DROID）
- [[Video Diffusion Model]]: DM 塔单独运行时退化为视频扩散模型
- [[World Action Model]]: 联合 forward dynamics + policy 头
- [[Action-Conditioned World Model]]: forward dynamics 模式
- [[JointSelfAttention]]: 两塔之间的信息流机制
- [[Classifier-Free Guidance]]: 推理采样
- [[Next-Token Prediction]]: AR 塔训练目标
- [[Embodiment Anchoring]]: 跨形态动作 projection
- [[Two-Tower MoT|Two-Tower Architecture]]: 整体框架

### 硬件/数据相关

- [[NVFP4]]: 4-bit 浮点量化部署
- [[DROID 数据集|DROID]]: VLA post-training 数据
- [[Physics IQ|Physics-IQ]]: 物理理解评测
- [[R-Bench]]: 视频生成评测

---

## 速查卡片

> [!summary] Cosmos 3: Omnimodal World Models for Physical AI
> - **核心**: Two-Tower MoT（AR 推理塔 + Diffusion 生成塔）统一 VLM / 视频生成 / 世界模拟 / VLA 四类任务
> - **方法**: 共享 attention + 模态独立 FFN，AR 单向 condition DM；3D mRoPE 对齐时空；Qwen3-VL 初始化
> - **规模**: Nano 16B / Super 64B / Edge 4B；训练数据 1.3B 样本、393 数据集、9 种 embodiment
> - **结果**: 开源 SOTA on R-Bench / Physics-IQ / Artificial Analysis / RoboArena / VANTAGE-Bench
> - **代码**: https://github.com/nvidia/cosmos （OpenMDW-1.1）

---

*笔记创建时间: 2026-06-05*
