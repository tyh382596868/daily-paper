---
title: "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"
method_name: "JEPA-WM"
authors: [Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, Yann LeCun]
year: 2025
venue: arXiv
tags: [world-model, jepa, robot-planning, model-based-rl, ablation-study, latent-prediction]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2512.24497v3
created: 2026-05-19
---

# 论文笔记：What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Meta FAIR, Inria Paris, ENS/PSL, NYU |
| 日期 | Dec 2025 |
| 项目主页 | https://github.com/facebookresearch/jepa-wms |
| 对比基线 | [[DINO-WM]] · V-JEPA-2-AC |
| 链接 | [arXiv](https://arxiv.org/abs/2512.24497) / [Code](https://github.com/facebookresearch/jepa-wms) |

---

## 一句话总结

> 对 [[JEPA]] 世界模型在物理规划任务上的设计空间进行系统消融，给出"何种设计真正决定成败"的 8 条经验答案。

---

## 核心贡献

1. **系统化设计空间消融**: 横跨 8 个维度（编码器、预测器结构、动作条件化、上下文长度、多步 rollout、本体感知、规划优化器、模型/数据规模）在模拟与真实任务上做穷尽对照实验。
2. **最优配方明确化**: 提出一条在 [[DROID]]、[[RoboCasa]]、[[Push-T]]、Maze、Metaworld 等 8 个 benchmark 上**全面超越 [[DINO-WM]] 与 V-JEPA-2-AC** 的配置（如 [[DINOv3]]-L + AdaLN+RoPE + 6-step rollout + CEM-L₂）。
3. **可操作的工程指南**: 区分"模拟任务"与"真实数据"的最佳超参，澄清了 [[DINO]] 类编码器优于 [[V-JEPA]] 类视频编码器、本体感知输入持续有益、AdaLN 条件化最稳健等关键经验。

---

## 问题背景

### 要解决的问题
[[潜空间世界模型|JEPA 世界模型]]（[[JEPA-WM]]）作为一类"在表征空间内做规划"的世界模型，近年涌现了 [[DINO-WM]]、V-JEPA-2-AC、[[PLDM]] 等多种实现，但**到底是什么设计决定了它们的成功**——是编码器？预测器结构？训练目标？规划算法？——缺乏系统答案。

### 现有方法的局限
- 各论文设计选择互相冲突（[[DINO-WM]] 用 feature conditioning + sincos；V-JEPA-2-AC 用 sequence conditioning + [[RoPE]]），无对照实验。
- 缺乏跨模拟/真实任务的统一评测，难以判断哪些经验可迁移。
- 多步 rollout、上下文长度、模型规模等"次级超参"的作用不清晰。

### 本文的动机
通过控制变量法在统一代码库上重新实现并对比所有主流设计，给出"What drives success"的可操作答案。

---

## 方法详解

### 模型架构

[[JEPA-WM]] 采用 **frozen encoder + 可训练 predictor** 的 [[潜空间世界模型]] 架构（不重建像素、不预测奖励）：

- **输入**: 语言/任务通过目标观测 $o_g$ 隐式给出；当前观测 $o_{t-w:t}$ + 动作序列 $a_{t-w:t}$ +（可选）本体感知 $s_t$
- **冻结视觉编码器** $E_\phi$: [[DINOv2]] / [[DINOv3]] / [[V-JEPA]] 任一，输出 token 序列嵌入
- **可训练动作编码器** $A_\theta$ 与可选**本体感知编码器** $E_\theta^{\text{prop}}$
- **预测器** $P_\theta$: ViT-style Transformer，深度 3-12 层，三种动作条件化方式可选（feature / sequence / [[AdaLN]]）
- **训练目标**: 仅在嵌入空间做 [[MSE]] 预测，没有重建头、价值头或策略头
- **规划**: 用 [[CEM]] / [[Nevergrad]] / Adam / GD 在动作空间搜索使预测终态嵌入接近目标嵌入的动作序列（即 [[MPC]]）

### 核心模块

#### 模块1: 视觉编码器（冻结）

**设计动机**: 利用大规模自监督 [[预训练]] 的视觉表征作为状态空间，避免端到端训练带来的 [[表征坍塌]] 问题。

**具体实现**:
- 候选: [[DINOv2]] (S/B/L)、[[DINOv3]]-L、[[V-JEPA]]-L、V-JEPA-2-L
- 视频编码器（[[V-JEPA]] 系）通过将单帧复制为 2 帧视频以独立编码
- **关键发现**: [[DINO]] 系编码器（dense 图像 [[预训练]]）在物理预测任务上显著优于 [[V-JEPA]] 系视频编码器——因为 [[DINO]] 给出**细粒度物体分割**式 token，物体运动转化为局部稀疏的 token 变化，更易被预测器学习。

#### 模块2: 预测器与动作条件化

**设计动机**: 让动作信息能"贯穿网络深度"地影响预测，是 JEPA-WM 的核心瓶颈。

**具体实现** 四种方案：
- **Feature conditioning + sincos**（[[DINO-WM]] 风格）: 动作线性投影后沿 embedding 维拼接，加 sincos 位置编码
- **Sequence conditioning + [[RoPE]]**（V-JEPA-2 风格）: 动作作为独立 token 沿序列维拼接
- **Feature conditioning + [[RoPE]]**: 混合方案
- **[[AdaLN]] + [[RoPE]]** ✅: 每个 Transformer block 内用动作特征做 [[AdaLN]] 缩放/平移——防止动作信息在网络深处衰减

#### 模块3: 多步 Rollout 训练

**设计动机**: 单步 [[Teacher Forcing]] 训练在自回归推理时会因 [[Compounding Errors|误差累积]] 而失败。

**具体实现**:
- 训练时累积 K 步预测损失（K=1,2,3,6）
- 模拟任务最佳 K=2（轻度数据增广），真实 [[DROID]] 数据最佳 K=6
- K 越大需要更大的训练上下文 W（最大 W=7）
- 理论上引入"准确率-鲁棒性 trade-off"：K 越大单步误差 $\delta_K$ 越大但传播常数 $\Lambda_K$ 越小

#### 模块4: 规划优化器

**设计动机**: 在嵌入空间执行 [[MPC]]，需要搜索使预测终态接近目标的动作序列。

**具体实现**: 四种优化器：
- **[[CEM]] + L₂** ✅: 总体最稳健
- **[[Nevergrad]] (NG)**: 零超参，在真实数据（[[DROID]]/[[RoboCasa]]）上与 [[CEM]] 持平
- **Adam**: 仅在 Metaworld 等"平滑代价面"任务胜出
- **GD**: 在导航/接触任务上因 local minima 失败

---

## 关键公式

### 公式1: [[JEPA-WM|JEPA 世界模型训练损失]]

$$
\mathcal{L} = \frac{1}{B}\sum_{b=1}^{B} L\Big[\,P_\theta\big(E_{\phi,\theta}(o_{t-w:t}),\ A_\theta(a_{t-w:t})\big),\ E_{\phi,\theta}(o_{t+1})\,\Big]
$$

**含义**: 在批 $B$ 个样本上，预测器 $P_\theta$ 接收过去 $w$ 帧的状态嵌入与动作嵌入，输出下一帧状态嵌入；用距离 $L$（默认 [[MSE]]）与真实下一帧编码对齐。**完全不重建像素**。

**符号说明**:
- $o_{t-w:t}$: 过去 $w$ 帧观测（含可选本体感知）
- $E_{\phi,\theta}$: 冻结视觉编码器 + 可训练本体感知编码器
- $A_\theta$: 可训练动作编码器
- $P_\theta$: 可训练预测器
- $L$: 距离度量（[[MSE]] 等）

### 公式2: [[MPC|规划代价]]

$$
\mathcal{L}^p_\alpha = \big(L_{\text{vis}} + \alpha L_{\text{prop}}\big)\Big(G_{\phi,\theta}(o_t, a_{t:t+H-1}),\ E_{\phi,\theta}(o_g)\Big)
$$

**含义**: 把候选动作序列 $a_{t:t+H-1}$ 通过预测器展开 $H$ 步得到预测终态嵌入 $G_{\phi,\theta}(\cdot)$，与目标观测嵌入比较；视觉与本体感知项分别用 $L_{\text{vis}}, L_{\text{prop}}$ 度量并以 $\alpha$ 加权。

**符号说明**:
- $G_{\phi,\theta}$: 由 $P_\theta$ 自回归 unroll 得到的前向动力学
- $o_g$: 目标观测
- $\alpha$: 本体感知项权重
- $H$: 规划 horizon

### 公式3: [[前向动力学]]（自回归展开）

$$
\hat{z}_{t+1} = P_\theta\big(E_{\phi,\theta}(o_{t-w+1:t}),\ A_\theta(a_{t-w+1:t})\big)
$$

$$
\hat{z}_{t+k+1} = P_\theta\big([\,E_{\phi,\theta}(o_{t-w+k+1:t}),\ \hat{z}_{t+1:t+k}\,],\ A_\theta(a_{t-w+k+1:t+k})\big),\quad k\ge 1
$$

**含义**: 第 1 步预测用真实编码；之后步用滑动窗口 $w$，将之前预测的嵌入 $\hat{z}$ 与早期真实嵌入拼接作为上下文继续预测。

**符号说明**:
- $\hat{z}_{t+k}$: 第 $t+k$ 步的**预测**状态嵌入
- $w$: 训练上下文窗口大小

### 公式4: [[Compounding Errors|误差传播]]（Appendix D）

$$
\|\hat{z}_{t+H} - z_{t+H}\|\ \le\ \sum_{k=1}^{H} \Lambda_K^{H-k}\,\delta_K
$$

**含义**: 自回归 $H$ 步后总误差被 [[Lipschitz 常数]] $\Lambda_K$ 的 $(H-k)$ 次方放大；K-step 训练同时减小 $\Lambda_K$ 但增大单步 $\delta_K$，构成根本的准确率-鲁棒性 trade-off。

**符号说明**:
- $\Lambda_K$: K-step 训练下预测器的有效 [[Lipschitz 常数]]
- $\delta_K$: K-step 训练下的单步预测误差上界
- $H$: 推理 horizon

---

## 关键图表

### Figure 1: JEPA-WM 训练与规划总览

![Figure 1](https://arxiv.org/html/2512.24497v3/x1.png)

**说明**: 左：训练阶段，编码器嵌入视频与（可选）本体感知，与动作一起喂给预测器，并行预测下一帧状态嵌入。右：规划阶段——采样动作序列 → 用预测器 unroll → 计算每条轨迹的规划代价 $\mathcal{L}^p$ → 用 [[CEM]] 等优化器迭代精炼动作分布。整张图是 [[MPC]] 在 [[潜空间世界模型]] 上的标准范式。

### Figure 2: Counterfactual Franka 杯子任务（定性对比）

![Figure 2 row 1](https://arxiv.org/html/2512.24497v3/x2.png)
![Figure 2 row 2](https://arxiv.org/html/2512.24497v3/x3.png)
![Figure 2 row 3](https://arxiv.org/html/2512.24497v3/x4.png)
![Figure 2 row 4](https://arxiv.org/html/2512.24497v3/x5.png)
![Figure 2 row 5](https://arxiv.org/html/2512.24497v3/x6.png)
![Figure 2 row 6](https://arxiv.org/html/2512.24497v3/x7.png)

**说明**: 在 Franka 真机数据上硬编码两组反事实动作（"张开+上移" vs "闭合+上移"），各方法做 5 步开环 rollout。V-JEPA-2-AC（首行）、[[DINO-WM]]（中行）、本文最优模型（末行）的预测帧解码可视化——本文方法在物体交互建模上明显更准确。

### Figure 3: 规划优化器 & 多步 Rollout 影响

![Figure 3a](https://arxiv.org/html/2512.24497v3/x8.png)
![Figure 3b](https://arxiv.org/html/2512.24497v3/x9.png)

**说明**: (a) 对比 [[CEM]]、[[Nevergrad]] (NG)、Adam、GD，分别配 L₁/L₂ 距离——**[[CEM]]+L₂ 总体最稳，Adam 仅在 Metaworld 胜出，GD 在导航/接触任务失败**。(b) 累积 K-step 损失的效果——模拟任务最佳 K=2，[[DROID]] 真机最佳 K=6。

### Figure 4: 本体感知输入 & 视觉编码器对比

![Figure 4a](https://arxiv.org/html/2512.24497v3/x10.png)
![Figure 4b](https://arxiv.org/html/2512.24497v3/x11.png)

**说明**: (a) 带本体感知输入（prop）的模型一致优于纯视觉模型（no-prop），尤其在最后阶段视觉位移微弱时帮助大。(b) 同样 ViT-L 规模下，[[DINO]] 系（[[DINOv2]]/[[DINOv3]]）显著优于 [[V-JEPA]]/V-JEPA-2；[[DINOv3]]-L 在真实数据（[[DROID]]/[[RoboCasa]]）最佳，[[DINOv2]]-S 在合成任务上已足够。

### Figure 5: 预测器结构 & 训练上下文长度

![Figure 5a](https://arxiv.org/html/2512.24497v3/x12.png)
![Figure 5b](https://arxiv.org/html/2512.24497v3/x13.png)

**说明**: (a) 四种动作条件化方式对比，**[[AdaLN]]+[[RoPE]] 平均最佳**，sincos+feature cond 在 Metaworld 略好（任务依赖）。(b) 训练上下文 $W$ 影响——W=1 严重欠缺（无法推断速度），W=2 跳变明显（速度可推），W=3 可推加速度；模拟任务 W=3，[[DROID]] 任务 W=5 最优；W 过大反而因独立训练切片变少而下降。

### Figure 6: 模型规模消融

![Figure 6a](https://arxiv.org/html/2512.24497v3/x14.png)
![Figure 6b](https://arxiv.org/html/2512.24497v3/x15.png)

**说明**: (a) 视觉编码器从 ViT-S→ViT-B→ViT-L 的影响——模拟任务**没有 scaling 收益**（任务过简单 / 高维嵌入空间反而让规划更难），[[DROID]] 真实数据上**有清晰正相关**。(b) 预测器深度 3→6→9→12——模拟最佳 depth=6，[[DROID]] 最佳 depth=12。结论：scale 应**按环境复杂度自适应**而非盲目放大。

### Figure 7: 数据规模消融

![Figure 7](https://arxiv.org/html/2512.24497v3/x16.png)

**说明**: 训练数据从 2% → 100% 时所有方法都持续受益，本文最优配置在 [[DROID]] 与 Wall 任务上数据效率最高。

### Table 1: 系统消融的设计空间总览

| 组件 | 候选选项 | 模拟导航/操作最佳 | 真实操作最佳 |
|------|----------|------------------|--------------|
| 视觉编码器 | [[DINOv2]] (S/B/L), [[DINOv3]]-L, [[V-JEPA]]-L, V-JEPA-2-L | [[DINOv2]]-S | [[DINOv3]]-L |
| 预测器结构 | Feat+sincos, Seq+RoPE, Feat+RoPE, AdaLN+RoPE | **AdaLN+[[RoPE]]** | **AdaLN+[[RoPE]]** |
| 预测器深度 | 3, 6, 9, 12 | 6 | 12 |
| Rollout 步数 K | 1 (TF), 2, 3, 6 | 2 | 6 |
| 上下文 W | 1, 2, 3, 5, 7, 9, 14 | 3 | 5 |
| 本体感知 | with / without | with | with |
| 规划优化器 | [[CEM]], [[Nevergrad]], Adam, GD | [[CEM]] | [[CEM]] |
| 代价度量 | L₁ / L₂ | L₂ | L₂ |

**说明**: 跨模拟与真实任务的最优设计高度一致，仅在编码器大小、深度、rollout 步数上**按复杂度放大**。

### Table 2: 主结果对比（[[任务成功率]] %, std）

| 任务 | [[DINO-WM]] | V-JEPA-2-AC | **本文最优** |
|------|-------------|-------------|--------------|
| Maze | 81.6 (3.4) | — | **83.9 (2.3)** |
| Wall | 64.1 (4.6) | — | **78.8 (3.9)** |
| [[Push-T]] | 66.0 (4.7) | — | **70.2 (2.8)** |
| [[Metaworld]]-Reach | 44.8 (8.9) | — | **58.2 (9.3)** |
| [[Metaworld]]-Reach-Wall | 35.1 (9.4) | — | **41.6 (10.0)** |
| [[RoboCasa]]-Reach | 19.1 (13.4) | 16.2 (8.3) | **25.4 (16.6)** |
| [[RoboCasa]]-Place | 21.7 (7.2) | **33.1 (7.2)** | 30.7 (8.0) |
| [[DROID]] | 39.4 (2.1) | 42.9 (2.5) | **48.2 (1.8)** |

**关键发现**: 本文方法在 **8 个 benchmark 的 7 个**上超越两个 baseline，仅 [[RoboCasa]]-Place 略输给 V-JEPA-2-AC；在最难的真实 [[DROID]] 上提升 +5.3 pp / +8.8 pp。

---

## 实验

### 数据集

| 数据集 | 类型 | 任务 | 用途 |
|--------|------|------|------|
| Maze, Wall | 模拟 2D 导航 | 点到点导航 | 训练 + 评测 |
| [[Push-T]] | 模拟操作 | T 形物体推动 | 训练 + 评测 |
| [[Metaworld]]-MW-R/MW-RW | 模拟操作 | Reach / Reach-Wall | 训练 + 评测 |
| [[DROID]] | 真实操作 | ~1M Franka 抓取轨迹 | 训练 + 评测 |
| [[RoboCasa]] | 仿真操作 | Reach / Place | 零样本迁移（用 [[DROID]] 模型评测） |
| Custom Franka 视频 | 真实操作 | 16 个 DROID-like 视频 | counterfactual 定性评测 |

### 实现细节

- **编码器**: 全程冻结（[[DINOv2]]/[[DINOv3]]/[[V-JEPA]] 任一）
- **预测器**: ViT-style Transformer，深度 3-12，[[AdaLN]] + [[RoPE]] 默认
- **训练**: [[MSE]] 损失，AdamW，K-step rollout（K=1-6）
- **规划**: [[CEM]] + L₂，horizon $H$ 与上下文 $W$ 任务依赖
- **评测**: 每 epoch 跑 96 episodes（[[DROID]] 64、[[RoboCasa]] 32），最终模型用 3 seed 报告
- **硬件**: Meta FAIR 集群，详见 Appendix C

### 可视化结果

Counterfactual Franka 实验（Figure 2）展示本文方法在反事实动作（"张开/闭合 + 上移"）下能更准确地预测物体行为：[[DINO-WM]] 对"闭合"动作的物体抓取建模失真，V-JEPA-2-AC 对小物体细节预测模糊，本文方法在抓取与升降两阶段都给出可读的视觉解码。

---

## 批判性思考

### 优点
1. **少有的高质量系统消融**：横跨 8 个设计维度 × 8 个 benchmark × 多 seed，是 JEPA-WM 领域目前最完整的对照实验。
2. **结论可操作**：不止"哪种配置最好"，还给出"模拟 vs 真实任务最优配置不同"的清晰分界，对从业者直接有用。
3. **理论与经验结合**：Appendix D 用 [[Lipschitz 常数]] 形式化推导了多步 rollout 的准确率-鲁棒性 trade-off，不只是堆实验。

### 局限性
1. **确定性预测器**：MSE 训练学到的是多模态未来的条件均值，无法捕捉 aleatoric 不确定性，作者也承认这点。
2. **依赖确定性环境/闭环 MPC**：开环 rollout 在随机环境下会失效，文中实验都依赖确定性物理或频繁重规划。
3. **零样本迁移有限**：[[DROID]] → [[RoboCasa]] 的迁移受 domain gap 影响显著，本体感知输入无法跨平台迁移。
4. **范围限制**：只研究"用预训练表征 + 帧级 MSE"这一类 JEPA-WM；与 [[DreamerV3]]、[[TD-MPC2]] 等**带 reward/value head 的 model-based RL** 的对比缺席。

### 潜在改进方向
1. **潜变量注入** → 随机 JEPA-WM，建模多模态未来
2. **Diffusion latent predictor** → 在嵌入空间做扩散去噪（与 [[VLA-JEPA]]、[[Drive-JEPA]] 等思路结合）
3. **轻量编码器适配**（concurrent 工作 Toso et al., 2026 已在探索 bisimulation 微调）
4. **稀疏特征分解**（concurrent Zhao et al. / Yin et al. 2026）：将 dynamics-relevant 与 -irrelevant 特征解耦

### 可复现性评估
- [x] 代码开源（[github.com/facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)）
- [x] 预训练模型（checkpoints 开源）
- [x] 训练细节完整（Appendix C）
- [x] 数据集可获取（[[DROID]]、[[RoboCasa]]、[[Push-T]]、[[Metaworld]] 全公开）

---

## 关联笔记

### 基于
- [[JEPA]]: 核心范式——预测在嵌入空间而非像素空间
- [[I-JEPA]] / [[V-JEPA]]: JEPA 的视觉/视频实例
- [[DINOv2]] / [[DINOv3]]: 提供冻结视觉编码器

### 对比
- [[DINO-WM]]: 本文主要 baseline 之一（feature cond + sincos）
- V-JEPA-2-AC: 本文主要 baseline 之一（sequence cond + RoPE）
- [[PLDM]]: 早期潜空间 planning 工作
- [[DreamerV3]]、[[TD-MPC2]]: 带 reward head 的 model-based RL（未直接对比）

### 方法相关
- [[MPC]] / [[CEM]]: 规划骨架
- [[AdaLN]]: 最佳动作条件化方式
- [[RoPE]]: 位置编码
- [[Compounding Errors]]: 多步 rollout 训练对应的根本问题
- [[Lipschitz 常数]]: 误差传播分析的数学工具

### 硬件/数据相关
- [[DROID]]: 主真实数据集
- [[RoboCasa]]: 零样本迁移评测
- [[Push-T]] / [[Metaworld]]: 模拟操作 benchmark

---

## 速查卡片

> [!summary] What Drives Success in JEPA World Models
> - **核心**: 系统消融 8 维设计空间，给出 JEPA-WM 成功配方（[[DINO]]+AdaLN+[[CEM]] L₂+多步 rollout+本体感知）
> - **方法**: 冻结 [[DINOv3]]-L 编码器 + AdaLN+[[RoPE]] 预测器 + [[MSE]] 多步训练 + [[CEM]] 规划
> - **结果**: 8 个 benchmark 中 7 个超过 [[DINO-WM]] 与 V-JEPA-2-AC，[[DROID]] 上 48.2% vs 42.9%
> - **代码**: https://github.com/facebookresearch/jepa-wms

---

*笔记创建时间: 2026-05-19*
