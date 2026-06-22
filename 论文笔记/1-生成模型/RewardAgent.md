---
title: "Reward as An Agent for Embodied World Models"
method_name: "RewardAgent + DynDiff-GRPO"
authors: [Pu Li, Zhigang Lin, Qiang Wu, Yongxuan Lv, Fei Wang, Shan You]
year: 2026
venue: arXiv
tags: [world-model, reinforcement-learning, reward-hacking, diffusion-policy, embodied-ai, grpo, video-generation]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.19990
created: 2026-06-22
---

# 论文笔记：Reward as An Agent for Embodied World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ACE Robotics |
| 日期 | June 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[Cosmos3]]、[[Kairos]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19990) |

---

## 一句话总结

> 提出 Reward Agent 多维 VLM 评估框架 + DynDiff-GRPO 动态感知探索，解决 Embodied World Model RL 后训练中奖励被 hack 与探索不足的核心矛盾。

---

## 核心贡献

1. **Reward Agent 框架**: 将奖励函数升级为具有规划、课程式评估、投票与反思能力的 VLM 智能体，从四个维度鲁棒评估视频生成质量，有效抵御奖励 hacking。
2. **DynDiff-GRPO**: 基于动态感知的差异化噪声分配策略，向动态活跃区域集中分配随机性，在保持物理约束的同时显著扩大轨迹探索空间。
3. **验证与探索协同扩展**: 实证表明当验证（reward）足够鲁棒时，更广泛的探索可以成功落地——在 PAI-Bench 上对 Cosmos-Predict2.5 和 Kairos-3.0-Robot 两个世界模型均取得显著提升。

---

## 问题背景

### 要解决的问题

如何对 [[Embodied AI|具身世界模型]] 做有效的 [[强化学习|RL]] 后训练？现有方法在将 RL 引入 [[Video Diffusion Model|视频扩散世界模型]] 时面临两大挑战：奖励函数不可靠（易被 hack）、探索空间过于保守。

### 现有方法的局限

- **奖励 hacking 普遍存在**：现有度量（VideoAlign、UnifiedReward 等）依赖代理信号，视频生成模型可以通过视觉遮蔽、运动退化、背景简化、物理违背等手段获得虚高分数而不完成真实任务。
- **探索过于保守**：以 Flow-GRPO、DanceGRPO、Mix-GRPO 等为代表的 [[GRPO]] 变体均以抑制 rollout 多样性为主要手段，依赖低温采样等保守策略，限制了 RL 的提升上限。
- **核心论点**：瓶颈不在探索本身，而在缺乏能支撑广泛探索的可靠验证策略（verification gap）。

### 本文的动机

类比代码执行能提供客观正确性反馈，本文希望为世界模型 RL 构建一个类似"客观验证器"的 Reward Agent，再搭配更激进的动态感知探索，从而实现验证与探索的协同扩展。

---

## 方法详解

### 整体框架

本文方法由两个互补模块组成：

- **Reward Agent**：可靠的多维评估框架（验证侧）
- **DynDiff-GRPO**：动态感知的 rollout 多样化策略（探索侧）

### 模块一：Reward Agent

**设计动机**: 静态标量奖励函数在分布偏移下容易被 [[Reward Hacking|奖励 hacking]]，需要具有推理与反思能力的 [[VLM]] 担任评估者。

**四个评估维度**（按课程式层级顺序）：

1. **Visual Quality（视觉质量）**: 帧一致性、清晰度、无伪影
2. **Instruction Following（指令跟随）**: 视频是否按照语言指令执行
3. **Task Completion（任务完成）**: 任务目标是否实际达成
4. **Physical Compliance（物理合规）**: 是否违背物理定律（穿透、悬浮等）

**五大集成组件**：

- **规划（Planning）**: 在详细评分前先对 rollout 质量做整体粗粒度分类
- **课程式评估（Curriculum Evaluation）**: 按 Visual Quality → Instruction Following → Task Completion → Physical Compliance 层级推进，每级需达到阈值才进入下一级
- **投票机制（Voting）**: 对每个维度内部进一步分解（如交互真实性、形变、穿透检测），多次采样取共识
- **反思（Reflection）**: 交叉校验 VLM 各评估的一致性，发现矛盾时自动修正
- **效率优化（Efficiency）**: 自适应帧采样 + 按维度分配视觉输入，降低计算开销

**四类奖励 hacking 模式**（已被 Reward Agent 识别）：

| 模式 | 描述 | 现有指标的盲区 |
|------|------|-------------|
| Visual Occlusion | 用模糊/阴影遮蔽缺陷区域 | VideoAlign VQ 打高分 |
| Motion Degradation | 通过近乎静止的运动规避错误 | UnifiedReward Alignment Score 高 |
| Background Simplification | 简化场景背景降低任务复杂度 | VideoAlign VQ 高 |
| Physical Invalidation | 违反物理定律但外观合理 | UnifiedReward Physics Score 较高 |

### 模块二：DynDiff-GRPO

**设计动机**: 在 [[Flow Matching|流匹配]] / [[扩散模型]] 的 SDE rollout 中，均匀随机性会破坏场景一致性；应当将随机性集中分配到动态活跃区域，从而在物理约束内最大化轨迹多样性。

**算法核心**: 构建动态感知掩码 $M_t$，替换标准 SDE 转移中的噪声尺度 $\sigma_{t^-}^{\text{noise}}$。

**具体实现步骤**：

1. 计算 clean sample 预测 $\hat{x}_0$ 的帧间残差，去除全局均值（空间归一化）得到动态敏感残差 $R^{(k)}$
2. 对残差的 $\ell_2$ 范数归一化，得到连续动态强度图 $D$
3. 通过百分位阈值 $\tau$ 对 $D$ 做二值化，得到稀疏动态先验 $B$
4. 融合 $D$ 和 $B$ 得到复合掩码 $M_t$，再与 $\sigma_{t^-}$ 逐元素相乘，实现特征维度各向异性的噪声分配

**训练目标**: 使用 [[GRPO]] 策略梯度 + 均值空间 KL 正则，保留完整高斯 likelihood（而非坍缩为标量方差），支持特征维度各向异性几何。

---

## 关键公式

### 公式 1：[[MDP|强化学习目标函数]]

$$
\text{MDP}=(\mathcal{S},\mathcal{A},P,R),\quad J(\pi)=\mathbb{E}_{\pi}\Big[\sum_{t=0}^{T}R(s_{t},a_{t})\Big]
$$

**含义**: 策略 $\pi$ 的目标是最大化期望累积奖励。

**符号说明**:
- $\mathcal{S}, \mathcal{A}$: 状态空间和动作空间
- $P$: 转移概率
- $R(s_t, a_t)$: 奖励函数
- $T$: 时间步长

### 公式 2：[[Reward Hacking|奖励 Hacking 形式化定义]]

$$
J(\pi_{\text{hack}}) > J(\pi_{\text{intended}}),\quad\text{while}\quad\text{TaskEval}(\pi_{\text{hack}}) \ll \text{TaskEval}(\pi_{\text{intended}})
$$

**含义**: Hacking 策略在代理奖励上得分更高，但真实任务评估分数远低于目标策略，定义了验证缺口（verification gap）的本质。

**符号说明**:
- $\pi_{\text{hack}}$: 利用奖励函数漏洞的策略
- $\pi_{\text{intended}}$: 真实目标策略
- $\text{TaskEval}(\cdot)$: 与代理奖励不同的真实任务评估

### 公式 3：[[Flow Matching|预测干净样本与噪声]]

$$
\hat{x}_{0}=x_{t}-\sigma_{t}v_{\theta},\quad\hat{\epsilon}_{t}=x_{t}+(1-\sigma_{t})v_{\theta}
$$

**含义**: 从当前带噪样本 $x_t$ 和速度预测 $v_\theta$ 还原干净样本与纯噪声，是 DynDiff-GRPO 动态掩码计算的基础。

**符号说明**:
- $v_\theta$: 神经网络预测的速度向量场
- $\sigma_t$: 时间步 $t$ 处的噪声水平
- $\hat{x}_0$: 预测的干净样本
- $\hat{\epsilon}_t$: 预测的纯高斯噪声

### 公式 4：[[扩散模型|混合 ODE/SDE 转移]]

$$
x_{t^{-}}=\hat{x}_{0}(1-\sigma_{t^{-}})+\hat{\epsilon}_{t}\sqrt{\sigma_{t^{-}}^{2}-(\sigma_{t^{-}}^{\text{noise}})^{2}}+\sigma_{t^{-}}^{\text{noise}}\odot z,\quad z\sim\mathcal{N}(0,I)
$$

**含义**: 在确定性 ODE 路径上注入受控噪声，形成 SDE rollout，其中 $\sigma_{t^-}^{\text{noise}}$ 控制随机程度。

**符号说明**:
- $\sigma_{t^-}^{\text{noise}}$: 注入噪声的尺度（DynDiff-GRPO 的核心修改量）
- $\odot$: 逐元素乘法
- $z \sim \mathcal{N}(0,I)$: 标准高斯随机噪声

### 公式 5：[[DynDiff-GRPO|动态感知噪声分配]]

$$
\sigma_{t^{-}}^{\text{noise}}=\sigma_{t^{-}}\odot M_{t}
$$

**含义**: 将标准噪声尺度按动态掩码 $M_t$ 逐元素缩放，实现向动态区域集中分配随机性。

**符号说明**:
- $M_t \in [r_{\text{base}}, 1]$: 动态感知掩码，静态区域接近 $r_{\text{base}}$，动态区域接近 1

### 公式 6：[[DynDiff-GRPO|动态敏感残差]]

$$
R^{(k)}=\Delta\hat{x}_{0}^{(k)}-\mathrm{Mean}_{h,w}\left(\Delta\hat{x}_{0}^{(k)}\right)
$$

**含义**: 计算相邻帧 clean sample 预测的差分，减去空间均值（去除全局运动），得到局部动态残差。

**符号说明**:
- $\Delta\hat{x}_0^{(k)}$: 第 $k$ 帧的 clean sample 预测与前帧之差
- $\mathrm{Mean}_{h,w}$: 在空间维度（高×宽）上取均值

### 公式 7：[[DynDiff-GRPO|连续动态强度图]]

$$
D=\mathrm{Normalize}\left(\|R\|_{2}\right)
$$

**含义**: 对残差的 $\ell_2$ 范数进行归一化，得到 $[0,1]$ 内的连续动态强度图。

**符号说明**:
- $\|R\|_2$: 残差的逐像素 $\ell_2$ 范数
- $\mathrm{Normalize}$: 全局最大值归一化到 $[0,1]$

### 公式 8：[[DynDiff-GRPO|稀疏动态先验]]

$$
B=\mathbb{I}\left[D>\mathrm{Quantile}(D,\tau)\right]
$$

**含义**: 对动态强度图按百分位阈值 $\tau$ 做二值化，提取稀疏的高动态区域掩码。

**符号说明**:
- $\tau$: 百分位阈值（超参数，通常取较高百分位如 0.7~0.9）
- $\mathbb{I}[\cdot]$: 指示函数

### 公式 9：[[DynDiff-GRPO|复合动态掩码]]

$$
M_{t}=r_{\text{base}}+(1-r_{\text{base}})\left(0.5D+0.5B\right)
$$

**含义**: 融合连续动态强度图 $D$ 和稀疏二值先验 $B$，保证所有区域都有最低噪声水平 $r_{\text{base}}$。

**符号说明**:
- $r_{\text{base}}$: 最小探索比例，确保低动态区域也有非零随机性
- $D$: 连续动态强度（覆盖渐变运动）
- $B$: 稀疏二值先验（覆盖局部突变）

### 公式 10：[[扩散模型|自适应方差分布]]

$$
x_{t^{-}}\sim\mathcal{N}\left(\mu,(\sigma_{t^{-}}\odot M_{t})^{2}\right)
$$

**含义**: 最终的 SDE 转移分布，方差在空间/特征维度各向异性，动态区域更大探索半径。

### 公式 12：[[Flow Matching|转移对数似然]]

$$
\log p(x_{t^{-}})=-\frac{(x_{t^{-}}-\mu)^{2}}{2(\sigma_{t^{-}}\odot M_{t})^{2}}-\log(\sigma_{t^{-}}\odot M_{t})-\log\sqrt{2\pi}
$$

**含义**: 用于计算 GRPO 中 importance ratio 的对数策略概率，保留完整高斯形式而非坍缩为标量。

### 公式 13：[[GRPO|截断组相对优势]]

$$
A_{i}=\mathrm{clip}\left(\frac{r_{i}-\mathrm{mean}(\{r_{j}\}_{j=1}^{G})}{\mathrm{std}_{\mathrm{dynamic}}(\{r_{j}\}_{j=1}^{G})+\epsilon},-A_{\max},A_{\max}\right)
$$

**含义**: 组相对归一化优势，截断阈值 $A_{\max}$ 从 5 降至 2.5（因为课程式投票聚合带来更高方差）。

**符号说明**:
- $r_i$: 第 $i$ 个 rollout 的奖励
- $G$: 组内 rollout 数量
- $A_{\max}$: 优势截断阈值（本文设为 2.5）
- $\mathrm{std}_{\mathrm{dynamic}}$: 动态标准差（实践中用指数移动均值平滑）

### 公式 16：[[GRPO|策略优化目标]]

$$
\mathcal{L}_{\mathrm{GRPO}}=-\mathbb{E}_{i,t}\left[\min\left(\rho_{i,t}A_{i},\,\mathrm{clip}(\rho_{i,t},1-\delta,1+\delta)A_{i}\right)\right]
$$

**含义**: 带重要性比率截断的 PPO 风格策略梯度损失，防止策略更新步长过大。

**符号说明**:
- $\rho_{i,t} = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$: 重要性比率
- $\delta$: 策略更新裁剪范围

### 公式 17：[[GRPO|均值空间 KL 正则]]

$$
\mathcal{L}_{\mu\text{-KL}}=\left\|\frac{\mu_{\theta}-\mu_{\mathrm{target}}}{\sigma_{\mathrm{target}}+\epsilon}\right\|_{2}^{2}
$$

**含义**: 在均值空间（而非对数概率空间）计算 KL 散度，实验表明比 log-prob 空间 KL 更有效。

### 公式 18：[[GRPO|完整训练目标]]

$$
\mathcal{L}=\mathcal{L}_{\mathrm{GRPO}}+\beta\mathcal{L}_{\mu\text{-KL}}
$$

**含义**: GRPO 策略梯度损失与均值空间 KL 正则项的加权组合，$\beta$ 为正则系数。

---

## 关键图表

### Figure 1: 典型奖励 Hacking 失败案例

![Figure 1](https://arxiv.org/html/2606.19990/2606.19990v1/x1.png)

**说明**: 四种 [[Reward Hacking]] 模式的典型例子：(a) 视觉遮蔽（VideoAlign VQ 0.79/1.0）；(b) 运动退化（UnifiedReward Alignment Score 2.76/5.0）；(c) 背景简化（VideoAlign VQ 0.80/1.0）；(d) 物理违背（UnifiedReward Physics Score 3.0/5.0）。生成视频获得虚高分数但严重违背真实任务目标。

### Figure 2: Reward Agent 框架概览

![Figure 2](https://arxiv.org/html/2606.19990/2606.19990v1/image/agent.png)

**说明**: Reward Agent 的整体流水线：规划 → 课程式奖励设计 → 投票机制 → 反思，集成为统一的奖励生成管道。四个评估维度（Visual Quality、Instruction Following、Physical Compliance、Task Completion）依次推进，前一维度不达阈值则后续维度不执行。

### Figure 3: Rollout 策略消融实验

![Figure 3](https://arxiv.org/html/2606.19990/2606.19990v1/x2.png)

**说明**: (a) 用 DINOv3 余弦距离衡量 rollout 多样性：DynDiff-GRPO 保持比 CPS 更高的轨迹多样性。(b) PAI-Bench robot domain 性能：DynDiff-GRPO（Domain 85.63，Overall 81.18）> CPS（84.03，80.31），质量分数（Quality）两者相当。

### Figure 4-6: 测试案例可视化

> 🖼️ **Figure 4**: 测试案例帧（原图见 [arXiv HTML](https://arxiv.org/html/2606.19990)）
> 🖼️ **Figure 5**: 测试案例帧（原图见 [arXiv HTML](https://arxiv.org/html/2606.19990)）
> 🖼️ **Figure 6**: 测试案例帧（原图见 [arXiv HTML](https://arxiv.org/html/2606.19990)）

**说明**: 三个典型测试案例的帧序列，展示 Reward Agent 对不同操作场景的评估效果。

### Table 1: PAI-Bench 机器人域性能对比

| 模型 | Domain Score | Overall Score |
|------|-------------|--------------|
| Cosmos-Predict2.5 (2B) | 83.61 | 79.42 |
| **DynDiff-GRPO + Cosmos (2B)** | **84.41 (+0.80)** | **80.75 (+1.33)** |
| Kairos-3.0-Robot (4B) | 83.87 | 80.35 |
| **DynDiff-GRPO + Kairos (4B)** | **86.95 (+3.08)** | **81.88 (+1.53)** |

**关键发现**: 两个开源世界模型均取得显著提升，Kairos 的 Domain Score 提升更大（+3.08），表明 DynDiff-GRPO 对更大模型增益更明显。

### Table 2: Reward 系统评估准确率（AgiBotWorld-Beta 100 测试用例）

| 评估维度 | 准确率 |
|---------|-------|
| Planning（规划） | 98.0% |
| Visual Quality（视觉质量） | 96.0% |
| Instruction Following（指令跟随） | 90.0% |
| Task Completion（任务完成） | 87.0% |
| Physical Compliance（物理合规） | 82.0% |
| **Overall（综合）** | **91.0%** |

**关键发现**: 总体准确率 91%，Physical Compliance 最具挑战性（82%），Physical Compliance 的难度反映了世界模型物理推理的内在困难。

### Table 3: CFG 消融（480×832 分辨率）

| 配置 | Domain Score | Overall Score |
|------|-------------|--------------|
| Without CFG | 85.07 | 80.71 |
| **With CFG** | **85.63** | **81.18** |

**关键发现**: [[Classifier-Free Guidance|CFG]] 在 rollout 阶段的引入对 Domain Score 有小幅提升。

### Table 4: 分辨率消融实验

| 训练分辨率 | 推理分辨率 | Quality Score | Domain Score | Overall Score |
|-----------|-----------|--------------|-------------|--------------|
| 480×832 + CFG | 480×832 | 76.74 | 85.63 | 81.18 |
| 640×820, No CFG | 640×820 | 77.24 | 85.60 | 81.42 |
| **640×820, No CFG** | **720×960** | **76.81** | **86.95** | **81.88** |

**关键发现**: 在较高分辨率训练后以更高分辨率推理（640→720）可以进一步提升 Domain Score，最终达到 86.95，接近 +3.08 的最大增益。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| AgiBotWorld-Beta | 机器人操作场景，用于 Reward Agent 验证 | 评估 Reward 系统准确率（100 测试用例） |
| HumanoidEveryday | 人形机器人日常操作 | 训练数据 |
| DROID | 多样化机器人操作 | 训练数据 |
| Kairos-3.0-Robot | 机器人视频生成数据集 | 模型训练 |
| PAI-Bench（robot 子集） | 具身 AI 基准评测，含 Domain/Quality/Overall Score | 评测指标 |

### 实现细节

- **基础模型**: Cosmos-Predict2.5（2B 参数）、Kairos-3.0-Robot（4B 参数）
- **参考指标**: Domain Score（任务域特定）、Quality Score（视觉质量）、Overall Score（综合）
- **优势截断**: $A_{\max} = 2.5$（相比标准 GRPO 的 5.0 更保守，因为课程式投票引入更高方差）
- **KL 正则**: 使用均值空间 KL（Eq.17）而非 log-prob 空间 KL，实验验证更有效
- **消融策略**: 与 CPS（Consecutive Prediction Sampling）作为主要对比基线

### 可视化结果

- DynDiff-GRPO 生成的轨迹在 DINOv3 embedding 空间中比 CPS 分布更分散（Figure 3a），证明多样性确实提升。
- Reward Agent 在 Planning 维度达 98% 准确率，说明整体粗粒度判断非常可靠；Physical Compliance 的 82% 准确率表明细粒度物理推理仍是挑战。

---

## 批判性思考

### 优点

1. **问题定义清晰**: 将 "验证缺口" 作为核心瓶颈的诊断很有洞见，区分了探索不足和验证不可靠两个独立问题。
2. **方法互补性强**: Reward Agent（验证侧）和 DynDiff-GRPO（探索侧）形成完整闭环，设计哲学自洽。
3. **工程细节充分**: 公式推导到 Eq.19 级别，各消融细粒度（CFG、分辨率、KL 空间），实验可信度高。

### 局限性

1. **评估规模有限**: PAI-Bench 实验仅限机器人子集，Reward Agent 验证仅 100 测试用例，泛化性需更大规模验证。
2. **Reward Agent 的 VLM 成本**: 课程式 + 投票 + 反思的多轮 VLM 调用成本未详细量化，大规模训练时的推理开销可能是瓶颈。
3. **Physical Compliance 仍是弱项**: 82% 的准确率意味着约 1/5 的物理判断有误，在安全敏感的机器人场景中风险较高。
4. **消融不完整**: 论文未系统比较 Reward Agent 单独的贡献 vs DynDiff-GRPO 单独的贡献（两者联合消融）。

### 潜在改进方向

1. 将 Reward Agent 的 VLM 推理蒸馏为更轻量的判别模型，降低训练推理成本。
2. 扩展到非机器人的具身 AI 场景（自动驾驶、游戏世界模型等）。
3. 探索动态掩码 $M_t$ 的在线自适应（随训练进程动态调整 $r_{\text{base}}$ 和 $\tau$）。

### 可复现性评估

- [ ] 代码开源（未提供代码链接）
- [ ] 预训练模型（依赖 Cosmos-Predict2.5 和 Kairos-3.0-Robot，均需独立获取）
- [x] 训练细节完整（公式推导和超参数较完整）
- [x] 数据集可获取（AgiBotWorld-Beta、DROID 已公开）

---

## 关联笔记

### 基于

- [[GRPO]]: 本文 DynDiff-GRPO 的核心优化算法基础
- [[Flow Matching]]: 扩散/流匹配采样框架，DynDiff-GRPO 在此基础上注入动态感知噪声
- [[Video Diffusion Model]]: 底层世界模型架构

### 对比

- [[Kairos]]: 用于 RL 后训练的基础世界模型（4B）
- [[Cosmos3]]: Cosmos-Predict2.5（2B），另一个实验基础模型
- [[CM-GRPO]]: 同类 GRPO 变体，主要用于视频生成

### 方法相关

- [[Reward Hacking]]: 本文核心解决的问题
- [[Classifier-Free Guidance]]: CFG 在 rollout 阶段的应用
- [[Model-Based RL]]: 本文属于 MBRL 的世界模型后训练场景
- [[Curriculum Learning with Extended Foresight]]: 与本文课程式评估思路相关

### 硬件/数据相关

- [[AgiBotWorld-Beta]]: 用于 Reward Agent 评估的机器人数据集
- [[PAI-Bench]]: 具身 AI 评测基准

---

## 速查卡片

> [!summary] Reward as An Agent for Embodied World Models
> - **核心**: 用 VLM Reward Agent 解决验证缺口，配合 DynDiff-GRPO 扩大探索
> - **方法**: 四维课程式 Reward Agent + 动态感知噪声掩码 $M_t$
> - **结果**: PAI-Bench 上 Cosmos +1.33、Kairos +1.53 Overall；Reward Agent 91% 准确率
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-22*
