---
title: "ROAD-VLA: Robust Online Adaptation via Self-Distillation for Vision-Language-Action Models"
method_name: "ROAD-VLA"
authors: [Kejing Wang, Toan Nguyen, Minh Hoang Nguyen, Simon Khan, Flora D. Salim]
year: 2026
venue: arXiv
tags: [vla, online-adaptation, reinforcement-learning, knowledge-distillation, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.25800
created: 2026-06-25
---

# 论文笔记：ROAD-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UNSW Sydney（推测，作者信息未完整披露）|
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[PPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25800) |

---

## 一句话总结

> ROAD-VLA 用「优势引导自蒸馏」替代文本特权信息，将稀疏奖励转化为逐 token 的密集监督，使 VLA 模型在分布偏移下高效在线适应。

---

## 核心贡献

1. **揭示文本特权信息失效**: 实验证明文本型教师（MCTS 提示、空间描述、高层规划）在 VLA 在线适应中完全失效（仅 ~4.7% 成功率），根因是符号指导与低层机器人动作之间的模态鸿沟。
2. **优势引导自蒸馏（Advantage-Guided Distillation）**: 提出通过校准后的优势估计扰动 action-token logits，构造「近端教师分布」，将稀疏奖励转化为 token 级密集 KL 蒸馏目标。
3. **理论保证与校准混合机制**: 在温和校准条件下推导出策略改进下界（Theorem 1），并引入在线/冻结双 critic 的协议门（agreement gate）防止冲突梯度。

---

## 问题背景

### 要解决的问题

[[VLA]] 经过大规模预训练后，面对新环境（视觉变化、目标组合变化、执行状态变化）时性能下降。在线强化学习（RL）是适应新环境的自然选择，但稀疏奖励导致学习效率低下。

### 现有方法的局限

- **标准 [[PPO]]**：直接对 VLA 做在线 RL，稀疏奖励信号弱，收敛慢、性能提升有限。
- **文本特权教师（Privileged Information）**：借助 LLM 生成 MCTS 动作提示、空间关系描述、高层规划等文本信号，但实验证明这些完全失效（~4.7% 成功率）。原因：VLA 经过动作后训练后，已丧失对通用推理文本的理解能力，模态鸿沟使符号信号无法转化为有效动作监督。
- **同策略（on-policy）[[知识蒸馏]]**：naive 用 JSD 蒸馏损失缺乏优势加权，负样本轨迹也会产生蒸馏目标，污染梯度。

### 本文的动机

自蒸馏（self-distillation）在同一模态内操作，天然规避了模态鸿沟。通过将优势估计注入 teacher 分布构造，可以在不引入任何外部教师的情况下，将稀疏标量奖励转化为逐 token 的密集分布监督——好轨迹被强化，差轨迹被抑制。

---

## 方法详解

### 整体框架

ROAD-VLA 以 [[OpenVLA]] 为基础，在线收集稀疏奖励轨迹，通过 **优势引导自蒸馏（Advantage-Guided Distillation, AGD）** 生成近端教师分布，再用前向 KL 蒸馏更新策略参数。

**输入**: 视觉观测 $o_{t-H+1:t}$（图像帧历史）+ 语言指令 $l$  
**输出**: 动作 token 序列 $a_t = (a_{t,1}, \ldots, a_{t,K})$，$K=7$  
**基础模型**: OpenVLA-7B（7B 参数 VLA）

### 核心模块

#### 模块 1：监督微调预热（SFT Warm-up）

在进入在线 RL 之前，先用 140 条专家轨迹对 OpenVLA 做 SFT，建立一个具备基础任务能力的检查点，供 ROAD-VLA 和 PPO baseline 共同出发。

#### 模块 2：校准优势混合（Calibrated Advantage Mixing）

VLA 在线训练时同时维护两个 critic：
- **在线 critic**（参数随策略更新）：计算 $\hat{A}_t^{int}$（GAE 优势）
- **冻结参考 critic**（不更新）：计算 $\hat{A}_t^{ref}$，与在线 critic 独立，提供更稳定的参考信号

两者之间存在量纲差异，需先归一化：

$$
\tilde{A}_t^{ref} = \mu_{int} + \frac{\sigma_{int}}{\sigma_{ref} + \varepsilon}(\hat{A}_t^{ref} - \mu_{ref})
$$

然后通过**协议门（agreement gate）** $g_t$ 有条件混合：

$$
g_t = \mathbb{1}[\mathrm{sign}(\hat{A}_t^{int}) = \mathrm{sign}(\tilde{A}_t^{ref})]
$$

$$
\hat{A}_t^{mix} = \hat{A}_t^{int} + \alpha \cdot g_t \left(\tilde{A}_t^{ref} - \hat{A}_t^{int}\right)
$$

只有当两个 critic 对优势符号达成共识时，才将参考 critic 的信号混入，防止冲突蒸馏梯度（conflicting distillation gradients）。

#### 模块 3：近端教师分布构造（Proximal Teacher Construction）

混合后的优势经过裁剪归一化，得到 token 级扰动权重：

$$
\omega_t = \mathrm{clip}\!\left(\frac{\hat{A}_t^{mix} - \mu_{mix}}{\sigma_{mix} + \varepsilon},\ -c,\ c\right)
$$

然后对 VLA 在该 token 位置的原始 logits $z_{t,k}$ 施加指数倾斜（exponential tilt），构造近端教师分布：

$$
q^\star_{t,k} = \mathrm{softmax}\!\left(z_{t,k} + \eta \cdot \omega_t \cdot e_{\hat{a}_{t,k}}\right)
$$

其中 $e_{\hat{a}_{t,k}}$ 是实际采样动作 token $\hat{a}_{t,k}$ 对应的 one-hot 向量，$\eta=1.0$ 为扰动强度。

该教师分布实际上是以下 KL 正则化局部改进问题的闭形式解：

$$
q^\star_{t,k} = \mathop{\mathrm{arg\,max}}_{q \in \Delta(\mathcal{V}_k)} \left\{ \mathbb{E}_{u \sim q}\left[r_{t,k}(u)\right] - \tau\, \mathrm{KL}(q \,\|\, p^\theta_{t,k}(\cdot|h_t)) \right\}
$$

闭形式解为指数倾斜：

$$
q^\star_{t,k}(u) \propto p^\theta_{t,k}(u) \cdot \exp\!\left(\frac{r_{t,k}(u)}{\tau}\right)
$$

#### 模块 4：优势引导蒸馏目标（AGD Loss）

对收集到的轨迹 $\hat{\tau}$，用前向 [[KL 散度]] 蒸馏教师分布：

$$
\mathcal{L}_{AGD} = \mathbb{E}_{\hat{\tau}}\!\left[\frac{1}{TK} \sum_{t=1}^T \sum_{k=1}^K \mathrm{KL}(q^\star_{t,k} \,\|\, p^\theta_{t,k})\right]
$$

前向 KL（mean-seeking）使学生策略的质量好于使用 JSD，因为 JSD 对正负轨迹一视同仁，而前向 KL 配合优势加权的教师分布可以自然抑制负轨迹的蒸馏信号。

---

## 关键公式

### 公式 1：[[SFT 损失|监督微调损失]]

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(o_{t-H+1:t},\, l,\, a_t) \sim \mathcal{D}^{tar}}\!\left[\log \pi_\theta(a_t | o_{t-H+1:t}, l)\right]
$$

**含义**: 在目标数据集 $\mathcal{D}^{tar}$ 上做最大似然估计，为 VLA 提供任务基础能力。

**符号说明**:
- $o_{t-H+1:t}$: 时间窗口 $H$ 内的图像观测历史
- $l$: 语言指令
- $a_t$: 专家动作（离散化为 $K=7$ 个 token）

### 公式 2：[[PPO 目标|PPO 裁剪目标]]

$$
\mathcal{L}_{PPO}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]
$$

**含义**: 通过裁剪概率比防止策略更新步骤过大的 on-policy RL 目标。

**符号说明**:
- $r_t(\theta) = \pi_\theta(a_t|\cdot) / \pi_{\theta_0}(a_t|\cdot)$: 策略概率比
- $\hat{A}_t$: GAE 优势估计
- $\varepsilon$: 裁剪范围

### 公式 3：[[校准优势归一化|参考 Critic 归一化]]

$$
\tilde{A}_t^{ref} = \mu_{int} + \frac{\sigma_{int}}{\sigma_{ref} + \varepsilon}(\hat{A}_t^{ref} - \mu_{ref})
$$

**含义**: 将冻结参考 critic 的优势归一化到在线 critic 的量纲，消除两个 critic 输出尺度不一致。

**符号说明**:
- $\mu_{int}, \sigma_{int}$: 在线 critic 优势的均值和标准差
- $\mu_{ref}, \sigma_{ref}$: 参考 critic 优势的均值和标准差
- $\varepsilon$: 数值稳定项

### 公式 4：[[协议门|Agreement Gate（协议门）]]

$$
g_t = \mathbb{1}[\mathrm{sign}(\hat{A}_t^{int}) = \mathrm{sign}(\tilde{A}_t^{ref})]
$$

$$
\hat{A}_t^{mix} = \hat{A}_t^{int} + \alpha \cdot g_t \left(\tilde{A}_t^{ref} - \hat{A}_t^{int}\right)
$$

**含义**: 仅在两个 critic 对优势正负号一致时才混合参考信号，$\alpha=0.5$ 控制混合强度。防止冲突蒸馏梯度。

**符号说明**:
- $g_t \in \{0, 1\}$: 协议门，1 表示两个 critic 符号一致
- $\alpha$: 混合权重（默认 0.5）

### 公式 5：[[近端教师分布|Proximal Teacher 构造]]

$$
q^\star_{t,k} = \mathrm{softmax}\!\left(z_{t,k} + \eta \cdot \omega_t \cdot e_{\hat{a}_{t,k}}\right)
$$

**含义**: 对 VLA 原始 logits 施加优势加权扰动，构造近端教师分布——正优势轨迹被推高概率，负优势被压低。

**符号说明**:
- $z_{t,k}$: VLA 在第 $t$ 步、第 $k$ 个 token 位置的原始 logit 向量
- $\eta = 1.0$: 扰动强度
- $\omega_t$: 裁剪归一化后的优势权重
- $e_{\hat{a}_{t,k}}$: 采样动作 token 的 one-hot 向量

### 公式 6：[[优势引导蒸馏损失|AGD 损失]]

$$
\mathcal{L}_{AGD} = \mathbb{E}_{\hat{\tau}}\!\left[\frac{1}{TK} \sum_{t=1}^T \sum_{k=1}^K \mathrm{KL}(q^\star_{t,k} \,\|\, p^\theta_{t,k})\right]
$$

**含义**: 用前向 KL 散度将教师分布 $q^\star$ 蒸馏到学生策略 $p^\theta$，提供 token 级密集监督。

**符号说明**:
- $q^\star_{t,k}$: 近端教师分布（由 AGD 构造）
- $p^\theta_{t,k}$: 当前策略在第 $k$ token 位置的分布
- $T$: episode 步数，$K=7$: 每步 action token 数

### 公式 7：[[策略改进下界|Theorem 1 — 策略改进下界]]

$$
J(\pi_{\theta'}) \geq J(\pi_\theta) + \frac{1}{T}\sum_{t=1}^T \mathbb{E}_{h_t \sim d_t^{\pi_{\theta'}}}\!\left[\beta\tau \mathrm{KL}(q^\star_t(\cdot|h_t) \,\|\, \pi_\theta(\cdot|h_t)) - C_B\sqrt{D_t^{dist}(h_t)}\right] - \varepsilon_{cal}
$$

**含义**: 在温和校准条件下，ROAD-VLA 的更新可以保证策略性能单调不下降，前提是蒸馏信号强度（KL 项）大于分布偏移惩罚项。

**符号说明**:
- $\beta$: 策略改进系数
- $\tau$: KL 正则化温度
- $D_t^{dist}(h_t)$: 当前策略与参考策略之间的分布漂移
- $C_B$: 常数界
- $\varepsilon_{cal}$: 校准误差项

### 公式 8：[[KL 链式分解|自回归教师的 KL 链式法则分解]]

$$
\mathrm{KL}(q^\star(\cdot|h_t) \,\|\, \pi_\theta(\cdot|h_t)) = \sum_{k=1}^K \mathbb{E}_{a_{<k} \sim q^\star}\!\left[\mathrm{KL}(q^\star_{t,k}(\cdot|a_{<k}, h_t) \,\|\, p^\theta_{t,k}(\cdot|a_{<k}, h_t))\right]
$$

**含义**: 将动作序列整体的 KL 散度分解为逐 token KL 之和，与自回归 VLA 解码结构对齐，支持 token 级监督。

---

## 关键图表

### Figure 1：ROAD-VLA 系统概览

![Figure 1](https://arxiv.org/html/2606.25800v1/image_cropped.png)

**说明**: ROAD-VLA 整体框架。左侧：OpenVLA 在线 rollout 收集带稀疏奖励的轨迹；中间：双 critic（在线 + 冻结）计算校准混合优势；右侧：通过优势权重 $\omega_t$ 扰动 logits，构造近端教师分布 $q^\star$，再以前向 KL 蒸馏更新策略 $p^\theta$。

### Figure 2：OOD 条件下的在线适应轨迹

![Figure 2](https://arxiv.org/html/2606.25800v1/x1.png)

**说明**: 七个机器人操作环境的成功率曲线。ROAD-VLA（蓝线）相比 PPO（橙线）收敛更快、峰值更高、后期更稳定，尤其在视觉鲁棒性场景（VR-DynamicTexture）和执行鲁棒性场景（ER-Repositioning）差距更明显。

### Figure 3：训练动态分析

![Figure 3](https://arxiv.org/html/2606.25800v1/x2.png)

**说明**: 三个子图分别展示：（左）策略熵演化——ROAD-VLA 维持更高熵，防止过早坍塌；（中）优势权重均值随训练上移，反映策略持续改进；（右）critic 协议率从 ~90% 下降至 71-75%，表明模型在探索更多元轨迹的同时协议门仍保持有效过滤。

### Table 1：七个环境的综合基准测试结果

| 类别 | 环境 | ID-PPO | ID-ROAD-VLA | OOD-PPO | OOD-ROAD-VLA | Δ-PPO | Δ-ROAD-VLA |
|------|------|--------|-------------|---------|--------------|-------|------------|
| 视觉鲁棒性 | VR-UnseenTable | 88±3 | 93±2 | 87±4 | 92±1 | 1 | 1 |
| 视觉鲁棒性 | VR-DynamicTexture | 87±1 | 88±1 | 65±5 | 69±5 | 22 | 19 |
| 视觉鲁棒性 | VR-DynamicNoise | 85±4 | 90±2 | 66±3 | 70±2 | 19 | 20 |
| 组合推理 | CR-MultiObject | 78±6 | 80±6 | 61±6 | 63±4 | 17 | 17 |
| 组合推理 | CR-MultiReceptacle | 84±3 | 84±3 | 57±1 | 62±2 | 27 | 22 |
| 执行鲁棒性 | ER-InitPose | 87±0 | 91±1 | 75±8 | 79±7 | 12 | 12 |
| 执行鲁棒性 | ER-Repositioning | 89±3 | 88±0 | 73±3 | 77±5 | 16 | 11 |
| **平均** | **All** | **85±3** | **88±2** | **69±4** | **73±4** | **16.3** | **14.6** |

**说明**: ROAD-VLA 在 ID 性能提升 3 个百分点（85→88%），OOD 提升 4 个百分点（69→73%），平均分布偏移降幅从 16.3% 降至 14.6%，跨所有 7 个环境全面优于 PPO。

### Table 2：消融实验（VR-UnseenTable OOD）

| 方法 | 成功率 (%) | 说明 |
|------|-----------|------|
| PPO | 87.2 ± 3.6 | 基线 |
| ROAD-VLA（RelSpatial PI）| 4.68 ± 0.0 | 相对空间描述文本教师 → 完全失效 |
| ROAD-VLA（Plan+RelSpatial PI）| 4.68 ± 0.0 | 高层规划+空间描述 → 完全失效 |
| ROAD-VLA（MCTS PI）| 75.8 ± 2.0 | MCTS 动作提示 → 低于 PPO |
| ROAD-VLA（JSD loss）| 85.9 ± 1.5 | 替换 KL 为 JSD → 性能下降 5.6pp |
| ROAD-VLA（w/o gated）| 89.8 ± 0.1 | 去掉协议门 → 轻微下降 |
| **ROAD-VLA（完整）** | **91.5 ± 1.2** | **最优** |

**关键发现**: 文本特权信息完全无效（模态鸿沟）；前向 KL 优于 JSD（+5.6pp）；协议门贡献 +1.7pp；完整方法较 PPO 提升 4.3pp。

### Table 3：环境技术标识符

| 类别 | 别名 | Simulator ID |
|------|------|--------------|
| 视觉鲁棒性 | VR-UnseenTable | PutOnPlateInScene25VisionImage-v1 |
| 视觉鲁棒性 | VR-DynamicTexture | PutOnPlateInScene25VisionTexture05-v1 |
| 视觉鲁棒性 | VR-DynamicNoise | PutOnPlateInScene25VisionWhole05-v1 |
| 组合推理 | CR-MultiObject | PutOnPlateInScene25MultiCarrot-v1 |
| 组合推理 | CR-MultiReceptacle | PutOnPlateInScene25MultiPlate-v1 |
| 执行鲁棒性 | ER-InitPose | PutOnPlateInScene25EEPose-v1 |
| 执行鲁棒性 | ER-Repositioning | PutOnPlateInScene25PositionChangeTo-v1 |

---

## 实验

### 数据集与环境

| 环境类别 | 环境数量 | 特点 | 评估类型 |
|---------|---------|------|---------|
| 视觉鲁棒性（VR）| 3 | 未见桌面/动态纹理/动态噪声 | ID + OOD |
| 组合推理（CR）| 2 | 多目标选择/多容器选择 | ID + OOD |
| 执行鲁棒性（ER）| 2 | 初始末端执行器位姿变化/目标重定位 | ID + OOD |

所有任务均为 **Put-on-Plate** 机器人操作任务，基于 LIBERO 仿真框架。

### 实现细节

- **基础模型**: [[OpenVLA]]-7B（7B 参数 VLA）
- **预热**: 140 条专家轨迹 SFT
- **在线 RL 框架**: GAE（$\gamma=0.99$，$\lambda=0.95$）
- **策略学习率**: $1 \times 10^{-4}$，余弦退火
- **梯度裁剪**: 1.0
- **蒸馏系数**: 0.5
- **扰动强度** $\eta$: 1.0
- **优势裁剪** $c$: 2.0
- **混合权重** $\alpha$: 0.5
- **并行环境**: 64 个
- **总交互步数**: 850,000 步/实验
- **硬件**: NVIDIA H100/H200 GPU

### 定量结果

ROAD-VLA 在所有 7 个环境上均超过 PPO：
- 分布内（ID）：88% vs 85%（+3pp）
- 分布外（OOD）：73% vs 69%（+4pp）
- OOD 性能衰退：14.6% vs 16.3%（减少 1.7pp 衰退）

提升在视觉鲁棒性（22%→19% 衰退）和执行鲁棒性（16%→11% 衰退）场景最显著。

---

## 批判性思考

### 优点

1. **动机清晰**: 模态鸿沟的发现和验证（文本教师完全失效的实验）给出了非常有说服力的 motivation。
2. **理论扎实**: Theorem 1 提供了策略改进的理论下界，不仅是经验性方法。
3. **设计简洁**: 不引入任何外部教师模型，完全自蒸馏，inference 时与标准 VLA 相同开销。
4. **实验全面**: 7 个不同类型 OOD 环境覆盖视觉/组合/执行三类偏移，消融充分。

### 局限性

1. **环境范围有限**: 所有实验均为 Put-on-Plate 单任务变体，泛化性到多任务/多机器人平台未验证。
2. **双 critic 开销**: 维护在线 + 冻结双 critic 增加了显存和计算成本，未量化额外开销。
3. **稀疏奖励假设**: 方法依赖任务成功/失败的二元信号，对更复杂的密集奖励设计未探索。
4. **实物实验缺失**: 论文仅在仿真中验证，真实机器人上的 sim-to-real 未涉及。

### 潜在改进方向

1. 将 AGD 扩展到多任务 VLA（如 OpenVLA-OFT、π0），验证跨任务泛化性。
2. 探索密集奖励下的 AGD，或结合人类偏好（RLHF）构造教师分布。
3. 研究协议门阈值动态调整策略，代替固定的符号一致性条件。

### 可复现性评估

- [ ] 代码开源（截至论文发表未公开）
- [ ] 预训练模型（使用公开的 OpenVLA-7B）
- [x] 训练细节完整（超参数列出完整）
- [x] 数据集可获取（LIBERO 仿真环境开源）

---

## 关联笔记

### 基于

- [[OpenVLA]]: 基础 VLA 模型，7B 参数，用于在线适应的起点
- [[PPO]]: 对比基线，直接在 VLA 上做在线 RL

### 对比

- [[Knowledge Distillation|知识蒸馏]]: ROAD-VLA 的核心机制，但引入优势加权使其与标准蒸馏不同
- [[MCTS（蒙特卡洛树搜索）|MCTS]]: 消融实验中尝试的文本特权信息来源之一

### 方法相关

- [[优势函数（Advantage Function）]]: 核心信号，用于构造教师分布
- [[KL 散度]]: AGD 损失的度量选择，相比 JSD 更适合 mean-seeking 蒸馏
- [[GAE（广义优势估计）]]: Critic 使用的优势估计方法
- [[自蒸馏（Self-Distillation）]]: ROAD-VLA 的核心框架类别

### 硬件/数据相关

- [[LIBERO]]: 使用的机器人操作仿真框架
- [[H100 GPU]]: 训练硬件

---

## 速查卡片

> [!summary] ROAD-VLA (2026)
> - **核心**: 优势引导自蒸馏，将稀疏奖励转化为逐 token 密集监督，用于 VLA 在线适应
> - **方法**: 双 critic 校准混合优势 + 协议门 + 指数倾斜教师分布 + 前向 KL 蒸馏
> - **结果**: OOD 73% vs PPO 69%，跨 7 个环境全面领先
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-25*
