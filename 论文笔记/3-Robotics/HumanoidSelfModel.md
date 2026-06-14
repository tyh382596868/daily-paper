---
title: "Proprioceptive-visual correspondence enables self-other distinction in humanoid robots"
method_name: "HumanoidSelfModel"
authors: [Yurun Chen, Tianyuan Gao, Yizhong Ge, Shikun Ban, Yizhou Wang, Hongkai Xiong, Wenjun Zeng, Wentao Zhu]
year: 2026
venue: arXiv
tags: [self-other-distinction, proprioception, humanoid-robot, self-modeling, contrastive-learning, neural-implicit-representation, motion-retargeting]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.13222v1
created: 2026-06-14
---

# 论文笔记：Proprioceptive-visual correspondence enables self-other distinction in humanoid robots

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 宁波东方理工大学、上海交通大学、北京大学、卡内基梅隆大学、华东师范大学、宁波数字孪生研究院 |
| 日期 | June 2026 |
| 项目主页 | [humanoid-self-model](https://euron-zc.github.io/humanoid-self-model/) |
| 对比基线 | [[NeRF]] / [[Neural Implicit Representation]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13222) |

---

## 一句话总结

> 仅靠本体感知-视觉对应关系，无需身份标签或运动学模型，让人形机器人从共享场景中区分"自我"与"他者"，并学习预测自身3D体积占用，支撑目标到达、碰撞规避和运动重定向等下游任务。

---

## 核心贡献

1. **无监督自我-他者区分**: 提出基于[[注意力引导对比学习|注意力引导对比学习]]的两层优化框架，无需身份标签，从 [[本体感知]]-视觉同步关系中区分自身 body mask（验证准确率 >99.5%）
2. **无运动学模型的自我建模**: 以[[神经隐函数|神经隐函数]]（pose-conditioned [[Neural Implicit Representation|隐式占用场]]）直接从本体感知状态预测机器人 3D 体积，无需 CAD 模型或正向运动学
3. **多任务下游能力**: 自我模型驱动目标到达（成功率 88.0%）、碰撞感知运动规划（成功率 71.4%）和人到机器人运动重定向（误差 36.1 mm）

---

## 问题背景

### 要解决的问题

人形机器人在与人类共享的场景中，需要实时识别哪一个视觉实体是自己的身体（自我-他者区分），并预测自身身体在给定关节配置下的三维形态（自我建模），这两个能力是安全交互的基础。

### 现有方法的局限

- 传统方法依赖**身份标签**（颜色标记、AR 标签）或**运动学模型**（CAD 模型 + 正向运动学），在开放环境中难以泛化
- 基于运动学的自我建模（如 FFKSM）无法捕获与视角相关的视觉细节（柔性链接、线缆等）
- 现有对比学习方法针对单实例自我识别，无法处理同类多实体（两台人形机器人）场景

### 本文的动机

婴儿在没有明确身份标签的情况下，通过本体感知（肢体运动感受）与视觉（看到对应肢体运动）的时间同步性，逐渐建立"镜像自我认知"。本文将这一生物学启发迁移到机器人：利用**本体感知状态与视觉观测之间的同步性**，作为区分自我和他者的唯一信号。

---

## 方法详解

### 系统框架

**HumanoidSelfModel** 采用**两阶段流水线**：

- **阶段一**: [[Self-Other Distinction|自我-他者区分]] — 从 RGB 图像候选 mask 中识别哪个 mask 属于机器人自身
- **阶段二**: [[Neural Implicit Representation|无运动学自我建模]] — 学习从本体感知状态预测自身 3D 体积占用场
- **输出**: 伪真值 (pseudo-GT) self mask + 3D 占用场，支撑下游控制任务

### Figure 1: 系统总览

![Figure 1](https://arxiv.org/html/2606.13222v1/x1.png)

**说明**: 人形机器人在共享场景中面临两个耦合问题：识别哪个可见身体是自己的，以及预测自身身体在给定本体感知状态下应有的外观。框架先区分自我与他者，再学习无运动学自我模型，最终支撑目标到达、碰撞规避规划和运动重定向。

### Figure 6: 方法架构详解

![Figure 6](https://arxiv.org/html/2606.13222v1/x6.png)

**说明**: 左侧为自我-他者区分模块（本体感知编码器 + 视觉实例编码器 + 注意力引导对比学习）；右侧为无运动学自我建模模块（部件感知编码器 + 隐函数 + 有界体积 mask 渲染）。

---

### 核心模块一：自我-他者区分

**设计动机**: 利用[[Self-Supervised Learning|自监督学习]]中的时间同步信号——机器人本体感知与自身视觉观测天然对齐，而与他者观测不对齐。

**输入**:
- [[本体感知|本体感知状态]] $\bm{\theta}_t = [\bm{q}_t, \psi_t] \in \mathbb{R}^{30}$（29 个关节角 + 全局偏航角）
- 视觉候选 mask（$K=2$ 个，来自图像分割），经位置和尺度归一化

**两层优化**:

- **内层（Inner）**: [[注意力引导对比学习]] 中的 Soft 注意力加权，动态选择与本体感知更匹配的候选 mask
- **外层（Outer）**: 跨帧 [[InfoNCE 损失]] 对齐，拉近正对（同帧本体感知 + 对应 bag 特征），推远负对

**图示结果（Figure 2）**:

![Figure 2](https://arxiv.org/html/2606.13222v1/x2.png)

**说明**: 上方展示验证准确率对比（本方法 vs. 基线），下方展示 mask 选择示例、余弦相似度分布（同步 vs. 异步）和 t-SNE 嵌入（训练前 vs. 后）。

---

### 核心模块二：无运动学自我建模

**设计动机**: 传统运动学模型依赖 CAD 数据，本文以[[Neural Implicit Representation|神经隐式表示]]（pose-conditioned density/visibility field）直接从本体感知状态预测 3D 体积。

**输入**:
- 完整本体感知状态 $\mathcal{S}_t = [\bm{q}_t, \bm{r}_t, \bm{p}_t] \in \mathbb{R}^{36}$（关节角 + 根节点角速度 + 全局位姿）
- 相机内参 + 伪真值 self mask

**部件感知编码器**: 将躯干和双侧肢体分开处理，适应 Unitree G1 的 29-DoF 身体构型。

**隐函数映射**:

$$
F_\phi(\bm{x}'_k, \bm{d}'_{uv}, \tilde{\mathcal{S}}_t) \mapsto (v_k, \sigma_k)
$$

输出每个 3D 采样点的密度 $\sigma_k$（表示体积占用）和可见性 $v_k$（视角相关）。

**有界体积 mask 渲染**: 借鉴[[NeRF]] 的 alpha 合成，限制射线采样在机器人体积范围内（bounded volumetric rendering），使用 [[Self-Supervised Learning|自监督]]的伪 GT mask 进行监督。

**图示结果（Figure 3）**:

![Figure 3](https://arxiv.org/html/2606.13222v1/x3.png)

**说明**: 仿真环境中不同姿态/视角下预测的体积占用（上）；真实世界中有人类干扰物的预测（中）；伪 GT mask 质量对预测的影响（下）以及量化指标（IoU、MSE、MAE、Chamfer Distance）。

---

## 关键公式

### 公式1: [[本体感知|完整本体感知状态]]

$$
\mathcal{S}_t = [\bm{q}_t, \bm{r}_t, \bm{p}_t] \in \mathbb{R}^{36}
$$

**含义**: 机器人在时刻 $t$ 的完整状态向量，维度 36

**符号说明**:
- $\bm{q}_t$: 29 维关节角向量
- $\bm{r}_t$: 根节点角速度
- $\bm{p}_t$: 全局位姿（位置 + 朝向）

### 公式2: [[Self-Other Distinction|自我-他者区分用状态]]

$$
\bm{\theta}_t = [\bm{q}_t, \psi_t] \in \mathbb{R}^{30}
$$

**含义**: 用于自我-他者区分的简化状态（不含全局平移），维度 30

**符号说明**:
- $\bm{q}_t$: 29 维关节角
- $\psi_t$: 全局偏航角

### 公式3: [[注意力引导对比学习|候选 mask 相似度]]

$$
s_k = \bm{f}_{\text{state}}^{\top} \bm{f}_{\text{image},k}, \quad k \in \{1,2\}
$$

**含义**: 计算本体感知特征与每个候选 mask 的视觉特征之间的余弦相似度

**符号说明**:
- $\bm{f}_{\text{state}}$: 本体感知编码器输出特征
- $\bm{f}_{\text{image},k}$: 第 $k$ 个候选 mask 的视觉特征

### 公式4: [[注意力引导对比学习|Soft 注意力权重]]

$$
\alpha_k^i = \frac{\exp(s_k^i/\tau_a)}{\sum_{\ell=1}^{2}\exp(s_\ell^i/\tau_a)}
$$

**含义**: 以 softmax 温度归一化，为两个候选 mask 分配注意力权重，实现隐式自我身份选择

**符号说明**:
- $\tau_a = 0.003$: 注意力温度参数
- $s_k^i$: 第 $i$ 帧中第 $k$ 个候选 mask 的相似度

### 公式5: [[注意力引导对比学习|Bag 特征聚合]]

$$
\bm{f}_{\text{bag}}^i = \sum_{k=1}^{2}\alpha_k^i \bm{f}^i_{\text{image},k}
$$

**含义**: 以注意力权重加权聚合候选 mask 特征，得到软选择的"自我"视觉表示

### 公式6: [[InfoNCE 损失|对比学习损失]]

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp((\bm{f}^i_{\text{state}})^{\top}\bm{f}^i_{\text{bag}}/\tau_l)}{\sum_{j=1}^{B}\exp((\bm{f}^i_{\text{state}})^{\top}\bm{f}^j_{\text{bag}}/\tau_l)}
$$

**含义**: 在一个 batch 内，最大化同帧本体感知特征与其对应 bag 特征的相似度，同时最小化与其他帧的相似度

**符号说明**:
- $B$: batch size（= 32）
- $\tau_l = 0.01$: 对比学习温度参数

### 公式7: [[Neural Implicit Representation|位姿归一化]]

$$
\tilde{\bm{p}}_t = \bm{R}_0^{-1}(\bm{p}_t - \bm{p}_0), \quad \tilde{\bm{R}}_t = \bm{R}_0^{-1}\bm{R}_t
$$

**含义**: 将全局位姿归一化到机器人初始坐标系，消除全局漂移影响，使模型只关注关节配置

**符号说明**:
- $\bm{R}_0, \bm{p}_0$: 初始根节点朝向和位置
- $\tilde{\bm{R}}_t, \tilde{\bm{p}}_t$: 归一化后的朝向和位置

### 公式8: [[Neural Implicit Representation|3D 点坐标变换]]

$$
\bm{x}'_k = \tilde{\bm{R}}_t^{-1}(\bm{x}_k - \tilde{\bm{p}}_t - \bm{c}) + \bm{c}
$$

$$
\bm{d}'_{uv} = \frac{\tilde{\bm{R}}_t^{-1}\bm{d}_{uv}}{\|\tilde{\bm{R}}_t^{-1}\bm{d}_{uv}\|_2}
$$

**含义**: 将世界坐标系下的 3D 采样点和射线方向变换到机器人局部坐标系，实现位姿条件化

**符号说明**:
- $\bm{x}_k$: 世界坐标下第 $k$ 个采样点
- $\bm{c}$: 机器人中心偏移 $[0, -0.457, 0.793]^\top$
- $\bm{d}_{uv}$: 相机射线方向

### 公式9-11: [[NeRF|有界体积 Alpha 合成]]

$$
\alpha_k = 1 - \exp(-\sigma_k \delta_k)
$$

$$
T_k = \prod_{j<k}(1-\alpha_j)
$$

$$
\hat{M}(\bm{\rho}) = \sum_{k=1}^{N} T_k \alpha_k v_k
$$

**含义**: 沿射线 $\bm{\rho}$ 合成预测 mask 值，结合密度 $\sigma_k$ 和可见性 $v_k$，实现有界体积渲染

**符号说明**:
- $\delta_k$: 采样步长，最后一步延伸至远端边界
- $T_k$: 累积透射率
- $\hat{M}(\bm{\rho})$: 预测的 mask 像素值

### 公式12: [[Neural Implicit Representation|Mask 重建损失]]

$$
\mathcal{L}_{\text{rec}} = \frac{1}{|\mathcal{R}|}\sum_{\bm{\rho} \in \mathcal{R}}(\hat{M}(\bm{\rho}) - \mathcal{M}(\bm{\rho}))^2
$$

**含义**: 预测 mask 与伪真值 mask 之间的 MSE 损失，驱动自我模型学习

**符号说明**:
- $\mathcal{R}$: 采样射线集合
- $\mathcal{M}(\bm{\rho})$: 伪真值 mask 值

### 公式13: [[本体感知|手部位置估计]]（目标到达）

$$
\hat{\bm{c}}_{\text{hand}}(\mathcal{S}) = \frac{\sum_{i=1}^{N}\mathrm{ReLU}(\sigma_{\text{hand}}(\bm{x}_i,\mathcal{S}))\bm{x}_i}{\sum_{i=1}^{N}\mathrm{ReLU}(\sigma_{\text{hand}}(\bm{x}_i,\mathcal{S})) + \epsilon}
$$

**含义**: 以密度加权平均计算手部中心位置，用于梯度优化关节角实现目标到达

**符号说明**:
- $\sigma_{\text{hand}}$: 手部区域隐函数的密度输出
- $\epsilon$: 数值稳定小量

### 公式14: [[本体感知|目标到达损失]]

$$
\mathcal{L}_{\text{target}} = \|\hat{\bm{c}}_{\text{hand}}(\mathcal{S}(\bm{q}_{\text{arm}})) - \bm{p}^*\|_2
$$

**含义**: 最小化预测手部位置与目标位置之差，通过反向传播优化手臂关节角

**符号说明**:
- $\bm{q}_{\text{arm}}$: 手臂关节角（优化变量）
- $\bm{p}^*$: 目标点 3D 位置

### 公式15: [[运动重定向|人到机器人运动重定向损失]]

$$
\mathcal{L}_{\text{retarget}}(\bm{q}) = \frac{1}{K}\sum_{k=1}^{K}\|\hat{\bm{c}}_k(\bm{q}) - \bm{c}^*_k\|_2
$$

**含义**: 对 $K$ 个关键点，最小化机器人预测位置与人类演示关键点位置之差，实现无运动学的全身重定向

**符号说明**:
- $K$: 关键点数量
- $\hat{\bm{c}}_k$: 机器人第 $k$ 个关键点的预测位置
- $\bm{c}^*_k$: 人类演示中第 $k$ 个关键点的目标位置

---

## 关键图表

### Figure 4: 目标到达 + 碰撞规避规划

![Figure 4](https://arxiv.org/html/2606.13222v1/x4.png)

**说明**: 左图展示人类将夜灯放在不同位置，机器人通过自我模型驱动的目标到达触碰点亮（前视和俯视）；右图展示机器人手臂穿过带圆孔隔板进行碰撞感知运动规划（从悬垂到穿孔到到达目标）。

### Figure 5: 人到机器人运动重定向

![Figure 5](https://arxiv.org/html/2606.13222v1/x5.png)

**说明**: 上行为人类演示动作序列，下行为通过优化机器人关节配置使自我模型关键点对齐人类关键点，得到的重定向机器人动作序列。

### Supplementary Table 1: 自我-他者区分消融实验

| 特征维度 | 尺度归一化 | 注意力温度 $\tau_a$ | 采样策略 | 验证准确率 |
|---------|-----------|-------------------|---------|----------|
| 8 | ON | 0.003 | All | 0.7538 |
| **16** | **ON** | **0.003** | **All** | **0.9969** |
| 32 | ON | 0.003 | All | 0.9340 |
| 64 | ON | 0.003 | All | 0.9860 |
| 16 | OFF | 0.003 | All | 0.8974 |
| 16 | ON | 0.03 | All | 0.8582 |
| 16 | ON | 0.3 | All | 0.6884 |
| 16 | ON | 0.003 | Single | 0.8765 |

**关键发现**: 特征维度 16、开启尺度归一化、温度 0.003、多帧采样（All）的组合达到最优准确率 99.69%；去掉尺度归一化、升高温度或改用单帧采样均导致显著性能下降。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实环境（人-机器人） | 25,667 帧 | Intel RealSense D455 RGB，含人类干扰物 | 训练 + 测试自我-他者区分 |
| 仿真环境（双人形） | 9,000 帧 | 两台 Unitree G1，同类多实体挑战场景 | 测试自我-他者区分 |
| 下游任务评估 | 50 次目标到达 / 14 次规划 | 真实机器人，不同目标位置和路径 | 测试目标到达 + 碰撞规避 |

### 实现细节

**自我-他者区分训练**:
- 特征维度: 16
- 注意力温度: $\tau_a = 0.003$，对比学习温度: $\tau_l = 0.01$
- Batch Size: 32
- 学习率: 0.001
- 训练轮数: 100
- 优化器: AdamW + 梯度裁剪
- 权重衰减: 0.01

**自我建模训练**:
- 射线采样范围: near = $d_{\text{cam}} - n_f$，far = $d_{\text{cam}} + n_f$
- 机器人中心偏移: $\bm{c} = [0, -0.457, 0.793]^\top$（米）
- 损失: MSE mask 重建 + 弱射线密度平滑（仅真实场景）
- 优化器: Adam + ReduceLROnPlateau 调度

**硬件与平台**:
- 机器人: Unitree G1（29-DoF 人形）
- 相机: Intel RealSense D455 RGB

### 主要实验结果

- **自我-他者区分准确率**: >99.5%（人-机器人场景）
- **目标到达成功率**: 88.0%（44/50 次）
- **碰撞规避规划成功率**: 71.4%（10/14 次）
- **运动重定向误差**: 36.1 mm（全身关键点平均误差）

### Supplementary Figure 1 & 2: 重建消融对比

![Supplementary Figure 1](https://arxiv.org/html/2606.13222v1/x7.png)

**说明**: 定性对比 ground-truth mask vs. scaled FFKSM baseline vs. 完整方法 vs. 仅密度变体 vs. FFKSM 渲染器变体，完整方法轮廓最精确且具有视角相关细节。

### Supplementary Figure 3: 运动重定向完整流程

![Supplementary Figure 3](https://arxiv.org/html/2606.13222v1/x9.png)

**说明**: 从单目人类演示 → 姿态估计 → 关键点提取 → 坐标变换 → 机器人兼容目标映射 → 神经逆运动学 → 全身机器人动作的完整流程。

---

## 批判性思考

### 优点

1. **生物启发的无监督框架**: 完全摒弃身份标签和运动学模型，仅用本体感知-视觉时间同步性，概念优雅
2. **端到端的 3D 自我表示**: 隐式占用场可微分，直接支持梯度优化的目标到达和重定向，无需额外规划器
3. **真实机器人验证**: 在 Unitree G1 上的实验证明了框架的工程可行性，不只停留于仿真

### 局限性

1. **双实例限制**: 目前只处理 $K=2$ 个候选 mask，场景中多于两个同类实体时需扩展
2. **无物理模型**: 自我模型只预测视觉占用，不包含力学/惯性信息，难以直接用于接触力控制
3. **规划成功率偏低**: 碰撞规避仅 71.4%，可能受伪 GT mask 精度和采样分辨率影响

### 潜在改进方向

1. 引入物理约束（质量、惯性）扩展自我模型为物理感知的自我模型
2. 将占用场扩展为支持接触检测的有符号距离场（SDF）
3. 与策略学习结合，将自我模型作为辅助状态表示

### 可复现性评估

- [ ] 代码开源（项目主页有页面，但未见代码链接）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（论文 Methods 节和 Supplementary 提供了完整超参数）
- [x] 数据集可获取（真实场景数据为自采集；Unitree G1 为商用机器人）

---

## 关联笔记

### 基于

- [[NeRF]]: 有界体积渲染的 alpha 合成方法借鉴自 NeRF 框架
- [[InfoNCE 损失]]: 外层对比学习的核心损失函数
- [[Self-Supervised Learning]]: 整体框架基于自监督本体感知-视觉对应

### 对比

- [[Neural Implicit Representation]]: FFKSM 等基于运动学的 baseline 使用显式模型，本文无运动学

### 方法相关

- [[注意力引导对比学习]]: 内层 Soft 注意力选择 self mask 的核心机制
- [[本体感知]]: 关节角 + 位姿状态，驱动整个框架的输入信号
- [[运动重定向]]: 利用自我模型实现无运动学的人到机器人运动迁移

### 硬件/数据相关

- [[Unitree G1]]: 29-DoF 人形机器人实验平台

---

## 速查卡片

> [!summary] Humanoid Self-Other Distinction via Proprioceptive-Visual Correspondence
> - **核心**: 无需标签/运动学，靠本体感知-视觉同步让人形机器人认识自己
> - **方法**: 注意力引导对比学习（区分）+ 神经隐式占用场（建模）
> - **结果**: 区分准确率 >99.5%，目标到达 88.0%，碰撞规避 71.4%，重定向误差 36.1 mm
> - **项目**: [humanoid-self-model](https://euron-zc.github.io/humanoid-self-model/)

---

*笔记创建时间: 2026-06-14*
