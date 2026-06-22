---
title: "Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation"
method_name: "FAFM"
authors: [Jianing Guo, Fangzheng Chen, Zihao Mao, Wong Lik Hang Kenny, Zhenhong Wu, Yu Li, Yishuai Cai, Yuanpei Chen, Yikun Ban, Kai Chen, Qi Dou, Yaodong Yang, Xianglong Liu, Huijie Zhao, Simin Li]
year: 2026
venue: arXiv
tags: [flow-matching, action-generation, frequency-domain, robot-manipulation, vla, smoothness, dct]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.20135
created: 2026-06-22
---

# 论文笔记：Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beihang University, Peking University, CUHK, PKU-Psibot Lab, Zhongguancun Laboratory |
| 日期 | June 2026 |
| 项目主页 | https://anonymous.4open.science/r/FAFM |
| 对比基线 | [[Diffusion Policy]], [[Flow Matching Policy]], [[FreqPolicy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.20135) / [Code](https://anonymous.4open.science/r/FAFM) |

---

## 一句话总结

> FAFM 将机器人动作 chunk 投影到 [[离散余弦变换 (DCT)|DCT]] 频域执行 [[Flow Matching]]，再施加一阶导数正则化，以零额外参数同时解决异构控制频率与时序抖动两个根本问题。

---

## 核心贡献

1. **频域 Flow Matching**: 在 DCT 系数空间而非离散时间步空间匹配流，使系数仅依赖物理时间而非采样频率，天然兼容异构控制频率数据
2. **解析导数正则化**: 对 DCT 基展开的解析一阶导数施加监督（$\mathcal{L}_{vel}$），理论等价于 H¹ Sobolev 范数投影误差，系统性压制高频抖动
3. **零参数 plug-in 设计**: 不引入额外网络参数，可直接插入独立策略和 VLA（π₀、π₀.₅），在合成任务、外科手术、LIBERO 及真实机械臂上全面超越基线

---

## 问题背景

### 要解决的问题

现有基于 [[Action Chunking]] 的策略存在两个根本缺陷：

1. **训练期可识别性失败（Identifiability Failure）**：Open X-Embodiment 数据集中控制频率从 3 Hz（RT-1）到 50 Hz（ALOHA）不等。离散 chunk 按时间步索引时，相同物理运动在不同频率下产生不同的「下一步动作目标」，导致策略无法区分频率引发的歧义。

2. **推理期时序不一致（Temporal Inconsistency）**：相邻动作预测可能方向相反却幅度均小，在离散序列上看似合理，实则产生抖动轨迹（jitter）。在液体倾倒、外科手术等高精度任务中尤为致命。

### 现有方法的局限

- [[Diffusion Policy]] / [[Flow Matching Policy]]：直接在离散动作步上建模，未考虑频率异构性
- [[FreqPolicy]]：频域自回归，但无法同时保证多模态表达与平滑性
- [[Movement Primitive Diffusion]]：用基函数平滑，但不处理频率异构问题
- [[Streaming Flow Policy|SFP]]：流式生成，平滑性有改善但多模态覆盖弱

### 本文的动机

将 [[Action Chunking]] 表示为连续函数而非离散序列——通过 [[离散余弦变换 (DCT)|DCT]] 将动作映射到频域系数，系数值仅由物理时间决定，天然与控制频率解耦。再对连续重建函数的一阶导数加正则化，使损失函数等价于 [[Sobolev 范数|H¹ Sobolev 范数]]投影，从理论上保证平滑性。

---

## 方法详解

### 模型架构

FAFM 采用**频域 Flow Matching** 框架：

- **输入**: 观测 $\mathbf{o}$ + 噪声化 DCT 系数 $\hat{\mathbf{c}}_t$ + 时间步 $t$
- **Backbone**: Transformer（6层/4头/256维；VLA 情形为 π₀/π₀.₅ 的动作专家）
- **核心模块**: [[离散余弦变换 (DCT)|DCT]] 频域映射 + [[Flow Matching]] + [[Sobolev 范数|解析导数正则化]]
- **输出**: 连续动作函数 $\hat{v}(\tau)$，按目标频率采样得到离散 chunk
- **总参数**: 与基线 FM 完全一致（零额外参数）

### 核心模块

#### 模块 1：频域表示（Section 3.1）

**设计动机**: 将 [[Action Chunking]] 从「离散序列」升维为「连续函数」，用 [[离散余弦变换 (DCT)|DCT]] 系数参数化，使系数仅依赖物理时间 $\tau_n = n / f_\xi$，而非控制频率 $f_\xi$。

**具体实现**:
- 给定示范轨迹 $\xi$，控制频率 $f_\xi$，chunk 长度 $K = \lfloor T_\xi \cdot f_\xi \rfloor$
- 对每个自由度计算 $M+1$ 个 [[离散余弦变换 (DCT)|DCT]] 系数（$M \approx K/3$，截断保留低频信息）
- [[Flow Matching]] 在系数空间 $\hat{\mathbf{c}} \in \mathbb{R}^{(M+1) \times d}$ 上执行
- 推理时将系数代入余弦基展开，按任意目标频率重采样

**关键性质**：当 $K \to \infty$ 时，不同频率 $f_\xi$ 的 DCT 系数收敛到相同目标（定理 O(1/K) 误差），解决了可识别性失败问题。

#### 模块 2：一阶导数正则化（Section 3.2）

**设计动机**: 连续动作函数在 M 模式余弦子空间中，其一阶导数可从系数**解析计算**（无需数值差分），从而可在训练时直接监督速度场，系统性压制高频抖动。

**具体实现**:
- 从 DCT 系数解析计算速度场 $\hat{\dot{v}}(\tau)$（见公式章节）
- 构造速度监督损失 $\mathcal{L}_{vel}$，与 [[Flow Matching]] 损失 $\mathcal{L}_{FM}$ 加权组合
- $\lambda = 1$（固定，跨所有任务通用）
- 从理论上，合并损失等价于 [[Sobolev 范数|H¹ Sobolev 范数]] 的投影误差，高频分量受 $\omega_j^2$ 二次惩罚

---

## 关键公式

### 公式 1：[[Flow Matching|标准 Flow Matching 损失]]

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{(\mathbf{o},\mathbf{A}),t,\varepsilon} \left\|v_{\theta}(\mathbf{A}^t, \mathbf{o}, t) - (\mathbf{A} - \varepsilon)\right\|^2
$$

其中 $\mathbf{A}^t = t\mathbf{A} + (1-t)\varepsilon$

**含义**: 标准 Flow Matching 直接在离散动作 chunk $\mathbf{A} \in \mathbb{R}^{K \times d}$ 上回归速度场，从噪声 $\varepsilon$ 到数据 $\mathbf{A}$ 的线性路径。

**符号说明**:
- $v_\theta$: 速度场网络
- $\mathbf{A}^t$: $t$ 时刻插值样本
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声

---

### 公式 2：[[离散余弦变换 (DCT)|DCT 系数计算]]

$$
\hat{c}_j = 2\sum_{n=0}^{K-1} \xi_n \cos\!\left(\frac{j\pi(2n+1)}{2K}\right), \quad j = 0, \ldots, M
$$

其中 $\xi_n = \xi^*(n/f_\xi)$，$K = \lfloor T_\xi \cdot f_\xi \rfloor$

**含义**: 将离散动作序列 $\xi_n$ 映射为 $M+1$ 个余弦系数。系数 $\hat{c}_j$ 仅取决于物理时间 $\tau_n = n/f_\xi$ 和轨迹时长 $T_\xi$，与控制频率 $f_\xi$ 解耦。

**符号说明**:
- $\xi^*$: 连续真值轨迹
- $f_\xi$: 控制频率（Hz）
- $T_\xi$: 轨迹物理时长（秒）
- $K$: 离散步数
- $M$: 截断阶数（$\approx K/3$）

---

### 公式 3：[[Action Chunking|连续动作函数重建]]

$$
\hat{v}(\tau) = \frac{1}{2}\hat{c}_0 + \sum_{j=1}^{M} \hat{c}_j \cos(\omega_j \tau), \quad \omega_j = \frac{j\pi}{T_\xi}, \quad \tau \in [0, T_\xi]
$$

**含义**: 从 DCT 系数重建连续动作函数 $\hat{v}(\tau)$，可在任意物理时刻 $\tau$ 采样，实现频率自适应。

**符号说明**:
- $\hat{c}_j$: 第 $j$ 阶 DCT 系数
- $\omega_j = j\pi / T_\xi$: 第 $j$ 阶角频率
- $\tau$: 物理时间（秒）

---

### 公式 4：[[Flow Matching|频域 Flow Matching 损失]]

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{(\mathbf{o},\hat{\mathbf{c}}^*),t,\varepsilon} \left\|v_\theta(\hat{\mathbf{c}}_t, \mathbf{o}, t) - (\hat{\mathbf{c}}^* - \varepsilon)\right\|^2
$$

**含义**: 将 Flow Matching 目标从离散动作空间转移到 DCT 系数空间，网络直接预测系数的速度场。

**符号说明**:
- $\hat{\mathbf{c}}^*$: 目标 DCT 系数（来自真值轨迹）
- $\hat{\mathbf{c}}_t = t\hat{\mathbf{c}}^* + (1-t)\varepsilon$: 插值系数

---

### 公式 5：[[Sobolev 范数|解析速度场]]

$$
\hat{\dot{v}}(\tau) = -\sum_{j=1}^{M} \hat{c}_j \omega_j \sin(\omega_j \tau)
$$

**含义**: 连续动作函数 $\hat{v}(\tau)$ 关于物理时间的一阶导数，可从 DCT 系数**解析计算**，无需数值差分。

**符号说明**:
- $\hat{\dot{v}}(\tau)$: 物理速度场
- $\omega_j$: 第 $j$ 阶角频率

---

### 公式 6：[[Sobolev 范数|速度监督损失]]

$$
\mathcal{L}_{vel}(\theta) = \mathbb{E}_{(\xi,\tau_k)} \left\|\hat{\dot{v}}(\tau_k) - \dot{\xi}(\tau_k)\right\|^2
$$

**含义**: 在随机物理时刻 $\tau_k$ 上监督预测速度场与真值速度的 L² 误差，促进时序平滑。

**符号说明**:
- $\hat{\dot{v}}(\tau_k)$: 预测连续动作函数在 $\tau_k$ 处的速度
- $\dot{\xi}(\tau_k)$: 真值轨迹在 $\tau_k$ 处的速度

---

### 公式 7：[[FAFM|FAFM 联合损失]]

$$
\mathcal{L}_{FAFM} = \mathcal{L}_{FM} + \lambda \cdot \mathcal{L}_{vel}, \quad \lambda = 1
$$

**含义**: 将频域 Flow Matching 损失与速度正则化损失以 $\lambda=1$ 组合，形成 FAFM 的训练目标。

**符号说明**:
- $\mathcal{L}_{FM}$: 频域 Flow Matching 损失（公式 4）
- $\mathcal{L}_{vel}$: 速度监督损失（公式 6）
- $\lambda = 1$: 正则化权重（跨任务固定）

---

### 公式 8：[[Sobolev 范数|H¹ Sobolev 范数理论等价（定理 1）]]

$$
\mathcal{L}_{FAFM}(\theta) = \left\|\hat{\xi}_\theta - P_M\xi^*\right\|_{H_\mu^1}^2 = \sum_{j=0}^{M}(1 + \lambda\omega_j^2)\left|\langle\hat{\xi}_\theta - P_M\xi^*, \phi_j\rangle_{L^2}\right|^2
$$

**含义**: FAFM 损失严格等价于 H¹ Sobolev 加权范数下的投影误差。高频分量受 $\omega_j^2$（即 $(j\pi/T_\xi)^2$）二次惩罚，天然压制抖动（jerk $\propto j^3$），同时保留低频多模态。

**符号说明**:
- $\hat{\xi}_\theta$: 网络预测的连续动作函数
- $P_M\xi^*$: 真值轨迹在 M 阶余弦子空间的投影
- $\mu_j = 1 + \lambda\omega_j^2$: 频率相关权重
- $\phi_j$: 余弦基函数

---

## 关键图表

### Figure 1：问题引入与方法概览

![Figure 1: Introduction overview](https://arxiv.org/html/2606.20135v1/figure/intro_v4.png)

**说明**: 左侧展示两个缺陷——「可识别性失败」（相同物理运动在不同频率下产生不同 chunk 目标）与「时序不一致」（相邻预测方向冲突导致抖动）。右侧展示 FAFM 解决方案：[[离散余弦变换 (DCT)|DCT]] 频域匹配 + 速度监督，以零参数实现连续平滑生成。

---

### Figure 2：FAFM 方法流程

![Figure 2: Method overview](https://arxiv.org/html/2606.20135v1/figure/method_v4.png)

**说明**: FAFM 的完整流程。示范轨迹以物理时间为锚点计算 [[离散余弦变换 (DCT)|DCT]] 系数，系数送入 [[Flow Matching]] 网络学习速度场，推理时对系数积分后用余弦基展开重建连续动作函数，解析导数提供速度监督。

---

### Figure 3：合成双模态轨迹对比

![Figure 3: Synthetic toy bi-modal trajectory](https://arxiv.org/html/2606.20135v1/figure/toy_rigorous_v3.png)

**说明**: 1D 双正弦交叉模态任务。[[Diffusion Policy]] 和 FM 基线或无法分离两个模态，或产生抖动；FAFM 是**唯一**同时实现双模态分离与平滑输出的方法。验证了 [[Sobolev 范数|H¹ 正则化]] 对多模态表达无负面影响。

---

### Figure 4：障碍物避障轨迹可视化

![Figure 4: Successful obstacle avoidance trajectories](https://arxiv.org/html/2606.20135v1/figure/obstacle_avoidance_v2.png)

**说明**: FAFM 生成的避障轨迹样本，同时展现多模态覆盖（12 条不同路径）与平滑性（LDLJ=-5.60，基线最好为 -6.78），两者不再矛盾。

---

### Figure 5：LapGym 外科手术任务

![Figure 5: LapGym surgical tasks and convergence](https://arxiv.org/html/2606.20135v1/figure/lapgym_v2.png)

**说明**: (a) 四个腹腔镜手术任务：绳线穿针（RT）、抓取提升接触（GLT）、双手组织操作（BTM）、结扎环（LL）。(b) 成功率收敛曲线显示 FAFM 不仅最终性能最优，收敛速度也更快。

---

### Figure 6：消融实验（绳线穿针任务）

![Figure 6: Ablation on rope threading](https://arxiv.org/html/2606.20135v1/figure/ablation.png)

**说明**: 消融 [[离散余弦变换 (DCT)|DCT]] 变换（w/o DCT 退化为标准 FM）和速度损失（w/o $\mathcal{L}_{vel}$）的对比。DCT 变换是主要贡献因子；$\mathcal{L}_{vel}$ 进一步提升平滑性。

---

### Figure 7：真实机械臂拾放任务

![Figure 7: Real-robot policy pick&place](https://arxiv.org/html/2606.20135v1/figure/real_sm.png)

**说明**: Franka 机械臂执行杯子放置任务的真实部署，使用独立策略（40 次示范，20 次测试）。FAFM 实现平滑、稳定的抓取-放置动作。

---

### Figure 8：LIBERO 基准测试

![Figure 8: LIBERO results](https://arxiv.org/html/2606.20135v1/figure/libero_v2.png)

**说明**: 在 LIBERO 基准（抽屉、盒子、桌面任务）上的结果。干净条件下 FAFM 领先，机械偏差条件（恒定偏移扰动）下优势更明显——频域中仅 DC 系数受影响，运动形状完整保留，展现鲁棒性。

---

### Figure 9：混合控制频率测试

![Figure 9: Mixed-frequency results](https://arxiv.org/html/2606.20135v1/figure/mix_fps_v2.png)

**说明**: 抽屉任务中，单频率（10Hz，43次示范）vs 混合频率（5/10/20Hz，共43次）。**π₀ 在混合频率下成功率跌至 0%**，而 FAFM 保持 **92%** 成功率，直接验证了频域表示解决可识别性失败的有效性。

---

### Figure 10：π₀.₅ 真实世界部署

![Figure 10: Real-world deployment with π₀.₅](https://arxiv.org/html/2606.20135v1/figure/realworld_v4.png)

**说明**: 在 π₀.₅ VLA 基础上插入 FAFM，进行拾放（+15% 成功率，p<0.001）和多障碍物避障（+10% 成功率，p<0.05）真实测试，同时统计显著提升轨迹平滑度（LDLJ 改善 p<0.05）。

---

### Table 1：障碍物避障实验结果

| 方法 | 成功率 (SR) | 对数无量纲抖动 (LDLJ) ↑ | 模式数 (M) |
|------|-----------|----------------------|----------|
| Diffusion Policy | 35% | -9.16±0.77 | 8 |
| Streaming Flow Policy | 49% | -6.98±0.82 | 3 |
| Movement Primitive Diffusion | 16% | -6.78±0.47 | 2 |
| FreqPolicy | 39% | -9.02±1.11 | 10 |
| Flow Matching (FM) | 48% | -8.62±0.69 | 14 |
| **FAFM (Ours)** | **61%** | **-5.60±1.08** | **12** |

**说明**: FAFM 在成功率（+12% vs 次优 FM）、平滑性（LDLJ 最高）和多模态覆盖（12 条不同路径）上同时领先。LDLJ 越接近 0 越平滑。

---

### Table 2：LapGym 外科手术实验结果（成功率 % / LDLJ 均值±标准差）

| 方法 | 绳线穿针 (RT) SR | LDLJ | 抓取提升 (GLT) SR | LDLJ | 组织操作 (BTM) SR | LDLJ | 结扎环 (LL) SR | LDLJ |
|------|----------------|------|-----------------|------|-----------------|------|--------------|------|
| Diffusion Policy | 92 | -10.88±1.61 | 100 | -16.62±0.52 | 96 | -6.05±2.18 | 100 | -13.04±0.93 |
| SFP | 72 | -11.54±2.26 | 13 | -17.09±1.44 | 95 | -5.62±2.53 | 99 | -12.46±1.33 |
| MPD | 94 | -9.38±2.12 | 99 | -14.37±0.52 | 100 | -3.81±1.92 | 100 | -11.47±1.00 |
| FreqPolicy | 89 | -9.33±1.40 | 100 | -15.52±0.55 | 83 | -4.71±1.88 | 100 | -12.49±0.90 |
| FM | 89 | -10.69±2.06 | 100 | -16.62±0.60 | 94 | -5.82±1.87 | 100 | -13.23±0.87 |
| **FAFM** | **97** | **-7.57±1.32** | **100** | **-14.31±0.58** | **99** | **-2.21±2.34** | **100** | **-11.28±1.17** |

**说明**: FAFM 在所有四个外科任务上同时达到最高成功率和最佳平滑性（LDLJ 最接近 0）。BTM 任务的 LDLJ 改善最显著（-2.21 vs 次优 -3.81），证明在软体接触操作中平滑性尤为重要。

---

## 实验

### 数据集

| 数据集 / 环境 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| 合成 1D 双模态轨迹 | 500 示范 | 两条交叉正弦曲线，可控多模态 | 验证多模态 + 平滑性 |
| PyBullet 障碍物避障 | 合成生成 | 多障碍路径规划，需多模态 | 成功率 + 模式多样性 |
| LapGym (4 任务) | 每任务独立 | 腹腔镜手术，高精度软体操作 | 外科场景平滑性 |
| LIBERO | 标准基准 | 多任务机器人学习，含分布偏移 | VLA + 频率异构测试 |
| 真实 Franka (独立) | 40 次示范，20 次测试 | Pick&Place | 真实世界验证 |
| 真实 Franka (π₀.₅) | 标准协议 | Pick&Place + 多障碍避障 | VLA 真实部署 |

### 实现细节

- **Backbone（独立策略）**: Transformer（6层，4头，256维 embedding）
- **Backbone（VLA）**: π₀ / π₀.₅ 全参数微调（VLM + 动作专家）
- **优化器**: AdamW（lr=1.0×10⁻⁴，betas=(0.95, 0.999)）
- **Batch Size**: 独立策略 256，VLA 32
- **训练轮数**: 独立策略 3000 epochs（合成玩具 500）
- **VLA 训练步数**: 30,000 步
- **推理步数**: Euler ODE，10 步
- **Chunk 长度**: 独立策略 K=12，VLA K=50
- **DCT 截断**: M≈K/3（独立策略 M=4，VLA M=16）
- **正则化权重**: λ=1（固定，无需调参）

### 可视化结果

合成双模态实验中，FAFM 是唯一同时捕捉两条交叉正弦模态并保持平滑输出的方法。在混合频率 LIBERO 测试中，π₀ 完全崩溃（0% 成功率），而 FAFM 维持 92%，展示了频域表示在异构数据上的决定性优势。

---

## 批判性思考

### 优点

1. **理论严谨**: FAFM 损失到 H¹ Sobolev 范数的等价性（定理 1）提供了坚实的理论保证，不仅是经验观察
2. **零参数 plug-in**: 不改变网络结构，直接替换训练目标，兼容任意 Flow Matching 基础架构
3. **多模态 + 平滑性兼得**: 通过频率权重机制，高频分量被压制而低频多模态保留，从根本上解决了二者的表观矛盾
4. **频率异构鲁棒性**: 混合频率实验（0% vs 92%）直接证明了方法对真实数据多样性的泛化能力

### 局限性

1. **DCT 截断超参 M**: 虽有 $M \approx K/3$ 经验规则，但对于不规则运动（如快速抓取后长时间保持）可能需要更精细调整
2. **物理速度监督依赖**: $\mathcal{L}_{vel}$ 需要从真值轨迹中计算数值速度，对噪声示范敏感
3. **仅评估末端执行器动作**: 未验证在关节空间（高维、强耦合）或触觉反馈场景的效果
4. **匿名代码库**: 代码仍处于匿名状态，完整复现依赖最终版本发布

### 潜在改进方向

1. **自适应 M 选择**: 根据任务复杂度或示范频谱自动选择截断阶数
2. **H² 正则化**: 同时监督二阶导数（加速度），进一步压制机械振动
3. **与 [[Diffusion Policy]] 结合**: 探索 DDPM 基础上的频域训练，扩展到非 Flow Matching 框架

### 可复现性评估

- [ ] 代码开源（目前为匿名仓库，待正式发布）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、架构均在附录详述）
- [x] 数据集可获取（LapGym、LIBERO 均为公开基准）

---

## 关联笔记

### 基于

- [[Flow Matching]]: 核心训练框架，FAFM 在其系数空间版本
- [[离散余弦变换 (DCT)|DCT 动作 Token 化]]: 频域动作表示基础（FAST 方法使用 DCT 量化）
- [[Action Chunking]]: FAFM 扩展的基础动作表示形式
- [[π₀]]: VLA 扩展实验的基础模型

### 对比

- [[Diffusion Policy]]: 离散扩散基线，无频率感知
- [[FreqPolicy]]: 同为频域策略但自回归方式，无法兼顾多模态和平滑
- [[Movement Primitive Diffusion]]: 用运动基元平滑，但不处理频率异构
- [[Streaming Flow Policy|Streaming Flow Policy (SFP)]]: 流式 flow，平滑性优于 DP 但多模态弱

### 方法相关

- [[Sobolev 范数]]: 理论基础（H¹ Sobolev 范数加权投影）
- [[离散余弦变换 (DCT)]]: 核心变换工具
- [[Flow Matching]]: 生成建模框架
- [[Action Chunking]]: 被替代的离散动作表示
- [[FAST]]: 同样用 DCT 的动作表示方法（量化 vs 流匹配）

### 硬件/数据相关

- [[Franka Panda]]: 真实机械臂实验平台
- [[LapGym]]: 腹腔镜手术仿真环境
- [[LIBERO]]: 机器人多任务基准

---

## 速查卡片

> [!summary] FAFM (Frequency-Aware Flow Matching)
> - **核心**: DCT 频域 Flow Matching + 解析导数正则化，零参数兼容 VLA
> - **方法**: $\mathcal{L}_{FAFM} = \mathcal{L}_{FM}^{coeff} + \lambda \mathcal{L}_{vel}$，等价 H¹ Sobolev 投影
> - **结果**: 障碍避障 61% SR，LapGym 97% RT，混合频率 92%（π₀ 0%），真实 +15%/+10%
> - **代码**: https://anonymous.4open.science/r/FAFM

---

*笔记创建时间: 2026-06-22*
