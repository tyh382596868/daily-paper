---
title: "GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning"
method_name: "GeneralVLA2"
authors: [Haoyu Wang, Guoqing Ma, Zeyu Zhang, Yandong Guo, Boxin Shi, Hao Tang]
year: 2026
venue: arXiv
tags: [vla, 3d-reconstruction, memory-management, robot-manipulation, multi-view, knowledge-bank]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17480
created: 2026-06-22
---

# 论文笔记：GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未标注（AIGeeksGroup） |
| 日期 | June 2026 |
| 项目主页 | [AIGeeksGroup/GeneralVLA-2](https://aigeeksgroup.github.io/GeneralVLA-2) |
| 对比基线 | [[VoxPoser]], [[CAP]], [[Hamster]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17480) / [Code](https://github.com/AIGeeksGroup/GeneralVLA-2) |

---

## 一句话总结

> GeneralVLA-2 通过几何感知多视图重建（GeoFuse-MV3D）和有治理的记忆库（Governed KnowledgeBank）两大模块，升级了通用 [[VLA]] 系统在 3D 物体重建精度和长期操作经验管理上的能力。

---

## 核心贡献

1. **GeoFuse-MV3D**: 将几何先验（外部 [[VGGT]] 重建）与遮罩验证、外观仿射校准、软视觉外壳约束和逐轴精细化融合，保守地修正 [[3D Gaussian Splatting]] 表示，在保持输入协议不变的前提下提升多视图重建质量。
2. **Governed KnowledgeBank**: 在原有 KnowledgeBank 基础上引入质量元数据（置信度、生命周期、冲突追踪）和精度导向检索，将仅追加型记忆转变为受控长期知识库，提升操作规划的经验复用效率。
3. **全面评测**: 在 GSO-30 重建 benchmark、Terminal-Bench 2.0、SWE-Bench Verified 以及 RLBench 14 个仿真任务和 4 个真实世界任务上系统验证两大模块的贡献，并提供完整消融分析。

---

## 问题背景

### 要解决的问题

通用 [[VLA]] 系统在机器人操作中面临两个关键瓶颈：（1）基于单目/多视图的 3D 物体重建存在几何幻觉问题，导致规划所依赖的 3D 证据不准确；（2）现有记忆管理（如 KnowledgeBank）采用仅追加写入策略，缺乏质量控制和生命周期管理，低质量经验会污染检索结果。

### 现有方法的局限

- **MV-SAM3D**（基线重建）：依赖单一 [[3D Gaussian Splatting]] pipeline，无法利用外部几何先验，在遮挡和视角稀疏时出现结构缺失。
- **原 KnowledgeBank**（ReasoningBank）：仅追加写入，无准入控制，无冲突消解，检索时退化为纯语义相似度匹配，无法区分高低质量记忆。

### 本文的动机

利用已有几何先验（如 [[VGGT]]）对 [[3D Gaussian Splatting]] 进行保守修正（而非激进替换），可在不改变输入协议的前提下降低重建误差；同时，为记忆系统引入结构化元数据和生命周期状态机，可将记忆管理从被动积累转变为主动治理，提升规划时上下文质量。

---

## 方法详解

### 系统架构

GeneralVLA-2 延续层级化 pipeline：

- **输入**: 语言指令 $q$ + 校准多视图 RGB-D 观测 $\{(I_i, M_i, K_i, T_i)\}_{i \in V_{in}}$
- **3D 证据生成**: [[GeoFuse-MV3D]] 将多视图输入转为精细化物体中心 3D 表示
- **记忆检索**: [[Governed KnowledgeBank]] 基于质量权重检索相关操作经验
- **规划**: [[3DAgent]] 结合 3D 证据 + 检索记忆生成多阶段末端执行器轨迹 $\tau_t$
- **执行**: 低层控制器跟随轨迹完成操作

轨迹表示为：

$$
\tau_t = \{(x_\ell, y_\ell, z_\ell, g_\ell)\}_{\ell=1}^{L_t}
$$

其中 $(x_\ell, y_\ell, z_\ell)$ 为末端执行器 3D 坐标，$g_\ell \in \{0, 1\}$ 为夹爪状态，$L_t$ 为轨迹步数。

---

### 核心模块一：GeoFuse-MV3D

**设计动机**: 利用 [[VGGT]] 等外部几何先验，通过遮罩验证过滤不可信的先验点，并以保守融合方式修正 [[3D Gaussian Splatting]] 初始重建，避免激进替换损害操作关键几何。

#### 多视图输入与投影

给定输入集：

$$
D = \{(I_i, M_i, K_i, T_i)\}_{i \in V_{in}}, \quad V_{in} = \{0,1,2,3,4\}
$$

3D 点 $p$ 投影到第 $i$ 视图像素坐标：

$$
\pi_i(p) = \Pi(K_i T_i [p; 1])
$$

**符号说明**:
- $I_i$: 第 $i$ 视角 RGB 图像
- $M_i$: 第 $i$ 视角物体遮罩
- $K_i$: 相机内参矩阵
- $T_i$: 相机外参（世界→相机变换）
- $\Pi$: 透视除法投影函数

#### 遮罩一致性分数

对每个 3D Gaussian 中心 $p$，计算跨视图遮罩一致性：

$$
s(p) = \frac{1}{\max(|V(p)|, 1)} \sum_{i \in V(p)} M_i(\pi_i(p))
$$

**含义**: 衡量点 $p$ 被多少视图的前景遮罩支持，得分越高说明几何可信度越高。

**符号说明**:
- $V(p)$: 能看到点 $p$ 的视角集合（正向深度测试通过）
- $M_i(\pi_i(p))$: 点 $p$ 在第 $i$ 视图投影位置的遮罩值（$\in [0,1]$）

#### 遮罩不一致损失

$$
E_{mask}(G) = \frac{1}{N} \sum_{j=1}^{N} (1 - s(x^j))
$$

**含义**: 衡量整个 Gaussian 集合 $G$ 中点的平均遮罩不支持程度，用于驱动轴向补偿优化。

#### 来源 A：几何先验 + 外观仿射校准

初始 Gaussian 表示：

$$
G_0 = \{(x_0^j, \theta_0^j)\}_{j=1}^N
$$

对每个视图进行外观仿射校准，消除光照差异：

$$
(g_i^*, b_i^*) = \mathop{\mathrm{arg\,min}}_{g_i, b_i} \sum_{u \in \Omega_i} \|g_i \odot \hat{I}_i(u) + b_i - I_i(u)\|_1 + \beta\|g_i - 1\|_2^2 + \beta\|b_i\|_2^2
$$

**含义**: 学习每视图的增益 $g_i$ 和偏置 $b_i$，使渲染图像 $\hat{I}_i$ 与真实图像 $I_i$ 尽量一致，避免外观偏差影响遮罩验证。

**符号说明**:
- $\hat{I}_i(u)$: Gaussian 渲染在视图 $i$ 像素 $u$ 处的颜色
- $\Omega_i$: 第 $i$ 视图有效像素集合
- $\beta$: 正则化系数，防止过拟合

#### 来源 B：无先验逐轴补偿

仅利用输入遮罩和相机位姿，对 Gaussian 位置进行逐轴线性变换：

$$
(a^*, \delta^*) = \mathop{\mathrm{arg\,min}}_{a, \delta} E_{mask}(T_{a, \delta}(G)) + \rho_a\|a - 1\|_2^2 + \rho_\delta\|\delta\|_2^2
$$

**含义**: 学习轴向缩放 $a$ 和平移 $\delta$，使变换后的 Gaussian 更好地符合遮罩约束，提供与来源 A 正交的几何修正路径。

逐轴精细化变换：

$$
p'' = c + (p' - c) \odot a + \delta
$$

**符号说明**:
- $c$: Gaussian 集合的中心点
- $a$: 逐轴缩放向量
- $\delta$: 平移偏移量
- $\odot$: 逐元素乘法

#### 保守几何修正（Soft Visual Hull）

对遮罩支持度低的点施加向内收缩（而非删除）：

$$
p' = c + (p - c)(1 - \lambda(p))
$$

其中收缩强度：

$$
\lambda(p) = \lambda_{max} \cdot \sigma\left(\frac{\tau - s(p)}{\eta}\right)^2
$$

**含义**: 遮罩支持度 $s(p)$ 越低，收缩越强；支持度高的点几乎不受影响。保守策略避免过度裁剪操作所需的几何细节。

**符号说明**:
- $\lambda_{max}$: 最大收缩比例
- $\tau$: 支持度阈值
- $\eta$: 软化温度参数
- $\sigma$: Sigmoid 函数

#### 置信度加权残差融合

将来源 A（几何先验修正后）与来源 B（逐轴补偿）融合：

$$
x_{out}^j = x_A^j + \alpha w^j (x_B^j - x_A^j), \quad \text{where} \quad w^j = \mathrm{clip}(s(x_A^j) s(x_B^j), 0, 1)
$$

**含义**: 权重 $w^j$ 由两个来源的遮罩支持度共同决定，支持度高的区域融合权重大；仅融合几何位置，外观属性保留来源 A。

输出 Gaussian：

$$
G_{out} = \{((1 - \alpha)x_A^j + \alpha x_B^j, \theta_A^j)\}_{j=1}^N
$$

**符号说明**:
- $\alpha$: 全局融合比例（超参数）
- $\theta_A^j$: 来源 A 的外观属性（颜色、不透明度等），保持不变

---

### 核心模块二：Governed KnowledgeBank

**设计动机**: 原有 KnowledgeBank 仅追加写入，无质量门控，导致低质量记忆污染检索结果。通过引入结构化元数据、生命周期状态机和准入控制，实现记忆的主动治理。

#### 记忆记录结构

每条记忆记录 $m$ 包含完整元数据：

$$
m = (i, q, c, y, s, z, \kappa, R, u, d, L, v)
$$

**符号说明**:
- $i$: 记录唯一 ID
- $q$: 查询上下文（任务描述）
- $c$: 内容摘要（操作经验）
- $y$: 结果标签（成功/失败）
- $s$: 来源信息
- $z$: 嵌入向量（用于语义检索）
- $\kappa$: 置信度分数
- $R$: 质量分数（验证器评分）
- $u$: 使用次数
- $d$: 最后使用时间戳
- $L$: 生命周期状态（provisional → active → summary → archive）
- $v$: 冲突标记（链接到冲突记录）

#### 验证器质量评分

操作完成后由验证器评估记忆质量：

$$
R_t = \frac{1}{|C|} \sum_{c \in C} \sum_{v \in V} p_\theta(v \mid q_t, X_t, \tau_t, c) \cdot \phi(v)
$$

**含义**: 综合多个评估标准 $C$（如任务完成度、安全性）和多个验证器 $V$ 的打分，计算轨迹 $\tau_t$ 的质量分数，作为记忆准入门控依据。

**符号说明**:
- $C$: 评估标准集合
- $V$: 验证器集合
- $p_\theta(v \mid \cdot)$: 验证器 $v$ 对当前轨迹的评分分布
- $\phi(v)$: 验证器 $v$ 的权重函数

#### 精度导向检索评分

$$
S(q_t, X_t, m) = r_{text}(q_t, m) + \kappa_m + b_{success}(m) + b_{recency}(m) + b_{usage}(m) - p_{conflict}(m) - p_{stale}(m)
$$

**含义**: 检索分数综合语义相似度、置信度奖励、成功经验奖励、近期使用奖励、高频使用奖励，同时惩罚冲突记录和过时记录，实现精度导向而非纯相似度导向的检索。

**符号说明**:
- $r_{text}(q_t, m)$: 查询与记忆的文本语义相似度
- $\kappa_m$: 记忆置信度奖励项
- $b_{success}(m)$: 成功经验额外加分
- $b_{recency}(m)$: 近期使用加分
- $b_{usage}(m)$: 高使用频率加分
- $p_{conflict}(m)$: 冲突惩罚项
- $p_{stale}(m)$: 陈旧记忆惩罚项

---

## 关键图表

### Figure 1: 系统总览

![Figure 1](https://arxiv.org/html/2606.17480v1/x1.png)

**说明**: GeneralVLA-2 整体 pipeline。当校准多视图观测可用时，[[GeoFuse-MV3D]] 将视图、遮罩和相机位姿转化为精细化物体中心 3D 证据；[[3DAgent]] 结合 [[Governed KnowledgeBank]] 检索结果生成多阶段末端执行器轨迹；执行模块跟随轨迹完成任务。

### Figure 2: GeoFuse-MV3D 重建分支

![Figure 2](https://arxiv.org/html/2606.17480v1/x2.png)

**说明**: GeoFuse-MV3D 保持与 MV-SAM3D 相同的多视图输入、遮罩和位姿，通过两条互补几何路径（来源 A：几何先验+外观仿射；来源 B：无先验轴向补偿）进行保守几何融合，输出精细化 [[3D Gaussian Splatting]] 表示。

### Figure 3: Governed KnowledgeBank 架构

![Figure 3](https://arxiv.org/html/2606.17480v1/x3.png)

**说明**: [[Governed KnowledgeBank]] 模块结构。写入路径经过验证器质量打分和准入控制；检索路径基于精度导向评分（含置信度、生命周期、冲突信息）筛选高质量记录；生命周期管理器定期将低活跃记忆从 active → summary → archive 状态转换。

### Figure 4: 真实世界无训练演示

![Figure 4](https://arxiv.org/html/2606.17480v1/x4.png)

**说明**: 四个真实世界操作任务（Move spray bottle、Open drawer、Open jar、Sort object）的 training-free 演示序列，展示 GeneralVLA-2 在实物环境的泛化能力。

### Figure 5: GSO-30 定性对比（第一组）

![Figure 5](https://arxiv.org/html/2606.17480v1/x5.png)

**说明**: MV-SAM3D 与 GeoFuse-MV3D 在 GSO-30 第一组物体上的定性对比。每行使用相同五视图输入，GeoFuse-MV3D 在物体完整性和位姿一致性上明显优于基线。

### Figure 6: GSO-30 定性对比（第二组）

![Figure 6](https://arxiv.org/html/2606.17480v1/x6.png)

**说明**: GeoFuse-MV3D 应用遮罩验证、外观保持几何精细化后，在第二组 GSO-30 物体上同样改善了物体完整性，输入协议与 MV-SAM3D 完全一致。

### Table 1: RLBench 仿真任务成功率（%）

| Method | Put_block | Play_jenga | Open_jar | Close_box | Open_box | Pickup_cup | Push_block |
|--------|-----------|-----------|----------|-----------|----------|-----------|-----------|
| VoxPoser | 70.70±2.31 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 26.70±14.00 | 25.33±8.33 |
| CAP | 84.00±16.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 14.67±4.62 | 8.00±4.00 |
| Hamster | 78.33±6.11 | 0.00±0.00 | 77.67±11.55 | 0.00±0.00 | 0.00±0.00 | 9.00±2.26 | 5.00±6.11 |
| **GeneralVLA-2 (Ours)** | **90.33±8.72** | **85.33±14.05** | **85.00±6.93** | **54.67±12.00** | **38.33±12.86** | **87.33±6.11** | 25.00±15.53 |
| GeneralVLA-2 w/o KB | 75.00±14.05 | 63.33±11.37 | 68.67±6.43 | 31.00±15.53 | 10.00±12.86 | 76.67±11.37 | 15.00±4.00 |

| Method | Take_umbrella | Sort_mustard | Open_wine | Lamp_on | Put_knife | Pick_&_lift | Insert_block |
|--------|--------------|-------------|-----------|---------|-----------|-----------|-------------|
| VoxPoser | 33.33±8.33 | 96.00±6.93 | 8.00±4.00 | 57.30±12.22 | 92.00±4.00 | 96.00±0.00 | 0.00±0.00 |
| CAP | 4.00±4.00 | 0.00±0.00 | 0.00±0.00 | 64.00±6.93 | 14.67±8.33 | 100.00±0.00 | 0.00±0.00 |
| Hamster | 8.67±2.31 | 44.33±12.86 | 34.33±20.13 | 61.00±8.00 | 23.00±0.00 | 96.00±0.00 | 0.00±0.00 |
| **GeneralVLA-2 (Ours)** | **68.00±15.62** | **79.33±19.22** | **44.67±14.05** | **78.67±10.58** | **63.67±12.86** | 90.67±12.00 | **34.33±11.14** |
| GeneralVLA-2 w/o KB | 48.00±15.53 | 57.33±6.43 | 26.33±14.05 | 58.67±15.62 | 43.67±6.11 | 66.00±19.22 | 13.33±6.93 |

**关键发现**: GeneralVLA-2 覆盖全部 14 个任务（对比方法覆盖 7-10 个），在大多数任务上达到最优，Play_jenga（85.33%）和 Close_box（54.67%）等难任务仅 GeneralVLA-2 能完成。移除 KnowledgeBank 后成功率普遍下降，验证其关键作用。

### Table 2: 真实世界任务完成率（%）

| Method | Move_spray_bottle | Open_drawer | Open_jar | Sort_object |
|--------|------------------|------------|----------|------------|
| CAP (0-shot) | 6.67 | 0.00 | 36.67 | 70.00 |
| RoboPoint (0-shot) | 0.00 | 0.00 | 20.00 | 63.33 |
| **GeneralVLA-2 (training-free)** | **63.33** | **40.00** | **53.33** | **83.33** |

**关键发现**: GeneralVLA-2 在四个真实世界任务上均优于基线，且为 training-free 设置，展示出强泛化性。

### Table 3: GSO-30 重建结果

| Method | CD ↓ (10⁻³) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-----------|--------|--------|---------|
| MV-SAM3D 基线 | 45.8876 | 13.2421 | 0.8051 | 0.2795 |
| **GeoFuse-MV3D** | **44.8770** (-2.20%) | **13.5547** (+2.36%) | **0.8134** (+1.03%) | **0.2739** (-2.02%) |

**关键发现**: GeoFuse-MV3D 在保持相同输入协议的前提下，在 Chamfer Distance（几何精度）和外观质量（PSNR、SSIM、LPIPS）上均有提升，验证几何融合策略的有效性。

### Table 4: GeoFuse-MV3D 组件消融（GSO-30）

| 变体 | CD ↓ (10⁻³) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----------|--------|--------|---------|
| MV-SAM3D 基线 | 45.8876 | 13.2421 | 0.8051 | 0.2795 |
| A: 几何先验+外观仿射 | 45.2204 (-1.45%) | 13.4546 (+1.60%) | 0.8103 (+0.65%) | 0.2876 (+2.90%) |
| A + softVH | 45.1879 (-1.52%) | 13.4470 (+1.55%) | 0.8104 (+0.66%) | 0.2877 (+2.93%) |
| B: 无先验轴向补偿 | 45.7427 (-0.32%) | 13.4985 (+1.94%) | 0.8109 (+0.72%) | 0.2758 (-1.32%) |
| A+B 几何融合 | **44.9530** (-2.04%) | **13.5549** (+2.36%) | **0.8135** (+1.04%) | **0.2735** (-2.15%) |

**关键发现**: 来源 A（几何先验）和来源 B（逐轴补偿）各自提升不同指标，融合后实现全面最优；仅用来源 A 时 LPIPS 略有下降，加入 B 后得到修正。

### Table 5: KnowledgeBank 基准结果（Terminal-Bench 2.0 & SWE-Bench Verified）

**Terminal-Bench 2.0（成功率 SR% / 平均步数 AS）**:

| 模型 | No Memory SR | ReasoningBank SR | KnowledgeBank SR |
|------|------------|-----------------|-----------------|
| Qwen-3.5-Flash | 48.5±1.6 | 52.8±2.0 | **55.8±1.8** |
| Qwen-3.5-Plus | 60.8±1.0 | 64.6±2.1 | **67.4±1.9** |
| Gemini-3-Flash | 55.6±1.5 | 59.0±1.2 | **61.8±1.7** |
| Gemini-3.1-Pro | 68.4±0.8 | 73.0±1.0 | **75.7±1.3** |

**SWE-Bench Verified（解决率 Resolve%）**:

| 模型 | No Memory | ReasoningBank | KnowledgeBank |
|------|-----------|--------------|--------------|
| Qwen-3.5-Flash | 67.0±1.1 | 70.8±1.5 | **73.4±1.2** |
| Qwen-3.5-Plus | 76.6±1.3 | 80.1±0.9 | **83.6±1.6** |
| Gemini-3-Flash | 74.2±0.8 | 78.0±1.5 | **80.4±1.4** |
| Gemini-3.1-Pro | 78.2±1.3 | 82.2±1.6 | **85.3±1.2** |

**关键发现**: KnowledgeBank 在所有模型上均优于 ReasoningBank，Terminal-Bench 平均提升 4.53 个百分点，SWE-Bench 平均提升 3.73 个百分点，且 AS（平均步数）更低，说明检索质量更高效。

### Table 6: 部署开销对比

| 模型 | 方法 | AS | Agent Token | 额外 Token | 总 Token | 延迟 | 存储 (MB) |
|------|------|-----|------------|-----------|---------|------|----------|
| Qwen-3.5-Flash | No Memory | 29.8 | 64.0k | 0.0k | 64.0k | 103.2s | 0.0 |
| Qwen-3.5-Flash | KnowledgeBank | 25.3 | 61.0k | 4.0k | 65.0k | 108.9s | 3.6 |
| Qwen-3.5-Plus | KnowledgeBank | 19.9 | 76.5k | 4.5k | 81.0k | 174.8s | 3.0 |

**关键发现**: KnowledgeBank 仅增加约 4k 额外 Token 和 5.7s 延迟，但通过减少 Agent Token 使用（步数更少），总体 Token 消耗与无记忆方案相近，存储开销仅 3-4MB。

### Table 7: KnowledgeBank 组件消融（Qwen-3.5-Flash）

| 变体 | Terminal SR | Terminal AS | SWE Resolve | SWE AS |
|------|------------|------------|------------|-------|
| Full | **55.8** | **38.8** | **73.4** | **39.9** |
| w/o Adm.（无准入控制） | 52.7 | 40.5 | 70.5 | 41.7 |
| w/o Gov.（无治理） | 51.6 | 41.4 | 69.1 | 43.0 |
| Sem. Ret.（仅语义检索） | 51.5 | 41.5 | 68.8 | 43.1 |

**关键发现**: 准入控制（Adm.）和治理机制（Gov.）各自贡献约 3 个百分点，精度导向检索（vs 纯语义检索）贡献约 1 个百分点；三者缺一不可。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| GSO-30 | 30 个物体 | Google Scanned Objects，多视图标准 benchmark | 重建评测 |
| RLBench | 14 个任务 | 仿真操作 benchmark | 操作策略评测 |
| Terminal-Bench 2.0 | - | 终端代理任务 | KnowledgeBank 评测 |
| SWE-Bench Verified | - | 软件工程任务 | KnowledgeBank 评测 |
| 真实世界 | 4 个任务 | 桌面操作（喷瓶、抽屉、开瓶、物体分类） | training-free 泛化评测 |

### 实现细节

- **重建后端**: MV-SAM3D（[[3D Gaussian Splatting]] 基础框架）
- **几何先验**: [[VGGT]]（来源 A）
- **评测指标**: Chamfer Distance（CD）、PSNR、SSIM、LPIPS（重建）；成功率、平均步数（操作）
- **训练方式**: GeoFuse-MV3D 和 KnowledgeBank 均为 training-free / zero-shot 推理
- **测试模型**: Qwen-3.5-Flash、Qwen-3.5-Plus、Gemini-3-Flash、Gemini-3.1-Pro

### 可视化结果

- GeoFuse-MV3D 在结构复杂物体（文具、家居用品）上改善最显著，尤其在视角稀疏时减少几何幻觉
- RLBench 中 Play_jenga、Insert_block 等精细任务对 3D 精度要求高，GeoFuse-MV3D 的改进直接体现在任务成功率上
- 真实世界 Open_drawer（40%）表现相对较低，可能受环境多样性和相机标定误差影响

---

## 批判性思考

### 优点

1. **保守融合策略**: GeoFuse-MV3D 不替换基线几何，而是保守修正，避免引入新误差，工程上更稳健
2. **可插拔设计**: 两大模块均可独立部署在已有 [[VLA]] pipeline 之上，无需重新训练整个系统
3. **多维度验证**: 同时在重建 benchmark、代理 benchmark 和真实机器人上验证，说服力较强
4. **低部署开销**: KnowledgeBank 增量 Token 和延迟开销极小（4k token，6s），实用性高

### 局限性

1. **重建提升幅度有限**: CD 仅改善 2.20%，PSNR 仅提升 2.36%，在任务成功率上的传导效果尚不明确
2. **KnowledgeBank 在 Push_block 上无提升**: 少数任务（push_block）未见改善，说明某些任务不依赖历史经验
3. **外部几何先验依赖**: 来源 A 依赖 [[VGGT]] 等外部重建模型，若先验质量差可能引入负迁移（Table 4 中单用 A 的 LPIPS 上升 2.90%）
4. **真实世界评测规模小**: 仅 4 个任务、3 次重复，统计显著性有限

### 潜在改进方向

1. **自适应融合权重**: 当前 $\alpha$ 为固定超参，可学习基于任务和重建置信度动态调整
2. **端到端联合优化**: 当前两模块独立优化，联合训练可能获得更好的协同效果
3. **KnowledgeBank 跨任务迁移**: 探索将不同物体/场景的经验跨任务迁移，扩大记忆泛化范围

### 可复现性评估

- [x] 代码开源（GitHub: AIGeeksGroup/GeneralVLA-2）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（消融实验完整）
- [x] 数据集可获取（GSO-30、RLBench、SWE-Bench 均为公开数据集）

---

## 关联笔记

### 基于

- [[VLA]]: 通用视觉-语言-动作模型基础框架
- [[3D Gaussian Splatting]]: GeneralVLA-2 的 3D 表示基础
- [[VGGT]]: GeoFuse-MV3D 的外部几何先验来源

### 对比

- [[VoxPoser]]: 基于语言模型的零样本操作规划方法
- [[MemoryVLA]]: 早期带记忆的 VLA 系统，KnowledgeBank 的前身

### 方法相关

- [[GeoFuse-MV3D]]: 本文核心重建模块
- [[Governed KnowledgeBank]]: 本文核心记忆治理模块
- [[3DAgent]]: GeneralVLA-2 的规划主干
- [[In-Context Retrieval]]: KnowledgeBank 精度导向检索的基础技术

### 硬件/数据相关

- [[RLBench]]: 仿真评测环境

---

## 速查卡片

> [!summary] GeneralVLA-2
> - **核心**: 几何感知重建 + 治理记忆双升级通用 VLA 系统
> - **方法**: GeoFuse-MV3D（遮罩验证+保守融合）+ Governed KnowledgeBank（质量元数据+精度检索）
> - **结果**: RLBench 覆盖 14 任务；GSO-30 CD -2.20%；KnowledgeBank 提升 SR +4.53%（Terminal-Bench）
> - **代码**: [GitHub](https://github.com/AIGeeksGroup/GeneralVLA-2)

---

*笔记创建时间: 2026-06-22*
