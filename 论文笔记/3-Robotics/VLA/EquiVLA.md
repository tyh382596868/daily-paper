---
title: "EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models"
method_name: "EquiVLA"
authors: [Thien-Loc Ha, Quang-Tan Nguyen, Trong-Bao Ho, Long Dinh, Minh Duc Nguyen, Gia-Binh Nguyen, Pham Tri Quang, Minh N. Vu, Duy M. H. Nguyen, An Thai Le, Ngo Anh Vien]
year: 2026
venue: arXiv
tags: [vla, equivariance, robot-manipulation, diffusion-policy, visuomotor-policy]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.19784
created: 2026-06-22
---

# 论文笔记：EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | VinRobotics（越南） |
| 日期 | June 2026 |
| 项目主页 | [equivla.github.io](https://equivla.github.io/) |
| 对比基线 | [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19784) |

---

## 一句话总结

> 首个将 SO(2) 旋转等变性端到端注入冻结 VLM + 扩散 Transformer 动作头的 VLA 框架，LIBERO 基准从 78.1% 提升至 92.6%，真机操作从 54% 提升至 72%。

---

## 核心贡献

1. **EquiPerceptor（Token 级 Frame Averaging）**: 将 [[Frame Averaging]] 从全局池化向量扩展到 ViT 空间 patch token 序列，无需修改冻结 VLM 权重即可产生近似等变的视觉表示
2. **EquiActor（等变 Flow-Matching DiT）**: 首个 [[SO(2)等变性|SO(2)-等变]] 扩散 Transformer 动作头，所有线性投影、注意力矩阵、动作编解码器均采用 G-可操控层
3. **端到端等变链**: 形式化证明两模块组合后近似 SO(2) 等变误差有界（Theorem 3），实验上将基线等变误差降低 27.3 倍

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在特定朝向下训练后，旋转工作空间或物体时性能显著下降——旋转后的场景对应旋转后的动作，这种几何对称性被完全忽略。模型将旋转相关的观测当作独立输入处理，浪费了大量数据。

### 现有方法的局限

- **VLM backbone**：图像处理时无旋转感知，[[ViT]] patch token 的空间索引在旋转下混乱
- **DiT 动作头**：标准 [[扩散变换器|DiT]] 使用无约束线性变换，无法保证旋转等变
- **现有等变策略**：要么从头训练小模型，要么用规范化包装器——均无法处理现代大规模 VLM + DiT 的完整流水线

### 本文的动机

几何结构与大规模预训练是互补的：在适当的架构位置注入等变性（视觉适配器 + 动作头），可以在保留 VLM 丰富视觉/语义先验的同时获得系统性旋转泛化能力。

---

## 方法详解

### 模型架构

EquiVLA 采用**冻结 VLM Backbone + 等变视觉适配器 + 等变动作头**的流水线架构：

- **输入**: 语言指令 $l$ + 外部相机观测 $o_t$ + 手腕图像 + 机器人状态 $s_t$
- **Backbone**: 冻结的 [[GR00T N1.5]] VLM
- **核心模块 1**: [[EquiPerceptor]] — Token 级 Frame Averaging，产生等变视觉 token
- **核心模块 2**: [[EquiActor]] — SO(2)-等变 [[Flow Matching]] DiT 动作头
- **对称群**: $C_8$（8 个离散旋转，间隔 45°）
- **输出**: 连续动作序列 $\hat{\mathbf{a}}_t$（相对或绝对控制模式）

> 🖼️ **Figure 1: EquiVLA 整体架构** — 图片见 [arXiv HTML](https://arxiv.org/html/2606.19784/x1.png)

**说明**: （上）EquiPerceptor 通过 Token 级 Frame Averaging 对冻结 VLM 的视觉流水线进行对称化，产生等变 token $\mathbf{z}^{\mathrm{eq}}$ 和不变 token $\mathbf{z}^{\mathrm{inv}}$，后者与手腕图像、语言指令一起送入冻结 VLM，产生 context token $\mathbf{z}^{\mathrm{ctx}}$，再由等变适配器与 $\mathbf{z}^{\mathrm{eq}}$ 融合，最终传入 EquiActor。（下）EquiActor 用 SO(2)-等变 DiT 替换标准 DiT 动作头，通过等变 cross/self-attention 层精炼状态与噪声动作 token，经 [[Flow Matching]] 解码输出动作 $\hat{\mathbf{a}}_t$。

---

### 核心模块

#### 模块 1：EquiPerceptor（Token 级 Frame Averaging）

**设计动机**：经典 [[Frame Averaging]] 只处理全局池化向量；ViT patch token 带有空间位置索引，直接平均会因坐标错位破坏结构。EquiPerceptor 通过**空间置换**对齐旋转后的 patch 位置，再配合**特征空间变换**，无需改动冻结权重即可在 token 序列上建立等变性。

**具体实现**：

- **等变 token 流**：对群 $G = C_u$ 的每个元素 $h$，旋转输入图像 $h \cdot \mathbf{x}$ 后送入冻结 ViT $f_\theta$，再用空间置换 $\tau(h^{-1})$ 将 patch 重映射回规范位置，同时用正则表示 $\rho_{\mathrm{reg}}(h^{-1})$ 变换特征维度，最后在群上平均
- **不变 token 流**：直接对旋转图像的 ViT 输出在群上平均（无位置变换），得到旋转不变的全局 token
- **等变适配器（Equivariant Adapter）**：用不变门控 $\alpha^{\mathrm{reg}}$ 融合等变流与来自 VLM 的 context token，所有门控输入均为不变量以保证等变输出

**Theorem 3（近似等变误差界）**：由于 patch 网格的离散化，空间置换 $\tau$ 只是近似——C4 子群（90°倍数）在正方形网格上**精确等变**（$\Delta=0$），C8 引入有界误差 $\|\varepsilon(g,\mathbf{x})\| \leq \Delta \cdot B(\mathbf{x})$，最大位移小于 patch 尺寸。

#### 模块 2：EquiActor（SO(2)-等变 Flow-Matching DiT）

**设计动机**：标准 [[扩散变换器|DiT]] 的线性层和注意力对旋转输入产生任意输出，无法保证动作随观测旋转而一致旋转。EquiActor 将所有可学习层替换为 G-可操控层（基于 escnn 库），使整个去噪过程满足等变性。

**具体实现**：

- **等变注意力**：用几何内积 $\langle q, k \rangle = \sum_g q[g] \cdot k[g]$ 计算注意力分数，正则表示的正交性保证得分为 G-不变量
- **Q/K/V 投影**：全部采用正则到正则的等变线性层
- **动作表示**：将 7 维机器人动作分解为不可约表示的直和，区分相对控制和绝对控制两种模式
- **架构规模**：16 个等变 DiT 层，192 个通道，从头训练（无法复用预训练权重）
- **训练目标**：标准 [[Flow Matching]] 均方误差目标，已证明对 SO(2) 不变

---

## 关键公式

### 公式 1：[[Frame Averaging|等变 Token 流]]

$$
\mathbf{z}^{\mathrm{eq}}(\mathbf{x}) = \frac{1}{|G|} \sum_{h \in G} \bigl[\tau(h^{-1}) \otimes \rho_{\mathrm{reg}}(h^{-1})\bigr] \cdot f_\theta(h \cdot \mathbf{x})
$$

**含义**：对每个旋转 $h$ 变换输入图像后过 ViT，再联合施加空间置换和特征变换后在群上平均，得到近似等变的 token 序列。

**符号说明**：
- $G = C_u$：$u$ 个离散旋转构成的循环群
- $\tau(h^{-1})$：将旋转后的 patch 中心重映射回规范网格的空间置换
- $\rho_{\mathrm{reg}}(h^{-1})$：正则表示，在特征维度上做循环置换
- $f_\theta$：冻结的 ViT（不更新权重）
- $\otimes$：空间索引置换与特征变换的联合作用

### 公式 2：[[Frame Averaging|不变 Token 流]]

$$
\mathbf{z}^{\mathrm{inv}}(\mathbf{x}) = \frac{1}{|G|} \sum_{h \in G} f_\theta(h \cdot \mathbf{x})
$$

**含义**：直接在群上平均 ViT 输出，不做位置修正，得到旋转不变的全局 token（送入冻结 VLM）。

**符号说明**：
- 与公式 1 符号相同，区别在于无 $\tau$ 和 $\rho_{\mathrm{reg}}$ 变换

### 公式 3：[[EquiPerceptor|等变适配器融合]]

$$
\mathbf{z}^{\mathrm{out}}_{\mathrm{eq}} = \bm{\alpha}^{\mathrm{reg}} \odot \mathbf{W}_s(\mathbf{s}^{\mathrm{inv}} \otimes \mathbf{1}_{|G|}) + (\mathbf{1} - \bm{\alpha}^{\mathrm{reg}}) \odot \mathbf{W}_g(\tilde{\mathbf{z}}^{\mathrm{eq}})
$$

**含义**：用不变门控权重 $\bm{\alpha}^{\mathrm{reg}}$ 在等变 token 流和 VLM context token 之间做加权融合，门控输入全为不变量以保证输出等变性。

**符号说明**：
- $\bm{\alpha}^{\mathrm{reg}}$：不变门控，展开为正则表示维度
- $\mathbf{s}^{\mathrm{inv}}$：由 VLM context token 聚合的不变摘要
- $\mathbf{1}_{|G|}$：扩展向量，将不变量复制到正则维度
- $\mathbf{W}_s, \mathbf{W}_g$：可学习的等变线性层
- $\odot$：逐元素乘

### 公式 4：[[SO(2)等变性|动作表示分解]]

$$
g \cdot \mathbf{a}_t = \begin{cases}
(\rho_1^3 \oplus (\rho_1 \oplus \rho_0) \oplus \rho_0)(g)\, \mathbf{a}_t & \text{绝对控制} \\[4pt]
P^{-1}(\rho_0^6 \oplus \rho_1^4 \oplus \rho_2)(g)\, P\, \mathbf{a}_t & \text{相对控制}
\end{cases}
$$

**含义**：将机器人动作按物理语义分解为 SO(2) 不可约表示的直和——向量量（平移/旋转）用频率 1 的 $\rho_1$ 表示，标量量（夹爪宽度、高度）用 $\rho_0$ 表示，以保证等变变换在物理上正确。

**符号说明**：
- $\rho_0$：平凡表示（不变标量，如夹爪宽度）
- $\rho_1$：频率 1 不可约表示（2D 向量旋转，如 xy 平移）
- $\rho_2$：频率 2 不可约表示（相对控制中的旋转分量）
- $\rho_1^3$：绝对控制末端执行器 6D 旋转（3 个 2D 对）
- $P$：从原始动作坐标到不可约表示分解的换基矩阵

### 公式 5：[[SO(2)等变性|端到端近似等变]]

$$
\hat{\mathbf{a}}_t(g \cdot \mathbf{o}_t,\, \rho_s(g)\mathbf{s}_t) \approx \rho_a(g) \cdot \hat{\mathbf{a}}_t(\mathbf{o}_t,\, \mathbf{s}_t) \quad \forall\, g \in C_u
$$

**含义**：EquiVLA 的整体性质——输入观测旋转 $g$ 后，输出动作以对应表示 $\rho_a(g)$ 变换，近似误差由 Theorem 3 的 $\Delta$ 界定。

**符号说明**：
- $\mathbf{o}_t$：相机观测
- $\rho_s(g)$：状态的群变换（旋转机器人状态）
- $\rho_a(g)$：动作的群变换（公式 4 定义的分解）
- $\approx$：近似等号，精确度取决于 $\Delta$

### 公式 6：[[Flow Matching|Flow-Matching 训练目标]]

$$
\mathcal{L} = \mathbb{E}_{k, \bm{\epsilon}, \mathbf{a}_t} \left[ \left\lVert \mathbf{v}_\theta(\mathbf{a}^k_t, k, \mathbf{z}^{\mathrm{out}}, \mathbf{z}^s_t) - (\mathbf{a}_t - \bm{\epsilon}) \right\rVert^2 \right]
$$

**含义**：Flow Matching 均方误差目标，预测从噪声 $\bm{\epsilon}$ 到真实动作 $\mathbf{a}_t$ 的速度场。已证明此目标对 SO(2) 不变（Proposition 7），等变模型最小化同一损失。

**符号说明**：
- $\mathbf{v}_\theta$：等变 DiT 预测的速度场
- $\mathbf{a}^k_t$：扩散步 $k$ 处的噪声动作
- $\bm{\epsilon} \sim \mathcal{N}(0, I)$：采样噪声
- $\mathbf{z}^{\mathrm{out}}$：EquiPerceptor 输出的视觉条件
- $\mathbf{z}^s_t$：机器人状态 token

### 公式 7：[[Frame Averaging|近似等变误差界]]（Theorem 3）

$$
\tilde{\mathbf{z}}^{\mathrm{eq}}(g \cdot \mathbf{x}) = \tilde{A}(g) \cdot \tilde{\mathbf{z}}^{\mathrm{eq}}(\mathbf{x}) + \bm{\varepsilon}(g, \mathbf{x}), \quad \left\lVert \bm{\varepsilon}(g, \mathbf{x}) \right\rVert \leq \Delta \cdot B(\mathbf{x})
$$

**含义**：量化近似空间置换 $\tilde{\tau}$ 引入的等变误差上界——最大位移 $\Delta$ 为次 patch 量级，乘以当前输入的 ViT 输出范数 $B(\mathbf{x})$。

**符号说明**：
- $\tilde{A}(g)$：$\tau$ 近似导致的实际变换矩阵
- $\Delta$：最大表示缺陷（representation defect），C4 时为 0，C8 时有界
- $B(\mathbf{x}) = \max_h \lVert f_\theta(h \cdot \mathbf{x}) \rVert$：群轨道上 ViT token 范数的最大值

---

## 关键图表

### Figure 2：真机任务展示

> 🖼️ **Figure 2: 五个 Mobile ALOHA 真机任务** — 图片见 [arXiv HTML](https://arxiv.org/html/2606.19784)

**说明**：展示五个真机任务的初始和目标状态：(a) Banana in Pot、(b) Block Storing、(c) House Building、(d) Letter Aligning、(e) Shorts Folding。Letter Aligning 是朝向敏感任务（+30pp），House Building 是精细组装任务（+35pp），Shorts Folding 是朝向无关任务（±0pp）。

### Figure 3：真机操作序列

> 🖼️ **Figure 3: 五个 Mobile ALOHA 任务的真机执行关键帧** — 图片见 [arXiv HTML](https://arxiv.org/html/2606.19784)

**说明**：每个任务展示从初始化到完成的六个关键帧，直观展示 EquiVLA 在不同类型操作任务上的泛化能力。

---

### Table 1：LIBERO 基准成功率（%）

| Method | 控制模式 | LIBERO-10 | Goal | Obj | Spat | Avg |
|--------|----------|-----------|------|-----|------|-----|
| π₀ † | Rel. | 73.0 | 93.0 | 86.0 | 90.0 | 86.0 |
| OpenVLA † | Rel. | 55.0 | 79.2 | 88.4 | 84.7 | 76.8 |
| SmolVLA | — | 61.0 | 61.4 | 66.0 | 74.0 | 65.6 |
| GR00T N1.5 | Rel. | 72.0 | 75.0 | 83.4 | 82.0 | 78.1 |
| GR00T N1.5 + EquiActor | Rel. | 82.6 | 88.0 | 95.2 | 98.2 | 91.0 |
| **EquiVLA** | **Rel.** | **87.6** | **89.4** | **98.0** | **95.4** | **92.6** |
| GR00T N1.5 | Abs. | 52.0 | 55.2 | 74.6 | 68.6 | 62.6 |
| GR00T N1.5 + EquiActor | Abs. | 63.0 | 70.0 | 79.4 | 82.0 | 73.6 |
| **EquiVLA** | **Abs.** | **73.6** | **70.4** | **83.0** | **77.6** | **76.1** |

**关键发现**：EquiVLA（相对控制）以 92.6% 平均成功率超越 [[GR00T N1.5]] 基线 14.5pp；单独添加 EquiActor 已带来 12.9pp 提升，EquiPerceptor 再贡献 1.6pp。

---

### Table 2：CALVIN ABCD→D 多步任务成功率

| Method | T1 | T2 | T3 | T4 | T5 | 平均序列长度 |
|--------|----|----|----|----|----|----|
| HULC †⋆ | 88.9 | 73.3 | 58.7 | 47.5 | 38.3 | 3.07 |
| MoDE †⋆ | 97.1 | 92.5 | 87.9 | 83.5 | 77.9 | 4.39 |
| GR00T N1.5 | 89.0 | 79.2 | 68.7 | 59.4 | 48.5 | 3.45 |
| GR00T N1.5 + EquiActor | 93.7 | 85.8 | 77.8 | 70.1 | 61.9 | 3.89 |
| **EquiVLA** | **95.0** | **88.5** | **81.1** | **73.8** | **64.3** | **4.03** |

**关键发现**：EquiVLA 将平均序列长度从 3.45 提升至 4.03（+0.58），Task 5 成功率从 48.5% 提升至 64.3%（+15.8pp），接近使用时序历史帧的多帧方法（MoDE 4.39）。

---

### Table 3：Mobile ALOHA 真机成功率（20 次试验）

| 模型 | Banana in Pot | Block Storing | House Building | Letter Aligning | Shorts Folding | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| EquiVLA | 15/20 | 11/20 | 10/20 | 19/20 | 17/20 | **72%** |
| GR00T N1.5 | 12/20 | 9/20 | 3/20 | 13/20 | 17/20 | 54% |

**关键发现**：总提升 +18pp；朝向敏感任务收益最大（Letter Aligning +30pp，House Building +35pp）；朝向无关任务（Shorts Folding）基本持平，说明等变性无负面代价。

---

### Table 4：数据效率消融（LIBERO 平均成功率）

| 方法 | 10% 数据 | 40% 数据 | 100% 数据 |
|------|:---:|:---:|:---:|
| GR00T N1.5 | 58.4 | 73.9 | 78.1 |
| GR00T N1.5 + EquiActor | 58.8 | 84.1 | 91.0 |
| **EquiVLA** | **60.2** | **84.5** | **92.6** |

**关键发现**：等变性在数据受限时尤为突出——40% 数据时 EquiActor 已超越全量数据的基线（84.1% > 78.1%）；10% 时收益相对较小（+1.8pp），在完整数据时收益最大（+14.5pp）。

---

### Table 5：对称群选择消融（LIBERO 成功率 vs. 推理延迟）

| 群 | LIBERO-10 | Goal | Obj | Spat | Avg | ms/step |
|-----|-----------|------|-----|------|-----|---------|
| C₄ | 82.5 | 94.0 | 94.5 | 95.5 | 91.6 | 161 |
| **C₈** | **87.5** | **89.5** | **98.0** | **95.5** | **92.6** | **194** |
| C₁₆ | 87.0 | 95.4 | 98.2 | 96.4 | 94.3 | 243 |

**关键发现**：C8 是精度-延迟的最佳权衡（92.6%，194ms/step）；C16 再提升 1.7pp 但延迟增加 25%；相比基线 64ms/step，Frame Averaging 需要 8 次 ViT 前向，是主要延迟来源。

---

## 实验

### 数据集与平台

| 数据集/平台 | 规模 | 特点 | 用途 |
|-------------|------|------|------|
| LIBERO | 4 套任务，50 个任务 | 桌面操作，4 个难度套件 | 仿真基准测试 |
| CALVIN ABCD→D | 连续多步任务 | 跨场景泛化，最多 5 步链 | 长时序仿真测试 |
| Mobile ALOHA | 5 个真机任务，20 次试验/任务 | 双臂移动操作平台 | 真机验证 |

### 实现细节

- **VLM Backbone**: 冻结的 [[GR00T N1.5]]（全程不更新权重）
- **EquiActor**: 16 层等变 DiT，192 通道，从头训练
- **对称群**: $C_8$（8 个离散 45° 旋转）
- **等变库**: escnn（G-可操控层实现）
- **推理延迟**: ~194 ms/step（基线 ~64 ms/step）
- **等变误差降幅**: 基线 7.754 → EquiActor 0.837 → EquiVLA 0.284（降低 27.3×）

### 关键定性观察

在朝向敏感任务（Letter Aligning：把字母牌放到特定朝向；House Building：精细组装）上提升最显著；朝向无关任务（Shorts Folding）提升为零，证明等变约束在对称任务上不带额外代价。

---

## 批判性思考

### 优点

1. **理论严谨**：提供了完整的等变性分析（Proposition 1/5/6/7，Theorem 3，Corollary 9），误差有界而非只做实验声称
2. **免训练 VLM**：完全冻结 VLM backbone，保留大规模预训练语义先验，工程落地友好
3. **通用框架**：两个模块可独立使用（实验证明 EquiActor 单独已带来 12.9pp 提升），适用于多种 VLA 架构

### 局限性

1. **对称性范围窄**：仅处理 SO(2) 平面旋转；真实 3D 操作（倾斜、翻转）需要 SO(3)/SE(3) 扩展，复杂度大幅增加
2. **推理延迟高**：Token 级 Frame Averaging 需要 $|G|=8$ 次完整 ViT 前向，延迟从 64ms 增至 194ms（+3×），对实时系统是挑战
3. **动作头无法迁移预训练权重**：G-可操控层的约束与标准 DiT 架构不兼容，EquiActor 只能从头训练，损失了潜在的预训练收益

### 潜在改进方向

1. **随机 Frame Averaging**：每次推理只采样部分群元素以降低延迟，作者已提及为 future work
2. **SO(3) 扩展**：通过球谐函数表示或四元数等变层处理完整 3D 旋转
3. **等变知识蒸馏**：将预训练 DiT 权重蒸馏入 G-可操控层，解决动作头冷启动问题

### 可复现性评估

- [x] 代码开源（equivla.github.io，escnn 库）
- [ ] 预训练模型（未公开 EquiActor 权重）
- [x] 训练细节完整（论文附录有架构和超参）
- [x] 数据集可获取（LIBERO、CALVIN、Mobile ALOHA 均公开）

---

## 关联笔记

### 基于

- [[GR00T N1.5]]: 冻结 VLM backbone，EquiVLA 在其上加等变适配器
- [[Frame Averaging]]: EquiPerceptor 的核心技术，扩展到 token 序列

### 对比

- [[GR00T N1.5]]: 主要基线，EquiVLA 在全部三个 benchmark 上超越
- [[SmolVLA]]: LIBERO 对比基线
- [[pi0.md|π₀]]: LIBERO 对比基线

### 方法相关

- [[SO(2)等变性]]: 核心几何约束
- [[Flow Matching]]: EquiActor 的训练范式
- [[扩散变换器|扩散 Transformer]]: EquiActor 替换的动作头架构
- [[ViT]]: EquiPerceptor 操作的视觉 backbone 组件

### 硬件/数据相关

- [[ALOHA|Mobile ALOHA]]: 真机实验平台，双臂移动操作

---

## 速查卡片

> [!summary] EquiVLA
> - **核心**: 首个端到端 SO(2) 等变 VLA 框架，冻结 VLM + 等变视觉适配器 + 等变 DiT 动作头
> - **方法**: EquiPerceptor（Token 级 Frame Averaging）+ EquiActor（等变 Flow-Matching DiT）
> - **结果**: LIBERO 92.6%（+14.5pp），CALVIN 4.03 avg（+0.58），真机 72%（+18pp）
> - **代码**: [equivla.github.io](https://equivla.github.io/)

---

*笔记创建时间: 2026-06-22*
