---
title: "3D-Belief: Embodied Belief Inference via Generative 3D World Modeling"
method_name: "3D-Belief"
authors: [Yifan Yin, Zehao Wen, Jieneng Chen, Zehan Zheng, Nanru Dai, Haojun Shi, Suyu Ye, Aydan Huang, Zheyuan Zhang, Alan Yuille, Jianwen Xie, Ayush Tewari, Tianmin Shu]
year: 2026
venue: arXiv
tags: [world-model, 3d-gaussian-splatting, embodied-ai, belief-inference, scene-diffusion, object-navigation, semantic-3d]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.11367v1
created: 2026-05-13
---

# 论文笔记：3D-Belief: Embodied Belief Inference via Generative 3D World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Johns Hopkins University / Lambda / University of Cambridge |
| 日期 | May 2026 |
| 项目主页 | https://3d-belief.github.io/ |
| 对比基线 | [[DFoT]], [[NWM]], [[VGGT]], [[GEN3C]], [[SPOC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.11367) / [Code](https://github.com/3D-Belief/3d-belief) / [HF](https://huggingface.co/datasets/SCAI-JHU/3d-belief) |

---

## 一句话总结

> 3D-Belief 把"世界模型"从追求像素逼真重新定义为"在 3D 空间中做信念推断"——用[[场景级 3D 扩散]]在线预测并更新带语义的 [[3D Gaussian Splatting|3DGS]] 场景，对未观测区域采样多种假设，从而支撑部分可观测下的具身导航。

---

## 核心贡献

1. **概念框架：具身信念推断**：主张面向具身智能体的生成式世界模型应当维护并更新关于"未观测 3D 世界"的信念，并提炼出 4 项必备能力——空间一致的场景记忆、多假设信念采样、序贯信念更新、语义引导的未观测区域预测。
2. **3D-Belief 模型**：一个生成式 3D 世界模型，用[[场景级 3D 扩散]]从部分观测中直接预测可执行(actionable)、带语义、带不确定性的 3DGS 场景信念，支持恒定单步开销的自回归在线更新。
3. **3D-CORE 基准**：在 AI2-THOR / ProcTHOR 上构建的对象级与场景级 3D 想象评测基准，含对象补全、房间补全、对象恒存(object permanence)三类任务。
4. **全面评测**：在 2D 视觉质量、3D 场景想象、模拟+真机开放词表物体导航三个层面均显著优于基线（模拟导航成功率 59.17% vs 最佳基线 45%，且每步零 VLM token 开销）。

---

## 问题背景

### 要解决的问题

部分可观测环境下的具身智能体（如在陌生房间里找物体的机器人）需要对**已观测**和**未观测**区域都形成一个"信念"，并随着新观测的到来不断更新这个信念，进而据此规划行动。当前的[[视频扩散模型]]/新视角合成模型擅长渲染逼真的新视角或预测未来帧，但只关注视觉逼真度，不提供具身智能体真正需要的结构化不确定性。

### 现有方法的局限

> 作者用 Table 4 系统对比了各类方法在 5 项能力（场景记忆 / 2D 想象 / 3D 想象 / 序贯更新 / 语义）上的覆盖情况：

- **位姿条件视频生成**（[[DFoT]]、[[NWM]]）：2D 想象强，但没有显式 3D 信念，长 horizon 下出现语义漂移和几何/位姿漂移；
- **带视觉生成的 3D 缓存**（Persistent EWM、[[GEN3C]]）：能序贯记忆，但 3D 主要用于重建而非想象；
- **新视角合成 / 前馈 3DGS**（[[MVSplat]]、[[ViewCrafter]]、[[VGGT]]）：维护显式 3D，但缺少不确定性感知的多假设想象；
- **语义 3D 表示**（[[LERF]]、[[ConceptGraph]]）：只对已观测内容附语义，不生成未观测区域；
- 最接近的 Lyra 2.0 有 3D 想象但不支持序贯更新和语义。没有一个方法同时具备全部 5 项能力。

### 本文的动机

"预测像素 ≠ 推断底层 3D 世界。" 既然具身规划需要在 3D 中评估候选路径并做语义查询，那么世界模型就应该直接输出**显式的 3D 表示**，而且这个表示要把"已观测的确定部分"和"想象出来的不确定部分"区分开、可被多次假设采样、可被新观测序贯修正。作者用 [[POMDP]] 的信念状态形式化这一点，再用[[扩散模型]]在 3DGS 场景上做生成来实现"多假设"。

---

## 方法详解

### 模型架构

3D-Belief 采用 **共享 [[U-ViT]] 主干 + 双头（几何头 / 语义头）的[[场景级 3D 扩散]]** 架构（见 Figure 2、Figure 3）：

- **输入**：一段自我中心 RGB 观测历史 $o^{1:t}$ 及其相机位姿 $\phi$，以及目标视角（target view）位姿与噪声；
- **Backbone**：共享 [[U-ViT]]（参考 Song et al. 2025 / Hoogeboom et al. 2024 的设计），从上下文视图(context view)和带噪目标视图中提取特征；
- **几何头 — MVS 风格 3DGS 预测器**：内含[[多视图 Transformer]]保证跨视图几何一致性 + [[代价体|Cost Volume]] 模块在离散化深度候选上存储跨视图特征匹配分数，由此预测深度并把每像素提升(lift)为 [[3D Gaussian Splatting|3DGS]] 原语；
- **语义头**：把主干特征做轻量线性投影得到每像素语义特征图，通过从 [[CLIP]] 风格嵌入做蒸馏来训练，使测试时可用文本做开放词表查询；
- **输出**：一个带语义嵌入的 3DGS 场景 $z^t$，可微渲染出 RGB、深度、语义特征图，既支持"心智模拟(mental simulation)"也支持基于语言的规划；
- **更新方式**：自回归 $z^{t+1} \sim p(z^{t+1}\mid o^{t+1}, z_o^t)$——只在短 horizon 配对上训练，测试时通过持续更新做 test-time scaling，每步计算开销恒定。

### 核心模块

#### 模块1：3D 信念表示（observed / imagined 分解）

**设计动机**：直接预测完整 3D 场景代价太高，改为预测一个 3D 表示 $z^t = \phi(s^t)$。选 [[3D Gaussian Splatting|3DGS]] 是因为它(i)是显式 3D 表示，(ii)每个原语可携带语义特征嵌入，(iii)渲染快、适合下游具身任务。

**具体实现**：场景由一组高斯原语 $\{g_k\}$ 构成，每个 $g_k=(\mu_k, \Sigma_k, \alpha_k, S_k, e_k)$（位置、协方差、不透明度、球谐外观、语义嵌入）。整个信念分解为 $z^t = z_o^t \cup z_i^t$——观测部分 $z_o^t$ 由历史决定（确定），想象部分 $z_i^t$ 是扩散采样出的多种假设（不确定）。

#### 模块2：场景级 3D 扩散

**设计动机**：与逐帧在像素/潜空间做扩散不同，3D-Belief 在**整个 3D 场景（高斯原语）上**做[[扩散模型|扩散]]，从而显式鼓励全局几何与多视图一致性。

**具体实现**：对目标视图的观测加噪（前向过程），网络 $\Phi_\theta$ 以上下文观测、带噪目标观测、时间步 $\tau$ 及各自位姿为条件，预测**中间干净的 3DGS 场景** $z_{\tau-1}$，再用渲染器 $\mathcal{G}$ 把它渲染回观测作为重建目标。即"预测场景 → 渲染 → 与真值比对"的 $x_0$-prediction 式扩散。

#### 模块3：信念引导的规划（推理时扩展）

**设计动机**：有了显式带语义的 3D 信念，规划就是在 3D 里"试走"。

**具体实现**：每一步——(1) 从自我中心 RGB 历史+位姿更新信念；(2) 采样多个似然的 3D 信念假设；(3) 沿候选路径渲染做评估；(4) 用开放词表语义查询给候选打分，选最优路径执行。不需要每步调用 VLM，故 token 开销为 0。

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

### 公式1：[[POMDP|信念状态定义]]

$$
b(s^t) = P(s^t \mid o^{1:t})
$$

**含义**：智能体在第 $t$ 步的信念，是在给定观测历史下对世界状态的后验分布。

**符号说明**：
- $s^t$：第 $t$ 步的世界（场景）状态
- $o^{1:t}$：从开始到第 $t$ 步的观测序列

### 公式2：[[POMDP|序贯信念更新]]

$$
b(s^{t+1}) = \sum_{s^t} P(s^{t+1} \mid o^{t+1}, s^t)\, b(s^t)
$$

**含义**：新观测 $o^{t+1}$ 到来时，对旧信念做转移并归一化，得到新信念——刻画"流式部分观测如何修正信念"。

**符号说明**：
- $P(s^{t+1}\mid o^{t+1}, s^t)$：在新观测和旧状态下到达新状态的概率
- $b(s^t)$：上一步信念

### 公式3：[[场景级 3D 扩散|3D 表示的自回归更新]]

$$
z^{t+1} \sim p(z^{t+1} \mid o^{t+1}, z_o^t)
$$

**含义**：不直接对 $s^t$ 建模，而是对 3D 表示 $z^t=\phi(s^t)$ 建模；新表示由"新观测 + 旧的已观测部分 $z_o^t$"条件采样得到。该形式带来三点好处：(i) 只需短 horizon 配对训练；(ii) 测试时可持续更新做 test-time scaling；(iii) 每步开销恒定。渲染为 $\hat{o} = \mathcal{G}(z^t, \theta)$，输出 RGB / 深度 / 语义图。

**符号说明**：
- $z^t = \{g_k\}_{k=1}^K$，$g_k=(\mu_k,\Sigma_k,\alpha_k,S_k,e_k)$：第 $k$ 个高斯原语（位置/协方差/不透明度/球谐/语义嵌入）
- $z^t = z_o^t \cup z_i^t$：observed（确定）+ imagined（不确定）分解
- $\mathcal{G}$：可微渲染器，$\theta$ 为渲染相机参数

### 公式4：[[扩散模型|扩散前向过程]]

$$
q(o_\tau^{\text{trgt}} \mid o_{\tau-1}^{\text{trgt}}) = \mathcal{N}\!\left(o_\tau^{\text{trgt}};\ \sqrt{1-\beta_\tau}\, o_{\tau-1}^{\text{trgt}},\ \beta_\tau I\right)
$$

**含义**：对目标视图观测按方差表 $\beta_\tau$ 逐步加高斯噪声。

**符号说明**：
- $o_\tau^{\text{trgt}}$：第 $\tau$ 个扩散步的（带噪）目标视图观测
- $\beta_\tau$：第 $\tau$ 步的噪声方差

### 公式5：[[场景级 3D 扩散|扩散反向过程（预测干净 3DGS 并渲染）]]

$$
z_{\tau-1} = \Phi_\theta\!\left(o^{\text{ctxt}},\, o_\tau^{\text{trgt}};\ \tau,\, \phi^{\text{ctxt}},\, \phi^{\text{trgt}}\right)
$$

$$
\hat{o}_{\tau-1}^{\text{trgt}} = \mathcal{G}\!\left(z_{\tau-1},\, \phi^{\text{trgt}}\right)
$$

**含义**：去噪网络 $\Phi_\theta$ 以上下文观测、带噪目标观测、时间步及双方位姿为条件，预测中间干净的 3DGS 场景 $z_{\tau-1}$；再渲染回目标视图得到去噪后的观测，作为重建监督目标。

**符号说明**：
- $o^{\text{ctxt}}$ / $o^{\text{trgt}}$：上下文 / 目标视图观测；$\phi^{\text{ctxt}}$ / $\phi^{\text{trgt}}$：对应相机位姿
- $\Phi_\theta$：去噪网络（共享 U-ViT + 双头）；$\mathcal{G}$：可微渲染器

### 公式6：[[感知图像相似度|RGB 重建损失]]

$$
\mathcal{L}_{\text{rgb}} = \sum_{v \in \{\text{trgt},\,\text{ctxt}\}} \big\| I(\hat{o}^v) - I(o^v) \big\|_2^2
$$

**含义**：渲染图像与真值图像的 L2 重建损失，同时约束上下文视图和目标视图。

**符号说明**：
- $I(\cdot)$：取观测的 RGB 图像分量
- $\hat{o}^v$ / $o^v$：视图 $v$ 的渲染观测 / 真值观测

### 公式7：[[CLIP|语义蒸馏损失]]

$$
\mathcal{L}_{\text{sem}} = \sum_{v \in \{\text{trgt},\,\text{ctxt}\}} \frac{1}{M}\sum_{j=1}^{M} \big\| S(\hat{o}^v)(\mathbf{u}_j) - f_{\text{clip}}\big(\pi(o^v, \mathbf{u}_j)\big) \big\|_2^2
$$

**含义**：让渲染出的每像素语义特征图对齐由 [[CLIP]] 提取的 patch 特征（监督式蒸馏），从而测试时可做文本查询。

**符号说明**：
- $S(\hat{o}^v) \in \mathbb{R}^{H\times W\times d}$：渲染语义特征图；$\mathbf{u}_j$：第 $j$ 个采样像素
- $f_{\text{clip}}(\cdot)$：CLIP 风格特征提取器；$\pi(o^v, \mathbf{u}_j)$：像素 $\mathbf{u}_j$ 对应的图像 patch；$M$：采样像素数

### 公式8：深度损失（可选，有真值深度时启用）

$$
\mathcal{L}_{\text{depth}} = \sum_{v \in \{\text{trgt},\,\text{ctxt}\}} \frac{1}{\sum_{\mathbf{u}} m^v(\mathbf{u})} \sum_{\mathbf{u}} m^v(\mathbf{u})\, \big| \hat{d}^v(\mathbf{u}) - d^v(\mathbf{u}) \big|
$$

**含义**：在有效掩码区域内对渲染深度做 L1 监督（仅在仿真等有真值深度时使用）。

**符号说明**：
- $\hat{d}^v$ / $d^v$：视图 $v$ 的渲染深度 / 真值深度
- $m^v(\mathbf{u}) \in \{0,1\}$：像素 $\mathbf{u}$ 是否有有效深度的掩码

> 总损失为 $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \mathcal{L}_{\text{sem}} (+ \mathcal{L}_{\text{depth}})$。

---

## 关键图表

<!-- 图片外链优先：arXiv HTML -->

### Figure 1: Key capabilities of a generative 3D belief model / 能力概览

![Figure 1](https://arxiv.org/html/2605.11367v1/x1.png)

**说明**：示意一个理想的生成式 3D 信念模型应具备的 4 项能力——多假设信念采样、序贯信念更新、空间一致的场景记忆、语义引导的未观测区域预测。

### Figure 2: 3D-Belief overview / 系统概览

![Figure 2](https://arxiv.org/html/2605.11367v1/x2.png)

**说明**：整体流程。输入自我中心观测历史 + 噪声，模型在线预测并更新带语义的 [[3D Gaussian Splatting|3DGS]] 场景表示，区分 observed / imagined 部分，渲染出 RGB / 深度 / 语义图供心智模拟与规划。

### Figure 3: Model architecture (3D Scene Diffusion) / 模型架构

![Figure 3](https://arxiv.org/html/2605.11367v1/x3.png)

**说明**：共享 [[U-ViT]] 主干 + 两个头。几何头是 MVS 风格 3DGS 预测器（[[多视图 Transformer]] + [[代价体|Cost Volume]]，离散深度候选上做跨视图匹配 → 深度 → 提升为高斯原语）；语义头是轻量线性投影 + 从 [[CLIP]] 蒸馏。扩散在整个 3D 场景上进行。

### Figure 4: Qualitative 2D visual quality / 2D 视觉质量定性对比

![Figure 4](https://arxiv.org/html/2605.11367v1/x4.png)

**说明**：与 [[DFoT]] / [[NWM]] 对比，基线出现语义漂移与几何/位姿漂移，3D-Belief 借助显式 3D 表示在已观测视角的重建上保持几何一致。

### Figure 5: 3D object imagination / 3D 对象想象定性结果

![Figure 5](https://arxiv.org/html/2605.11367v1/x5.png)

**说明**：在仅部分可见目标物体的情况下，模型补全其 3D 几何与外观，并保持语义结构正确。

### Figure 6: Belief-guided planning / 信念引导规划可视化

![Figure 6](https://arxiv.org/html/2605.11367v1/x6.png)

**说明**：展示想象区域（imagination regions）以及在想象渲染中检测到的目标物体，说明智能体如何在 3D 信念上评估候选路径。

### Figure 7: Belief updating during navigation / 导航中的信念更新

![Figure 7](https://arxiv.org/html/2605.11367v1/x7.png)

**说明**：初始信念与随导航推进不断被新观测修正后的信念对比，体现序贯更新能力。

### Figure 8: Qualitative results on RealEstate10K / RE10K 定性结果

![Figure 8](https://arxiv.org/html/2605.11367v1/x8.png)

**说明**：在真实相机轨迹数据 [[RealEstate10K]] 上的视觉预测定性结果。

### Figure 9: Qualitative results on AI2-THOR / AI2-THOR 定性结果

![Figure 9](https://arxiv.org/html/2605.11367v1/x9.png)

**说明**：在 [[AI2-THOR]] 导航数据上的定性结果。

### Table 1: 2D Visual Quality of Belief Prediction（AI2-THOR）

观测区域（条件于轨迹端点，预测中间新视角，有真值）：

| Method | LPIPS↓ | PSNR↑ | SSIM↑ |
|--------|--------|-------|-------|
| [[NWM]] | 0.1876 | 18.75 | 0.702 |
| [[DFoT]] | 0.1206 | 23.35 | 0.841 |
| **Ours (3D-Belief)** | **0.0502** | **28.81** | **0.928** |

想象场景（仅条件于起始帧，生成 FoV 之外的视角，用分布指标）：

| Method | FVD↓ | FID↓ |
|--------|------|------|
| [[NWM]] | 487.4 | 89.28 |
| [[DFoT]] | 429.7 | 72.82 |
| **Ours (3D-Belief)** | **271.8** | **47.24** |

**说明**：3D-Belief 在已观测视角重建和未观测区域想象两端都显著领先。

### Table 2: 3D-CORE Benchmark 结果

| Method | BEV IoU↑ | 3D IoU↑ | Chamfer↓ | SigLIP↑ | Recognition↑ | Obj. F1↑ | Occ. Acc.↑ | Occ. IoU↑ | LPIPS↓ | SigLIP↑ |
|--------|----------|---------|----------|---------|--------------|----------|------------|-----------|--------|---------|
| DFoT-[[VGGT]] | 0.362 | 0.243 | 0.830 | 0.798 | 0.767 | 0.531 | 0.252 | 0.110 | 0.555 | 0.907 |
| **3D-Belief** | **0.484** | **0.318** | **0.216** | **0.855** | **0.930** | **0.536** | **0.900** | **0.442** | **0.123** | **0.978** |

**说明**：对象补全（前 5 列：BEV/3D IoU、Chamfer、SigLIP、VLM 识别率）几何与语义均更好；房间补全（中间 3 列：对象 F1、占据精度、占据 IoU）对象 F1 持平但占据预测远超基线（0.900 vs 0.252）；对象恒存（后 2 列：LPIPS、SigLIP）长 horizon 大视角变化下一致性大幅领先（LPIPS 0.123 vs 0.555）。

### Table 3: Object Navigation 结果（模拟 + 真机）

**模拟（AI2-THOR，135 个未见房屋，开放词表物体导航）：**

| Metric | [[VGGT]] (frontier) | VGGT (Gemini 3.0) | DFoT-VGGT | NWM-VGGT | Gemini 3.0 | Qwen3-VL-8B | [[SPOC]] | **Ours** |
|--------|--------------------|--------------------|-----------|----------|------------|-------------|----------|----------|
| SR% ↑ | 27.50 | 25.00 | 26.05 | 25.00 | 45.00 | 18.33 | 31.67 | **59.17** |
| SPL% ↑ | 25.82 | 24.10 | 24.59 | 23.46 | 37.81 | 14.08 | 30.97 | **39.07** |
| SEL% ↑ | 22.66 | 22.66 | 23.43 | 20.76 | 41.47 | 16.94 | 30.56 | **40.24** |
| Token (/step) ↓ | 0 | 1448.02 | 1210.08 | 552.12 | 7512.70 | 220.29 | 0 | **0** |

**真机（Hello Robot Stretch，模拟公寓，全部未见物体；目标如 "red mug"）：**

| Metric | Gemini 3.0 | **Ours** |
|--------|------------|----------|
| SR% ↑ | 23.08 | **55.56** |
| SEL% ↑ | 13.55 | **35.91** |

**说明**：3D-Belief 模拟成功率 59.17%（最佳基线 Gemini 3.0 直接视觉导航 45%），且每步 0 VLM token；真机成功率 55.56% vs Gemini 3.0 的 23.08%，显示在传感器噪声下的真实迁移。

### Table 4: Capability Comparison Across Methods / 能力对比

| Models | Scene Mem. | 2D Imag. | 3D Imag. | Sequential | Semantic |
|--------|-----------|---------|---------|-----------|----------|
| [[DFoT]] | ✗ | ✓ | ✗ | ✓ | ✗ |
| [[NWM]] | ✗ | ✓ | ✗ | ✓ | ✗ |
| Persistent EWM | ✓ | ✓ | ✗ | ✓ | ✗ |
| [[GEN3C]] | ✓ | ✓ | ✗ | ✓ | ✗ |
| [[ViewCrafter]] | ✓ | ✓ | ✗ | ✗ | ✗ |
| [[MVSplat]] | ✓ | ✗ | ✗ | ✗ | ✗ |
| [[VGGT]] | ✓ | ✗ | ✗ | ✗ | ✗ |
| DFM | ✓ | ✓ | ✗ | ✓ | ✗ |
| Lyra 2.0 | ✓ | ✓ | ✓ | ✗ | ✗ |
| **3D-Belief** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**：3D-Belief 是唯一同时满足全部 5 项能力的方法。

### Table 5: RealEstate10K Visual Prediction 结果

| Method | Obs PSNR↑ | Obs SSIM↑ | Obs LPIPS↓ | Imag FVD↓ | Imag FID↓ |
|--------|-----------|-----------|------------|-----------|-----------|
| [[DFoT]] (pretrained) | 24.31 | 0.862 | 0.0892 | 392.1 | 58.24 |
| [[GEN3C]] (pretrained) | 25.48 | 0.891 | 0.0642 | 365.8 | 52.16 |
| **3D-Belief** | **27.89** | **0.916** | **0.0421** | **298.5** | **41.37** |

**说明**：在真实场景数据 [[RealEstate10K]] 上，对已观测视角重建和未观测区域想象均优于预训练的 DFoT / GEN3C。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[AI2-THOR]] (ProcTHOR houses) | 大量导航 episode | 程序化生成室内场景，自我中心 RGB + 已知相机位姿 + 真值 3D 几何/语义 | 训练 + 测试（Exp 1、Exp 2、Exp 3 模拟） |
| [[RealEstate10K]] | 真实房产视频相机轨迹 | 真实世界相机轨迹，无 3D 真值 | 测试（Exp，Table 5） |
| 3D-CORE（本文构建） | 233 + 263 + 474 = 970 个任务 | 基于 ProcTHOR，含对象补全 / 房间补全 / 对象恒存三类任务，带几何与语义真值 | 测试（Exp 2） |

### 实现细节

- **Backbone**：共享 [[U-ViT]]（沿用 Song et al. 2025 / Hoogeboom et al. 2024 设计）；几何头为 MVS 风格 3DGS 预测器（[[多视图 Transformer]] + [[代价体|Cost Volume]]），语义头为线性投影 + [[CLIP]] 蒸馏
- **训练**：在整个 3D 场景（高斯原语）上做扩散；损失 = RGB L2 + 语义蒸馏 L2（+ 可选深度 L1）；只在短 horizon 配对上训练，测试时自回归持续更新（论文 Appendix 7 给出完整扩散公式）
- **对比基线**：[[DFoT]]（history-guided video diffusion）、[[NWM]]（navigation world model）、[[VGGT]]（visual geometry grounded transformer，做 3D 重建/lifting）、[[GEN3C]]、[[SPOC]]（学习式导航策略）、Gemini 3.0 / Qwen3-VL-8B（VLM 智能体）
- **真机平台**：Hello Robot Stretch，模拟公寓 + 家居物体（训练时全部未见）

### 可视化结果

- 基线（DFoT/NWM）在长 horizon 出现明显语义漂移（物体身份变化）和几何/位姿漂移；3D-Belief 因为有显式 3DGS 表示，重建一致性与对象恒存上明显更稳；
- 对象/房间补全的可视化显示模型能对部分可见物体给出合理的 3D 完形，并对未观测房间布局做语义合理的预测；
- 导航可视化显示"想象区域 → 沿路径渲染 → 语义查询打分"的规划流程在多房间探索中有效。

---

## 批判性思考

### 优点

1. **问题重定义有洞见**：把世界模型从"像素逼真"转向"3D 信念推断"，并用 [[POMDP]] 信念状态 + 4 项能力清单把模糊的"世界模型应该长什么样"讲清楚了，Table 4 的能力对比一目了然。
2. **架构选择自洽**：用 3DGS 作为 3D 表示同时满足"显式 3D + 可带语义 + 渲染快"，扩散在场景上做又自然带来"多假设"，三者环环相扣。
3. **规划侧的实用性**：每步 0 VLM token 而成功率反超 Gemini 3.0 直接导航，说明显式 3D 信念确实能省掉昂贵的在线大模型推理，对真机部署友好。
4. **评测扎实**：自建 3D-CORE 覆盖几何/语义/长 horizon 三个维度，且有真机实验，不只是仿真刷点。

### 局限性

1. **静态世界假设**：模型假设场景静态，无法处理动态物体/被搬动的物体，这对很多具身任务是硬伤（作者也承认）。
2. **想象不可控**：多假设采样是无条件的，无法用语言/场景图等高层信息引导"想象什么"，作者列为 future work。
3. **依赖位姿真值**：训练（甚至推理）需要相机位姿；仿真里好办，真机用了 Stretch 的里程计，但更一般的场景下位姿估计误差会传播到 3DGS。
4. **3DGS 表示的代价/上限**：长时间序贯更新下高斯原语数量、内存与漂移如何 scale，论文给的是"恒定单步开销"的设计目标，长 horizon 真实表现需更多数据支撑。

### 潜在改进方向

1. 引入动态世界建模（时变高斯 / 4D 表示），支持物体运动与交互。
2. 用语言描述或场景图条件化 imagined 部分的采样，实现可控想象。
3. 联合估计相机位姿（类似 VGGT 的 pose-free 思路）以摆脱对位姿真值的依赖。
4. 把这套 3D 信念接到操作（manipulation）策略上，而不仅仅是导航。

### 可复现性评估

- [x] 代码开源（https://github.com/3D-Belief/3d-belief）
- [x] 预训练模型 / 数据（HF: SCAI-JHU/3d-belief）
- [ ] 训练细节完整（完整扩散/超参在 Appendix，正文较略）
- [x] 数据集可获取（AI2-THOR / RealEstate10K 公开，3D-CORE 随论文发布）

---

## 关联笔记

### 基于
- [[3D Gaussian Splatting]]：核心 3D 场景表示
- [[扩散模型]] / [[场景级 3D 扩散]]：生成多假设信念的机制
- [[POMDP]]：信念状态的形式化框架
- [[U-ViT]]：去噪网络主干

### 对比
- [[DFoT]]：history-guided 视频扩散，2D 想象但无显式 3D
- [[NWM]]：navigation world model，同上
- [[VGGT]]：前馈 3D 重建，无想象/无语义；也用作 DFoT-VGGT / NWM-VGGT 的 lifting 模块
- [[GEN3C]]：带 3D 缓存的视频生成
- [[MVSplat]] / [[ViewCrafter]]：前馈 3DGS / 新视角合成
- [[SPOC]]：学习式导航策略基线

### 方法相关
- [[多视图 Transformer]]：几何头核心组件
- [[代价体|Cost Volume]]：跨视图深度匹配
- [[CLIP]]：语义头蒸馏来源
- [[新视角合成]]：3D-Belief 在已观测区域的子任务

### 硬件/数据相关
- [[AI2-THOR]]：主要训练/评测仿真器
- [[RealEstate10K]]：真实场景评测数据
- [[Hello Robot Stretch]]：真机实验平台

---

## 速查卡片

> [!summary] 3D-Belief: Embodied Belief Inference via Generative 3D World Modeling
> - **核心**：把世界模型重定义为"3D 信念推断"——在线预测并序贯更新带语义、带不确定性的 3DGS 场景
> - **方法**：共享 U-ViT 主干 + MVS 风格 3DGS 几何头 + CLIP 蒸馏语义头，在整个 3D 场景上做扩散；规划时采样多假设信念、沿路径渲染、语义查询打分（每步 0 VLM token）
> - **结果**：2D 视觉质量、3D-CORE 基准全面领先；模拟物体导航 SR 59.17%（最佳基线 45%），真机 55.56%（Gemini 3.0 为 23.08%）
> - **代码**：https://github.com/3D-Belief/3d-belief

---

*笔记创建时间: 2026-05-13*
