---
title: "Latent Spatial Memory for Video World Models"
method_name: "Mirage"
authors: [Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, Bohan Zhuang]
year: 2026
venue: arXiv
tags: [video-world-model, spatial-memory, video-generation, 3d-consistency, diffusion-model, camera-control, scene-generation]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.09828
created: 2026-06-13
---

# 论文笔记：Latent Spatial Memory for Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University, Microsoft Research, Adelaide University, Monash University |
| 日期 | June 2026 |
| 项目主页 | [aka.ms/latent-spatial-memory](https://aka.ms/latent-spatial-memory) |
| 对比基线 | [[Voyager]], [[Spatia]], [[VMem]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09828) / [Code](https://aka.ms/latent-spatial-memory) |

---

## 一句话总结

> Mirage 将视频世界模型的空间记忆从 RGB 像素空间迁移到 VAE latent 空间，以 55× 更低显存和 10.57× 更快速度实现跨帧一致的无边界场景漫游。

---

## 核心贡献

1. **Latent Spatial Memory 表示**: 以 $(\mathbf{p}_i, \mathbf{f}_i)$ 形式在世界坐标下缓存 VAE latent token，彻底消除 RGB 点云方案中的 rasterise-and-encode 往返开销。
2. **高效三阶段流水线（初始化 → 读出 → 更新）**: 通过深度引导反投影建立 3D cache，z-buffer 直接读出到 latent 空间作为 [[ControlNet]] 条件，无需额外编码器桥接。
3. **动态物体过滤 + 两阶段训练**: 用开放词汇分割器剔除运动区域避免几何污染，Stage 1 冻结主干训练侧分支，Stage 2 以 rank-64 [[LoRA]] 联合微调，在 WorldScore 和 RealEstate10K 上均达 SOTA。

---

## 问题背景

### 要解决的问题

现有[[视频生成模型|视频世界模型]]（如 Voyager、WonderJourney）在长时 rollout 中无法维持跨 chunk 的空间一致性：每个视频片段独立生成，当相机轨迹回到已访问区域时，场景外观与之前不一致。

### 现有方法的局限

基于 RGB 点云的空间记忆（如 Voyager、Spatia）在每步条件化时需要：
1. **光栅化**：将 3D 彩色点投影回 2D 图像（像素空间）
2. **重新编码**：调用 VAE 编码器将像素图压缩回 latent

这带来两个严重问题：
- **计算瓶颈**：VAE 编码代价随帧数线性增长，$\Theta(\Phi_{\mathcal{E}}(H,W))$ 级别开销
- **累积误差**：反复的像素 → latent → 像素循环积累量化噪声，语义信息逐渐退化

### 本文的动机

[[扩散模型|视频扩散模型]]的骨干网络天然在 latent 空间运算。如果将空间记忆也存储为 latent 特征，读出时就无需任何桥接编码器——读出的特征天然与骨干网络同分布，可直接以 ControlNet 方式注入。

---

## 方法详解

### 模型架构

![Figure 1: Mirage 生成的几何一致视频示例](https://arxiv.org/html/2606.09828/x1.png)

**说明**: 给定单张输入图像和用户指定的相机轨迹，Mirage 通过在 latent 空间直接缓存 3D 信息保持空间一致性。

Mirage 采用**三阶段 3D 缓存**架构：

- **输入**: 初始帧 $I_0$（图像）+ 相机内外参序列 $\{K^t, \mathbf{E}^t\}$
- **Backbone**: [[Wan2.2-TI2V]] (Wan2.2-TI2V-5B，[[Flow Matching]]训练目标)
- **核心模块**: [[Latent Spatial Memory]] — 世界坐标下的 latent 特征点云
- **条件注入**: [[ControlNet]] 风格侧分支，输入为 latent-resolution 的读出帧
- **输出**: 几何一致的长序列视频帧

### 核心模块

#### 模块 1：Latent Spatial Memory 表示

**设计动机**: 利用 [[VAE]] latent 空间的紧凑语义表示，取代低效的 RGB 点云

**具体实现**:

- 记忆表示为点对集合 $\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}$，其中 $\mathbf{p}_i \in \mathbb{R}^3$ 为世界坐标，$\mathbf{f}_i \in \mathbb{R}^C$ 为 VAE latent 特征
- 对比 RGB 记忆 $\mathcal{M}_{\text{rgb}} = \{(\mathbf{p}_i, \mathbf{c}_i)\}$（$\mathbf{c}_i \in [0,1]^3$），latent 维度更高但保留了更完整的语义

#### 模块 2：初始化（Initialization）

**设计动机**: 将首帧图像投影到 3D 世界坐标系，建立初始记忆

**具体实现**:
- 用 VAE 编码器编码 $I_0$ 得到 latent map $\mathbf{z} \in \mathbb{R}^{C \times h \times w}$
- 用 [[DepthAnything]] 估计像素级深度 $D(u,v)$
- 通过深度引导反投影（depth-guided back-projection）将每个 latent cell 提升到 3D：每个格点对应一个 $(\mathbf{p}_{uv}, \mathbf{F}_{uv})$ 对

#### 模块 3：读出（Readout）

**设计动机**: 将缓存的 3D latent 投影回目标相机视角，作为生成条件

**具体实现**:
- 将 cache 中所有点投影到目标相机平面（在 **latent 分辨率** 下操作）
- 用 [[z-buffer]] 深度测试，对每个 latent cell 选取最近点：$i^t(u,v) = \mathop{\mathrm{arg\,min}}_{i \in \Omega^t(u,v)} [\mathbf{E}^t \mathbf{p}_i]_z$
- 直接读出对应 latent 特征 $\hat{\mathbf{z}}^t(u,v) = \mathbf{F}_{i^t(u,v)}$，**无需 VAE 编码桥接**

#### 模块 4：更新（Cache Update）

**设计动机**: 将新生成帧的观测融入记忆，扩展已知场景范围

**具体实现**:
- 重新编码生成帧得到 clean VAE latents
- 用开放词汇实体提取器 + 视频分割模型过滤**动态物体**（人、车等）和天空区域
- 将合格 token 反投影加入 cache：$\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{p}_{uv}, \mathbf{F}_{uv})\}_{(u,v) \in \Lambda^t}$

#### 模块 5：两阶段训练

**Stage 1（侧分支预热）**:
- 冻结主干 [[视频扩散模型|扩散模型]]和 VAE
- 只训练 ControlNet 风格侧分支，学习将 latent cache 读出注入生成过程

**Stage 2（联合微调）**:
- 附加 rank-64 [[LoRA]] 适配器到自注意力投影层
- 联合优化侧分支与 LoRA，使用[[Flow Matching]] 目标函数

---

## 关键公式

### 公式 1：[[Latent Spatial Memory|记忆表示]]

$$
\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}, \quad \mathbf{p}_i \in \mathbb{R}^3,\ \mathbf{f}_i \in \mathbb{R}^C
$$

**含义**: 记忆为世界坐标点与 latent 特征的配对集合，取代 RGB 点云

**符号说明**:
- $\mathbf{p}_i$: 第 $i$ 个点在世界坐标系中的 3D 位置
- $\mathbf{f}_i$: 对应的 VAE latent 特征向量，维度 $C$（通常 $C=48$）

---

### 公式 2：[[深度引导反投影|Latent 初始化]]

$$
\mathbf{p}_{uv} = \pi^{-1}(u, v, D(u,v); K, \mathbf{E}), \quad \mathbf{F}_{uv} = \mathbf{z}[:,v,u]
$$

**含义**: 将 latent grid 的每个格点 $(u,v)$ 和其对应深度 $D(u,v)$ 反投影到世界坐标，同时读取对应 latent 特征

**符号说明**:
- $\pi^{-1}$: 相机反投影函数（pinhole 模型）
- $D(u,v)$: 深度估计值（来自 DepthAnything）
- $K, \mathbf{E}$: 相机内参矩阵和外参（世界到相机变换）
- $\mathbf{z}[:,v,u]$: VAE latent map 在位置 $(v,u)$ 的特征向量

---

### 公式 3：[[Pinhole Camera Model|针孔相机反投影]]

$$
\pi^{-1}(u, v, d; K^\ell, \mathbf{E}) = \mathbf{E}^{-1} \begin{bmatrix} d(K^\ell)^{-1}[u+\tfrac{1}{2},\ v+\tfrac{1}{2},\ 1]^\top \\ 1 \end{bmatrix}\Bigg|_{1:3}
$$

**含义**: 将 latent 分辨率下的像素坐标加偏移 $\tfrac{1}{2}$（对齐到格点中心）后，乘以深度标量，再经相机逆变换映射到世界坐标

**符号说明**:
- $K^\ell = \mathrm{diag}(w/W, h/H, 1)K$: latent 分辨率内参（将原图内参缩放到 latent 尺寸 $h \times w$）
- $d$: 该点深度值
- $\mathbf{E}^{-1}$: 相机外参逆矩阵（相机坐标 → 世界坐标）

---

### 公式 4：[[z-buffer|深度测试读出]]

$$
i^t(u,v) = \mathop{\mathrm{arg\,min}}_{i \in \Omega^t(u,v)} [\mathbf{E}^t \mathbf{p}_i]_z, \quad \hat{\mathbf{z}}^t(u,v) = \mathbf{F}_{i^t(u,v)}
$$

**含义**: 对目标相机 $t$ 下的每个 latent cell $(u,v)$，在投影到该位置的所有记忆点 $\Omega^t(u,v)$ 中选取**深度最小**（最近）的点，取其 latent 特征作为读出值

**符号说明**:
- $\Omega^t(u,v)$: 在第 $t$ 帧相机下投影到格点 $(u,v)$ 的所有记忆点索引集合
- $[\mathbf{E}^t \mathbf{p}_i]_z$: 将世界坐标点 $\mathbf{p}_i$ 变换到第 $t$ 帧相机坐标后的 z 分量（深度）
- $\hat{\mathbf{z}}^t$: 读出的 latent 条件图，维度与 backbone latent 一致

---

### 公式 5：[[Cache Update|Cache 更新]]

$$
\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{p}_{uv}, \mathbf{F}_{uv})\}_{(u,v) \in \Lambda^t}
$$

**含义**: 将新观测到的合格 latent token（排除动态物体和天空）反投影后并入全局记忆

**符号说明**:
- $\Lambda^t$: 第 $t$ 帧中满足过滤条件的有效 latent 格点集合（动态物体和天空被排除）
- $\mathbf{p}_{uv}$: 由深度估计反投影得到的 3D 坐标
- $\mathbf{F}_{uv}$: 重新编码生成帧后提取的 clean latent 特征

---

### 公式 6：[[RGB 点云|RGB 读出对比（基线）]]

$$
\hat{\mathbf{z}}^t = \mathcal{E}(\mathrm{Rasterise}(\mathcal{M}_{\text{rgb}}; \mathbf{E}^t, K^t))
$$

**含义**: 现有 RGB 点云方案需要先光栅化到像素图，再经 VAE 编码器 $\mathcal{E}$ 压回 latent，开销是 Mirage 的约 55×

**符号说明**:
- $\mathrm{Rasterise}(\cdot)$: 将 RGB 点云投影渲染到 2D 图像的光栅化操作
- $\mathcal{E}$: VAE 编码器，计算复杂度 $\Theta(\Phi_{\mathcal{E}}(H,W))$

---

## 关键图表

### Figure 1: Mirage 生成示例

![Figure 1: 输入单张图像和相机轨迹生成的几何一致视频](https://arxiv.org/html/2606.09828/x1.png)

**说明**: 给定单张输入图像和用户指定的相机轨迹，Mirage 通过在 latent 空间缓存 3D 信息维持空间一致性。支持任意长轨迹的无缝场景漫游。

---

### Figure 2: Latent Spatial Memory vs. RGB 点云对比

![Figure 2: Mirage 的 latent 空间记忆与 RGB 点云记忆的对比图示](https://arxiv.org/html/2606.09828/x2.png)

**说明**: **上方（RGB 点云方案）**：每步条件化需要光栅化（Rasterise）→ VAE 编码的完整往返，计算代价高昂。**下方（Mirage）**：latent 特征直接存储在 3D 点上，读出后直接注入 backbone，消除编码开销。

---

### Figure 3: Mirage 整体框架

![Figure 3: Mirage 的三阶段流水线——初始化、读出、更新](https://arxiv.org/html/2606.09828/x3.png)

**说明**: Mirage 的完整流水线：(1) 从 $I_0$ 编码 VAE latent 并经深度反投影初始化 3D cache；(2) 将 cache 投影到目标相机得到读出帧，经 [[ControlNet]] 侧分支注入生成；(3) 生成后重新编码，过滤动态物体，更新 cache。

---

### Figure 4: 开放域视频对比

![Figure 4: 室外和自然场景的开放域生成对比](https://arxiv.org/html/2606.09828/x4.png)

**说明**: 在 RealEstate10K 训练分布之外的户外和自然场景上测试，Mirage 相比 Voyager、Spatia、VMem 保持更强的几何一致性，展示了泛化能力。

---

### Figure 5: 效率随 rollout 进度的变化

![Figure 5: 单帧 cache 读出时间（左）和 cache 显存占用（右）随轨迹长度的变化](https://arxiv.org/html/2606.09828/x5.png)

**说明**: 在 NVIDIA H100 上测量。随着 rollout 进行，RGB 点云的显存线性增长，而 Mirage 的 latent cache 显存增速仅为前者的 $\approx 1/55$。读出时间也随点数增加而更平缓。

---

### Figure 6: RealEstate10K 视频对比

![Figure 6: RealEstate10K 轨迹上 Voyager、Spatia、VMem 和 Mirage 的对比](https://arxiv.org/html/2606.09828/x6.png)

**说明**: 每个块对应一条 RealEstate10K 轨迹，行分别为 Voyager、Spatia、VMem 和 Mirage。Mirage 在纹理细节和几何一致性上均优于基线。

---

### Figure 7: 闭环回访对比

![Figure 7: 相机轨迹回到出发点时的闭环一致性测试](https://arxiv.org/html/2606.09828/x7.png)

**说明**: 闭环测试中，相机轨迹逐渐返回起始点。Mirage 凭借持久化的 latent cache，在回访已见区域时重现一致的场景外观，而基线方法出现明显不一致。

---

### Figure 8: 额外室内场景对比

![Figure 8: 复杂室内轨迹下的额外视频对比](https://arxiv.org/html/2606.09828/x8.png)

**说明**: 在一个具有挑战性的室内轨迹上，Mirage 在完整轨迹中维持一致的布局，展示了在复杂遮挡和视角变化下的鲁棒性。

---

### Table 1: WorldScore 全面评测

**世界模型方法（World Model Methods）**

| Method | Avg↑ | Static↑ | Dynamic↑ | Cam Ctrl↑ | Obj Ctrl↑ | Content↑ | 3D Cons↑ | Photo Cons↑ | Style Cons↑ |
|--------|------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| WonderJourney | 54.19 | 63.75 | 44.63 | 84.60 | 37.10 | 35.54 | 80.60 | 79.03 | 62.82 |
| InvisibleStitch | 51.95 | 61.12 | 42.78 | 93.20 | 36.51 | 29.53 | 88.51 | 89.19 | 32.37 |
| WonderWorld | 61.79 | 72.69 | 50.88 | 92.98 | 51.76 | 71.25 | 86.87 | 85.56 | 70.57 |
| Voyager | 66.08 | 77.62 | 54.53 | 85.95 | 66.92 | 68.92 | 81.56 | 85.99 | 84.89 |
| FlashWorld | 60.23 | 70.85 | 49.60 | 84.43 | 50.28 | 56.54 | 85.87 | 86.72 | 79.36 |
| LucidDreamer | 59.84 | 70.40 | 49.28 | 88.93 | 41.18 | 75.00 | 90.37 | 90.20 | 48.10 |
| Spatia | 69.73 | 72.63 | 66.82 | 75.66 | 52.32 | 69.95 | 86.40 | 89.10 | 80.09 |

**通用视频生成模型（Foundation Video Models）**

| Method | Avg↑ | Static↑ | Dynamic↑ | Cam Ctrl↑ | Obj Ctrl↑ | Content↑ | 3D Cons↑ | Photo Cons↑ | Style Cons↑ |
|--------|------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| VideoCrafter2 | 50.03 | 52.57 | 47.49 | 28.92 | 39.07 | 72.46 | 65.14 | 61.85 | 43.79 |
| EasyAnimate | 52.25 | 52.85 | 51.65 | 26.72 | 54.50 | 50.76 | 67.29 | 47.35 | 73.05 |
| Allegro | 53.64 | 55.31 | 51.97 | 24.84 | 57.47 | 51.48 | 70.50 | 69.89 | 65.60 |
| CogVideoX-I2V | 60.64 | 62.15 | 59.12 | 38.27 | 40.07 | 36.73 | 86.21 | 88.12 | 83.22 |
| Vchitect-2.0 | 40.38 | 42.28 | 38.47 | 26.55 | 49.54 | 65.75 | 41.53 | 42.30 | 25.69 |
| LTX-Video | 55.99 | 55.44 | 56.54 | 25.06 | 53.41 | 39.73 | 78.41 | 88.92 | 53.50 |
| Wan2.1 | 55.21 | 57.56 | 52.85 | 23.53 | 40.32 | 45.44 | 78.74 | 78.36 | 77.18 |
| **Mirage (Ours)** | **70.36** | **73.60** | **67.11** | 55.36 | **74.17** | 42.09 | **92.21** | **93.95** | **96.91** |

**关键发现**: Mirage 在综合得分（70.36）、3D 一致性（92.21）和光度一致性（93.95）上均为最优。动态对象控制（74.17）远超所有方法。相机控制（55.36）略低于专注几何的方法（如 InvisibleStitch 93.20），因为 Mirage 更注重视频质量而非单纯相机精度。

---

### Table 2: RealEstate10K 定量结果

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | PSNRC↑ | SSIMC↑ | LPIPSC↓ |
|--------|------|------|-------|-------|-------|--------|
| SEVA | 13.07 | 0.515 | 0.445 | — | — | — |
| VMem | 14.62 | 0.522 | 0.426 | — | — | — |
| ViewCrafter | 15.78 | 0.580 | 0.396 | 14.79 | 0.481 | 0.365 |
| FlexWorld | 16.25 | 0.593 | 0.370 | 12.20 | 0.428 | 0.598 |
| Voyager | 17.79 | 0.636 | 0.297 | 17.66 | 0.540 | 0.380 |
| Spatia | 18.58 | 0.646 | 0.254 | 19.38 | 0.579 | 0.213 |
| **Mirage (Ours)** | 18.38 | **0.779** | **0.250** | **20.05** | **0.825** | **0.228** |

**关键发现**: Mirage 在 SSIM（结构相似性）和 LPIPS（感知质量）上大幅领先（SSIM 0.779 vs 次优 0.646，提升 20.6%）。闭环指标（PSNRC/SSIMC）同样最优，说明回访区域的记忆保真度最高。PSNR 略低于 Spatia（18.38 vs 18.58），但 SSIM 和 LPIPS 显著更优，说明 latent cache 保留了更多感知相关的高频细节。

---

### Table 3: 消融实验

| 变体 | Avg↑ | Static↑ | Dynamic↑ | 3D Cons↑ | Photo Cons↑ |
|-----|------|---------|---------|---------|-----------|
| **Mirage（完整）** | **70.36** | **73.60** | **67.11** | **92.21** | **93.95** |
| 替换为 RGB 点云 | 67.71 | 70.49 | 64.93 | 90.75 | 91.10 |
| 像素分辨率提升 + 特征上采样 | 60.85 | 62.41 | 59.28 | 84.90 | 79.81 |
| 无动态物体过滤 | 61.20 | 62.69 | 59.70 | 80.88 | 76.10 |
| 单阶段训练 | 63.18 | 65.15 | 61.20 | 87.11 | 84.47 |

**关键发现**:
- **Latent vs RGB**: 使用 RGB 点云使 Avg 下降 2.65，3D 一致性下降 1.46，证明在 latent 空间操作的质量优势
- **动态物体过滤最关键**: 去除后 3D 一致性从 92.21 暴跌到 80.88（-11.33），说明运动物体的错误几何是最大噪声来源
- **两阶段训练**: 比单阶段高 7.18 Avg，预热侧分支后再解冻主干可避免早期训练不稳定

---

### Table 4: 深度估计源敏感性

| 深度估计器 | Avg↑ | 3D Cons↑ | Photo Cons↑ |
|----------|------|---------|-----------|
| DepthAnything 3（默认） | 70.36 | 92.21 | 93.95 |
| MapAnything | 69.66 | 91.89 | 93.32 |
| UniDepth | 69.13 | 91.63 | 92.79 |

**关键发现**: 方法对深度源不敏感，不同估计器带来的差异 <1.5%，说明 latent cache 的鲁棒性不依赖于特定的深度模型。

---

### Table 5: 深度下采样方法

| 下采样方式 | Hole Rate↓（空洞率）|
|----------|-----------|
| 双线性插值（默认） | 42.53% |
| 最近邻 | 47.78% |
| 面积池化 | 53.72% |
| 中值池化 | 52.22% |

**关键发现**: 双线性插值在将像素分辨率深度下采样到 latent 分辨率时产生最少的空洞（未被 cache 覆盖的 latent cell），保持最完整的场景覆盖。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RealEstate10K | 约 76K 视频 | 室内场景，带相机轨迹标注 | 训练 + 测试 |
| WorldScore | 多维度基准 | 评估静态/动态/3D一致性/相机控制等 | 测试（zero-shot） |
| 开放域（Out-of-domain） | 用户自选 | 户外、自然场景 | 泛化测试 |

### 实现细节

- **Backbone**: [[Wan2.2-TI2V]] (Wan2.2-TI2V-5B，[[Flow Matching]]训练目标，VAE stride s=16，C=48 通道)
- **侧分支**: [[ControlNet]] 风格，接受 latent-resolution 读出帧作为条件
- **LoRA rank**: 64（附加到自注意力 Q/K/V/O 投影）
- **深度估计**: [[DepthAnything3]]（默认），也支持 MapAnything / [[UniDepth]]
- **动态过滤**: [[Qwen3-VL]] 开放词汇实体提取 + [[SAM2|SAM 3]] 视频分割
- **相机位姿**: [[VIPE]] 从训练视频中提取相机位姿和深度
- **硬件**: 训练用 32 张 A100，推理用单张 H100（单机 UniPC 调度器，40 步采样）
- **训练**: 两阶段（Stage 1 冻结主干，Stage 2 联合微调）
- **输出分辨率**: 704×1280 像素，33 帧/chunk（latent 9×44×80）

### 效率分析

| 指标 | RGB 点云 | Mirage（latent） | 提升 |
|-----|---------|----------------|------|
| 端到端生成速度 | 基线 | **10.57×** 更快 | — |
| GPU 显存峰值 | 基线 | **55×** 更低 | $s^2 \cdot (3/C)$（$s=16, C=48$）|

渐近时间复杂度对比：
- RGB cache 读出：$\Theta(N \log N + HW) + \Theta(\Phi_{\mathcal{E}}(H,W))$（含 VAE 编码）
- Latent cache 读出：$\Theta(N \log N + hw)$（无需编码）

显存比例：$s^2 \cdot (3/C)$，其中 $s=16$（空间下采样率），$C=48$（latent 通道数），约 $256 \times (3/48) \approx 1/16$（实测因批量和 overhead 约 $1/55$）

---

## 批判性思考

### 优点

1. **理论一致性优雅**: 将 cache 与 backbone 统一在同一 latent 流形上，无需桥接编码器——设计哲学上追求"消除阻抗不匹配"
2. **效率突破实用化门槛**: 55× 显存降低使长轨迹生成在单张 H100 上可行，为世界模型走向实际部署奠定基础
3. **消融设计严谨**: 逐一验证了记忆空间、动态过滤、训练策略的贡献，结论可信

### 局限性

1. **动态物体的完全排除**: 运动物体被过滤出 cache，在人流密集、车辆频繁的场景中，记忆会出现大量"空洞"，世界连续性受损
2. **深度估计引入的几何噪声**: 虽然对深度源不敏感，但单目深度估计的尺度模糊和边界失真仍然是潜在误差来源
3. **相机控制精度偏低**: WorldScore 相机控制分项（55.36）明显低于专注几何一致性的方法（InvisibleStitch 93.20），说明在精确相机跟随上仍有差距

### 潜在改进方向

1. **动态物体的时序一致性建模**: 引入对象级跟踪，让动态实体也有自己的 latent 记忆，而非完全排除
2. **结合度量深度或 SLAM**: 用带尺度的深度替换单目估计，提升多次回访的几何精度
3. **在线记忆压缩**: 随轨迹增长，cache 点数不断增加；可引入点云聚合/剔除策略，限制 cache 规模

### 可复现性评估

- [x] 代码开源（Microsoft GitHub）
- [ ] 预训练模型（论文中提及，待发布确认）
- [x] 训练细节完整（消融和超参数均有描述）
- [x] 数据集可获取（RealEstate10K 公开）

---

## 关联笔记

### 基于

- [[ControlNet]]: 侧分支条件注入架构来源
- [[Flow Matching]]: 扩散模型的训练目标
- [[DepthAnything]]: 深度估计 backbone
- [[VAE]]: latent 空间编解码基础

### 对比

- [[Voyager]]: 主要对比基线，同样使用 RGB 点云记忆的世界模型
- [[Spatia]]: 综合性能最接近的竞争方法
- [[VMem]]: Video Memory 方案的代表性基线

### 方法相关

- [[Latent Spatial Memory]]: 本文核心贡献
- [[LoRA]]: 两阶段训练中的高效微调方法
- [[z-buffer]]: 深度测试机制用于 latent 读出
- [[视频扩散模型]]: backbone 类型

### 硬件/数据相关

- [[RealEstate10K]]: 训练和评测数据集
- [[WorldScore]]: 多维度评测基准

---

## 速查卡片

> [!summary] Latent Spatial Memory for Video World Models (Mirage)
> - **核心**: 将 RGB 点云式空间记忆迁移到 VAE latent 空间，消除 rasterise-and-encode 往返
> - **方法**: 3D latent cache + ControlNet 读出注入 + 动态物体过滤 + 两阶段 LoRA 训练
> - **结果**: WorldScore 综合 70.36（SOTA），RealEstate10K SSIM 0.779（SOTA）；速度 10.57×，显存 1/55
> - **代码**: Microsoft GitHub（具体链接待论文附录）

---

*笔记创建时间: 2026-06-13*
