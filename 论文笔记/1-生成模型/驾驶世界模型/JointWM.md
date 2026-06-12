---
title: "Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving"
method_name: "JointWM"
authors: [Lijun Zhou, Hongcheng Luo, Zhenxin Zhu, Cheng Chi, Mingfei Tu, Lei Gong, Zhanqian Wu, Kaixin Xiong, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, "et al. (37 authors, Xiaomi)"]
year: 2026
venue: arXiv
tags: [driving-world-model, video-diffusion, 3d-gaussian-splatting, feed-forward-reconstruction, rectified-flow, ode-distillation, autonomous-driving]
zotero_collection: 1-生成模型/驾驶世界模型
image_source: online
arxiv_html: https://arxiv.org/html/2605.18137v4
created: 2026-05-27
---

# 论文笔记：Xiaomi Auto World Model — Joint World Model for Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 小米汽车（Xiaomi Auto） |
| 日期 | May 2026（arXiv:2605.18137v4，提交 2026-05-18） |
| 项目主页 | <https://JointWM.github.io> |
| 对比基线（重建） | [[MVSplat]]、[[NoPoSplat]]、[[DepthSplat]]、[[STORM]]、[[DGGT]] |
| 对比基线（生成） | [[MagicDrive]]、[[Vista]]、[[DiVE]]、[[Genesis]]、[[Epona]]、[[UniScene]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.18137) / [PDF](https://arxiv.org/pdf/2605.18137) / [Project](https://JointWM.github.io) |

---

## 一句话总结

> 小米把 [[Feed-Forward 重建|前馈式 3DGS 重建]] 和 [[扩散变换器|DiT]] 视频生成捏成一个 [[Joint World Model|联合世界模型]]——重建端用稀疏 [[Sparse Scene Query|3D 场景查询]] 把 10 秒片段从「4 小时优化」压到「10 秒前馈」，生成端用 [[Rectified Flow|整流流]] + 三阶段 [[Causal Fine-tuning|因果微调]]（[[Teacher Forcing]]、[[ODE Distillation|ODE 蒸馏]]、[[DMD]]）把推理步数从 50 砍到 4，最后让重建的 [[3D Gaussian Splatting|高斯]] 渲染图回灌成生成端的条件，把长程漂移按死。

---

## 核心贡献

1. **WorldRec — 前馈稀疏查询重建**: 在世界坐标里初始化 $N$ 个稀疏 3D 查询，跨视角/跨时刻聚合 [[Cross-View Attention|多视图特征]]，由 MLP 一把吐出 [[3D Gaussian Splatting|3D 高斯]] 属性。Waymo 上 28.48 PSNR / 0.861 SSIM，**比 [[DGGT]] 高 1 PSNR，比逐场景优化方案快 1400×**（10s vs 4h）。
2. **WorldGen — 双向预训+因果蒸馏的 DiT**: 用 [[Rectified Flow|整流流]] 做全局双向预训练学时空分布，再切到 [[Causal Fine-tuning|因果掩码]] 微调，叠加 [[Teacher Forcing]] + [[ODE Distillation|ODE 蒸馏]] + [[DMD|分布匹配蒸馏]]。最终 4 步出 81 帧、**0.19 s/frame**，FID 7.04 / FVD 64.97。
3. **Joint World Model — 双向耦合**: WorldRec 给 WorldGen 喂「投影到目标视角的 RGB 先验」当条件；WorldGen 帮 WorldRec 增量扩张未观测区域的高斯——几何与生成相互约束，长程一致性显著超过两者单独使用。
4. **三大瓶颈一次性回应**: 把现有方案的「优化慢」「步数多」「重建/生成两张皮」三个老大难放进同一篇技术报告，并给出工业级量产指标（H20 上单视 0.19s、三视 0.46s/frame）。
5. **零样本跨域**: 在 Waymo 上训，**nuScenes 零样本** PSNR 26.54——基本跑赢前 SOTA 微调后的数。

---

## 问题背景

### 要解决的问题

[[驾驶世界模型|Driving World Model]] 同时被两类需求拉扯：感知/仿真侧想要 **可解释的显式 3D 表征**（[[3D Gaussian Splatting|3DGS]] 渲染、可编辑、可几何监督），数据生成侧想要 **强先验的视频生成**（长尾天气/动物/夜景/cut-in 都能造出来）。

现实是：单做 [[Feed-Forward 重建|重建]] 的方案不会"做梦"，只能复现已观测内容；单做 video diffusion 的方案没几何约束，越生越漂。

### 现有方法的局限

1. **逐场景优化重建（[[StreetGaussians]]、[[DrivingGaussian]]、OmniRe）**: 表征精度高，但单段 clip 训练 ~4 小时，**完全无法量产**。
2. **前馈重建（[[STORM]]、UFO、[[DGGT]]）**: 速度有了，几何一致性和泛化又差，密集逐像素高斯还很冗余。
3. **驾驶视频生成（[[GAIA-1]]、DriveDreamer、[[MagicDrive]]、GAIA-2、[[Vista]]）**: 没几何 prior，靠从零端到端预训练学世界规则代价巨大；推理动辄数百步，工业落地不友好。
4. **自回归长程生成（[[Epona]]）**: 缓解漂移但 1.06 s/frame 太慢；同时 [[Compounding Errors|暴露偏差]] 还在。
5. **重建+生成混合方案（NeoVerse 等）**: 通用域工作，**缺多相机一致性、ego-motion 控制、layout 控制** 这些驾驶专属设计。

### 本文的动机

观察：**几何重建提供"deterministic 锚点"，生成模型提供"分布外补全"**——两者的失效模式恰好互补。所以把它们装进一个回路里：

- 重建端必须**前馈、稀疏、可扩展**（否则被工业流水线拒绝）
- 生成端必须**因果、少步、长程稳**（否则被 latency / drift 拒绝）
- 接口必须**简单**：高斯渲染图直接做条件、新观测增量融入高斯缓存

---

## 方法详解

### 整体架构

JointWM 由三部分组成，按数据流可以画成：

- **输入**: 多视角图像序列 $\{I_c^t\}$、相机内外参、ego trajectory、3D bounding box layout、文本 prompt
- **WorldRec 分支**: 共享 backbone → [[Sparse Scene Query|稀疏 3D 查询]] → [[Cross-View Attention|跨视图聚合]] → MLP 解码 → [[3D Gaussian Splatting|3D 高斯]] $\mathcal{G}$
- **WorldGen 分支**: [[VAE]] + umT5 编码 → 条件 [[扩散变换器|DiT]] → 4 步采样 → 视频 latent → VAE 解码
- **联合接口**: $\mathcal{G}$ 渲染到目标视角得到 $I_{\text{ren}}$，与 ego trajectory / layout 一起拼成 WorldGen 条件；WorldGen 输出反过来通过 scene fusion 增量更新 $\mathcal{G}$ 的未观测区域

输出：单视 0.19 s/frame、三视 0.46 s/frame，最长 81 帧、~1 分钟驾驶视频。

### 模块 1: WorldRec — 稀疏查询 3DGS 前馈重建

**设计动机**: 摒弃 "per-pixel dense Gaussians" 那种把高斯当像素的做法，改成 **场景级稀疏查询**——一组可学习的 3D 锚点，从多视图特征里"自取所需"，再统一吐高斯属性。

**具体实现**:

1. **多尺度特征提取**: 共享 visual backbone 输出多尺度特征 $\{\mathbf{F}^t_{c,l}\}$，$c$ 为相机索引，$l$ 为尺度。
2. **3D 查询初始化**: 世界坐标系下散布 $N$ 个 [[Sparse Scene Query|稀疏 3D 查询]] $\mathbf{p} = [X, Y, Z]^\top$，初始位置覆盖驾驶 ROI（前方/两侧路面）。
3. **投影采样**: 每个查询用 [[相机投影|projection]] $\pi_c$ 投到各视图 $\mathbf{u}^{c,l} = \pi_c(\mathbf{p})$，[[Bilinear Interpolation|双线性插值]] 取特征 $\mathbf{f}^{c,l}$。
4. **可见性感知聚合**: 用模型预测的归一化权重 $w^{c,l}$（含可见性 + 特征质量）做跨视图/跨时刻加权融合 $\mathbf{q}_i = \sum_{c,l} w^{c,l}\mathbf{f}^{c,l}_i$。
5. **高斯属性解码**: MLP 一把出 $(\Delta\mathbf{p}_i, \mathbf{c}_i, \alpha_i, \mathbf{s}_i, \mathbf{r}_i)$，即位置偏移、RGB、不透明度、尺度、旋转四元数。
6. **可微渲染监督**: [[Differentiable Gaussian Rasterization|可微高斯光栅化]] 渲染目标视角，像素 + 感知双损失。

**核心妙处**: 稀疏查询本身就是「场景 token」，天然支持增量补充——这是后面 Joint 阶段做增量重建的关键设计前提。

### 模块 2: WorldGen — 双向预训 + 三阶段因果微调

**Backbone**: 标准 [[扩散变换器|DiT]] + [[VAE]] 视觉编码 + umT5 文本编码 + 数值嵌入（ego trajectory / camera params）+ 栅格化 layout map。

#### Stage 1: Bidirectional Pre-training

用 [[Rectified Flow|整流流]] 形式做 **全双向时空注意力** 预训练，让模型先把整段视频分布学透：

- 噪声/数据线性插值 $x_t = (1-t)z + tx_0$, $t \sim \mathcal{U}(0,1)$
- 监督 velocity field $v^*(x_t,t) = x_0 - z$
- 流匹配目标，全局上下文最大化

#### Stage 2: Causal Fine-tuning（三步走）

**Step 2a — [[Teacher Forcing|教师强制]]**: 把 attention 改成 [[非对称注意力掩码|因果掩码]] $M_{ij} = 0\ (j\le i)$ / $-\infty\ (j>i)$，**用 ground-truth 帧当历史 context**，学因果分布。

**Step 2b — [[ODE Distillation|ODE 蒸馏]]**: 把 50 步 teacher 的 [[probability-flow ODE|概率流 ODE]] 解蒸到 4 步 student：$\mathcal{L}_{\text{ODE}} = \|f_\varphi(x_T, K=4) - \text{sg}[\hat x_0^{\text{teacher}}]\|_2^2$，得到 ~12× 推理加速。

**Step 2c — [[DMD|Distribution Matching Distillation]]**: Teacher Forcing 训练时见的全是干净 GT，推理时见的是 student 自己上一步的"脏"输出——典型的 [[Compounding Errors|暴露偏差]]。DMD 让 student 用**自己生成的历史** $\hat x^{(<i)} = G_\varphi(\epsilon, c)$ 当 context，配合 KL 散度匹配数据分布，把 train/test gap 闭合。

**链式效果**: 双向预训提供强先验 → 因果掩码切换学时序 → 步数蒸馏砍 latency → 分布匹配修暴露偏差。**四步流水线，缺一不可**。

### 模块 3: Joint World Model — 双向耦合

#### 3a. WorldRec 适配：增量场景重建

新到来的图像走完查询管线后，**通过 [[Cross-Attention|交叉注意力]] 把新 token 融入缓存的高斯 token**，而不是从零重建。
- 已观测区域：refine（多视图证据精化）
- 未观测区域：expand（拓展几何边界）

#### 3b. WorldGen 适配：RGB 先验条件

把当前高斯 $\mathcal{G}$ 用相机参数渲染到 target view 得到 $I_{\text{ren}}$（往往有空洞、局部失真），作为**第四种条件模态**注入 DiT：

- 显式几何 scaffold：让生成尊重已有结构
- 空洞由 generation 来补：避免重建空洞带来的几何崩塌
- 联合训练时强制 photometric 一致

#### 3c. 闭环增益

- **长程时序一致性**：几何锚点抑制 autoregressive drift
- **多视图空间一致性**：渲染先验天然多视一致
- **多次推理稳定性**：相同高斯 prior 下，不同噪声种子生成结构骨架一致

---

## 关键公式

### 公式 1: [[相机投影|3D 查询的相机投影]]

$$
\mathbf{u}^{c,l} = \pi_c(\mathbf{p})
$$

**含义**: 把世界系下的 3D 查询点 $\mathbf{p}$ 通过相机 $c$ 的内外参 $\pi_c$ 投到第 $l$ 尺度特征图的 2D 像素坐标。

**符号说明**:
- $\mathbf{p} \in \mathbb{R}^3$: 世界坐标下的稀疏查询点
- $\pi_c(\cdot)$: 第 $c$ 个相机的投影函数（含内参 $K_c$、外参 $[R_c|t_c]$）
- $\mathbf{u}^{c,l}$: 该查询在第 $c$ 相机第 $l$ 尺度特征图上的浮点像素坐标

### 公式 2: [[Bilinear Interpolation|双线性特征采样]]

$$
\mathbf{f}^{c,l} = \mathrm{BilinearInterp}(\mathbf{F}^t_{c,l}, \mathbf{u}^{c,l})
$$

**含义**: 把浮点投影坐标处的特征用双线性插值"拿"下来，作为该查询从该视图该尺度看到的视觉证据。

**符号说明**:
- $\mathbf{F}^t_{c,l}$: 时间 $t$、相机 $c$、尺度 $l$ 的特征图
- $\mathbf{f}^{c,l} \in \mathbb{R}^D$: 采样得到的 $D$ 维特征向量

### 公式 3: [[Sparse Scene Query|可见性感知聚合]]

$$
\mathbf{q}_i = \sum_{c,l} w^{c,l}_i\, \mathbf{f}^{c,l}_i
$$

**含义**: 每个查询 $i$ 把所有视图所有尺度的采样特征做加权和。权重 $w^{c,l}_i$ 由模型预测，编码"该视图能不能看到这个点 + 看得清不清楚"。

**符号说明**:
- $w^{c,l}_i$: 归一化权重，$\sum_{c,l} w^{c,l}_i = 1$
- $\mathbf{q}_i$: 聚合后的查询 token，送入 MLP 解码器

### 公式 4: [[3D Gaussian Splatting|高斯属性解码]]

$$
(\Delta\mathbf{p}_i,\ \mathbf{c}_i,\ \alpha_i,\ \mathbf{s}_i,\ \mathbf{r}_i) = \mathrm{MLP}(\mathbf{q}_i)
$$

**含义**: 一个 MLP head 直接把 query token 解码成完整的 3D 高斯属性集合。

**符号说明**:
- $\Delta\mathbf{p}_i \in \mathbb{R}^3$: 相对初始位置的位移
- $\mathbf{c}_i \in \mathbb{R}^3$: RGB 颜色（也可用 [[球面谐波系数|SH]]）
- $\alpha_i \in [0,1]$: 不透明度
- $\mathbf{s}_i \in \mathbb{R}^3$: 三轴尺度
- $\mathbf{r}_i \in \mathbb{R}^4$: 旋转四元数

### 公式 5: [[Differentiable Gaussian Rasterization|重建损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{pixel}} + \lambda\, \mathcal{L}_{\text{perceptual}}
$$

**含义**: 像素级 L1/L2 + [[LPIPS]] 感知损失的加权和，作用于可微高斯渲染的输出图。

**符号说明**:
- $\mathcal{L}_{\text{pixel}}$: RGB 像素重建损失
- $\mathcal{L}_{\text{perceptual}}$: VGG/LPIPS 感知损失
- $\lambda$: 经验权重（论文未给具体值）

### 公式 6: [[Rectified Flow|整流流插值]]

$$
x_t = (1-t)\, z + t\, x_0,\qquad t \sim \mathcal{U}(0,1)
$$

**含义**: 在数据样本 $x_0$ 和高斯噪声 $z$ 之间做线性插值，作为扩散过程的中间样本。

**符号说明**:
- $x_0$: 真实视频 latent
- $z \sim \mathcal{N}(0,I)$: 标准高斯噪声
- $t$: 在 $[0,1]$ 均匀采样的"时间"

### 公式 7: [[Rectified Flow|速度场目标]]

$$
v^*(x_t, t) = \frac{\mathrm{d} x_t}{\mathrm{d} t} = x_0 - z
$$

**含义**: 整流流的关键观察——线性插值路径下，真实速度场恒等于 "数据 - 噪声"，是常数（沿 $t$ 不变）。

### 公式 8: [[Flow Matching|流匹配预训练损失]]

$$
\mathcal{L}_{\text{rf}} = \mathbb{E}_{t,\,x_0,\,z}\big[\|v_\theta(x_t, t, c) - (x_0 - z)\|_2^2\big]
$$

**含义**: Stage 1 双向预训练目标——让网络预测的 velocity $v_\theta$ 拟合真实速度 $x_0 - z$，$c$ 是所有条件（ego trajectory / camera / layout / text）。

### 公式 9: [[非对称注意力掩码|因果注意力掩码]]

$$
M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

**含义**: Stage 2a 切换到因果模式——第 $i$ 帧只能看历史帧 $j \le i$。这是 [[Causal Fine-tuning|因果微调]] 的几何核心。

### 公式 10: [[Teacher Forcing|教师强制损失]]

$$
\mathcal{L}_{\text{TF}} = \mathbb{E}_{t,\,\epsilon}\Big[\big\|\epsilon_\theta\big(x_t^{(i)}, t, c, x_{\text{GT}}^{(<i)}\big) - \epsilon\big\|_2^2\Big]
$$

**含义**: 用 ground-truth 历史 $x_{\text{GT}}^{(<i)}$ 作 context，监督模型 denoise 第 $i$ 帧。简单稳定，但训测分布不一致。

**符号说明**:
- $x_t^{(i)}$: 第 $i$ 帧加噪到时间 $t$ 的 latent
- $x_{\text{GT}}^{(<i)}$: 第 $i$ 帧之前的真实历史
- $\epsilon$: 注入噪声

### 公式 11: [[probability-flow ODE|概率流 ODE]]

$$
\frac{\mathrm{d} x_t}{\mathrm{d} t} = f_\theta(x_t, t, c)
$$

**含义**: 把扩散过程视作确定性 ODE，便于做步数蒸馏。50 步 teacher 解此 ODE 得到高质量 $\hat x_0$。

### 公式 12: [[ODE Distillation|ODE 蒸馏损失]]

$$
\mathcal{L}_{\text{ODE}} = \mathbb{E}_{x_T}\Big[\big\|f_\varphi(x_T, K=4) - \mathrm{sg}\big[\hat x_0^{\text{teacher}}\big]\big\|_2^2\Big]
$$

**含义**: Stage 2b 蒸馏目标——student 用 4 步达到 teacher 50 步的效果，$\mathrm{sg}[\cdot]$ 表示 stop-gradient。约 12× 加速。

**符号说明**:
- $f_\varphi$: 学生模型（参数 $\varphi$）
- $K=4$: 学生采样步数
- $\hat x_0^{\text{teacher}}$: 50 步 teacher ODE 求解结果

### 公式 13: [[DMD|生成式历史]]

$$
\hat x^{(<i)} = G_\varphi(\epsilon, c),\quad \epsilon \sim \mathcal{N}(0, I)
$$

**含义**: Stage 2c 关键——用 student 自己一次性生成的历史 $\hat x^{(<i)}$ 当 context，**模拟推理时真实输入分布**。

### 公式 14: [[DMD|分布匹配蒸馏损失]]

$$
\mathcal{L}_{\text{DMD}} = \mathcal{L}_{\text{reg}} + \lambda\, D_{\text{KL}}(p_\varphi \,\|\, p_{\text{data}})
$$

**含义**: 去噪回归项保持像素精度，KL 项把 student 的 marginal 分布拉向真实数据分布，闭合 [[Compounding Errors|exposure bias gap]]。

**符号说明**:
- $\mathcal{L}_{\text{reg}}$: 类似 $\mathcal{L}_{\text{TF}}$ 的去噪回归
- $p_\varphi$: student 生成分布
- $p_{\text{data}}$: 真实驾驶视频分布
- $\lambda$: KL 项权重

---

## 关键图表

### Figure 1: Paradigm Comparison / 三类世界模型范式对比

> 🖼️ **Figure 1** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 自上而下分别是 reconstruction-only（只显式 3D 但不能"做梦"）、generation-only（能生成但无几何约束、长程漂移）、JointWM（两者闭环）。这张 teaser 直接传达 motivation——**两类方法的失效模式恰好互补**。

### Figure 2: WorldRec Network Architecture / 重建网络架构

> 🖼️ **Figure 2** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 多相机输入 → 共享 backbone 多尺度特征 → [[Sparse Scene Query|稀疏 3D 查询]] 投影采样 → [[Cross-View Attention|跨视图/跨时间聚合]] → MLP 解出高斯属性 → 可微光栅化渲染。**重点**：高斯是 query-level 稀疏的，不是像素稠密的。

### Figure 3: WorldGen Architecture & Two-Stage Training / 生成网络与两阶段训练

> 🖼️ **Figure 3** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: DiT 主干 + 多模态条件（trajectory / camera / layout / text）。左侧是 Stage 1 [[Flow Matching|双向预训练]]，右侧是 Stage 2 [[Causal Fine-tuning|三阶段因果微调]]（[[Teacher Forcing]] → [[ODE Distillation]] → [[DMD]]）。注意右侧每个 stage 的注意力 mask 变化——双向 → 因果 → 因果+少步 → 因果+生成式 context。

### Figure 4: Joint World Model Architecture / 联合世界模型

> 🖼️ **Figure 4** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: WorldRec 与 WorldGen 在两个接口耦合：(1) 高斯渲染图 → WorldGen 的 RGB 先验条件；(2) WorldGen 输出 → WorldRec 增量扩张。形成闭环。

### Figure 5a: JointWM-WorldRec Incremental Reconstruction / 增量重建

> 🖼️ **Figure 5a** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 新到 frame 的查询 token 通过 [[Cross-Attention|cross-attention]] 与已缓存的 Gaussian token 融合：已观测区域 refine，未观测区域 expand。整个机制不需要重新优化。

### Figure 5b: JointWM-WorldGen RGB Conditioning / RGB 先验条件

> 🖼️ **Figure 5b** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 当前 Gaussian 场被 rasterize 到 target camera 得到 $I_{\text{ren}}$（含空洞、伪影），作为第四种条件模态拼入 DiT。生成模型"看着这张草图"补全细节。

### Figure 6: WorldRec on Waymo / Waymo 原轨迹重建

> 🖼️ **Figure 6** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 与 [[MVSplat]]、[[NoPoSplat]]、[[DepthSplat]]、[[STORM]]、[[DGGT]] 在原轨迹上的对比。本文方法在路面纹理、远端建筑、动态车辆边缘清晰度上明显占优。

### Figure 7: WorldRec Novel View on Waymo / Waymo 新视角合成

> 🖼️ **Figure 7** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 偏离训练轨迹的 NVS。重点看远端几何稳定性，本文方法没有 DGGT/STORM 那种"远处糊掉"现象。

### Figure 8: WorldRec on Private Data / 内部数据集

> 🖼️ **Figure 8** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 小米内部数据集上的 NVS 结果，验证跨数据集泛化与产品落地可行性。

### Figure 9: WorldRec BEV Reconstruction / 鸟瞰图重建

> 🖼️ **Figure 9** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: BEV 视角看几何一致性——道路边界、车道线、车辆位置在俯视图下不应错位。这是驾驶场景独有的评估维度。

### Figure 10: WorldGen Long-Tail Scenes / 长尾场景生成

> 🖼️ **Figure 10** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 文本/布局条件下生成"道路上的动物"等罕见场景，证明 generation 能力补充了真实数据采集的盲区。

### Figure 11: WorldGen Extreme Weather / 极端天气生成

> 🖼️ **Figure 11** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 雨/雪/雾天气可控合成——对自动驾驶数据闭环来说，这类长尾数据是金钱难买的。

### Figure 12: WorldGen Long-Horizon Controllable Generation / 长程可控生成

> 🖼️ **Figure 12** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 10fps / 30fps 模式下生成接近 1 分钟连续视频，trajectory 与 layout 条件保持可控。Stage 2c 的 DMD 在这里发挥关键作用——没有它，长程必漂。

### Figure 13: JWM Long-Horizon Consistency #1

> 🖼️ **Figure 13** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 联合世界模型长程时序一致性示例 1。注意路面纹理/建筑结构在数十秒后仍稳定，这是单纯 generation 做不到的。

### Figure 14: JWM Long-Horizon Consistency #2

> 🖼️ **Figure 14** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 长程时序一致性示例 2。复杂交叉路口场景下几何与动态目标都不漂。

### Figure 15: JWM Long-Horizon Consistency #3

> 🖼️ **Figure 15** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 长程时序一致性示例 3。从 Joint vs WorldGen-only 的对比可看出几何锚点的关键作用。

### Figure 16: JWM Multi-View Spatial Consistency / 多视图空间一致性

> 🖼️ **Figure 16** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 多个相机视角下，同一时刻物体位置、光照、阴影保持物理上自洽——因为它们共享同一份高斯场作为先验。

### Figure 17: JWM Multi-Run Stability / 多次推理稳定性

> 🖼️ **Figure 17** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.18137)）

**说明**: 同一 prompt 不同随机种子，**结构骨架（建筑、道路、车道）保持一致**，只有非结构性细节有变化。这是工业部署看重的"可重复性"。

### Table 1: World Modeling Paradigms Comparison / 范式对比

| Capability | Recon-only | Gen-only | NeoVerse | AlpaDreams | **Ours** |
|---|---|---|---|---|---|
| Explicit 3D Scene | ✓ | ✗ | ✓ | ✗ | **✓** |
| Generative Capability | ✗ | ✓ | ✓ | ✓ | **✓** |
| Novel View Synthesis | ✓ | ✓ | ✓ | ✓ | **✓** |
| Future Prediction | ✗ | ✓ | ✓ | ✓ | **✓** |
| Geometry Consistency | Strong | Weak | Medium | Weak | **Strong** |
| Long-horizon Stability | Static | Drift | Medium | Medium | **Stable** |

**说明**: 把 JointWM 放在五种范式里，**唯一在所有维度都是 ✓/Strong/Stable 的方案**。

### Table 2: Reconstruction Quantitative Results / 重建定量结果

| Method | Waymo PSNR↑ | Waymo SSIM↑ | nuScenes ZS PSNR↑ | nuScenes ZS SSIM↑ | nuScenes FT PSNR↑ | nuScenes FT SSIM↑ |
|---|---|---|---|---|---|---|
| [[MVSplat]] | 20.56 | 0.697 | 17.84 | 0.563 | — | — |
| [[NoPoSplat]] | 24.31 | 0.751 | 19.75 | 0.545 | — | — |
| [[DepthSplat]] | 23.26 | 0.696 | 19.52 | 0.601 | — | — |
| [[STORM]] | 26.38 | 0.794 | 17.77 | 0.669 | 24.54 | 0.784 |
| [[DGGT]] | 27.41 | 0.846 | 25.31 | 0.794 | 26.63 | 0.813 |
| **Ours** | **28.48** | **0.861** | **26.54** | **0.821** | **27.50** | **0.826** |

**说明**: 所有指标都是 SOTA。**zero-shot nuScenes 26.54 PSNR** 直接超过多数方法 fine-tune 后的数——稀疏查询设计带来的泛化红利。

### Table 3: Driving Video Generation Comparison / 视频生成对比（nuScenes）

| Model | Bi/AR | Venue | FID↓ | FVD↓ | Frames | Infer. Time |
|---|---|---|---|---|---|---|
| [[MagicDrive]] | Bi | ICLR'24 | 16.20 | — | 1 | — |
| MagicDrive-V2 | Bi | ICCV'25 | 20.91 | 94.84 | 16 | — |
| [[Vista]] | Bi | NeurIPS'24 | 6.9 | 89.4 | 16 | — |
| [[DiVE]] | Bi | arXiv'25 | 7.14 | 68.4 | 8 | — |
| Delphi | Bi | arXiv'24 | 15.08 | 113.5 | 8 | — |
| [[UniScene]] | Bi | CVPR'25 | 6.12 | 70.52 | 8 | — |
| [[Genesis]] | Bi | NeurIPS'25 | 6.45 | 67.87 | 16 | — |
| [[Epona]] | AR | ICCV'25 | 7.5 | 82.8 | 16 | 1.06 |
| **Ours** | AR | — | **7.04** | **64.97** | **81** | **0.19** |

**说明**: 自回归路线里同时拿下 FVD 最优、生成长度最长（81 帧 vs ≤16）、延迟最低（**5.5× 快于 Epona**）。FID 与 UniScene/Genesis 接近，但本文是自回归长程模式，**不可比"长度"维度**。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Waymo Open Dataset | 1000+ segments | 多视角 + LiDAR | 重建主基准 |
| [[nuScenes]] | 1000 scenes | 6 cam, 360° | 重建+生成评测，跨域评测 |
| 小米内部数据 | 未披露 | 量产车采集 | 泛化与产品落地验证 |

### 评测指标

- **重建**: [[PSNR]] / [[SSIM]]（含原轨迹、NVS 两种）
- **生成**: [[FID]] / [[FVD]] / 生成帧数 / 单帧推理延迟
- **联合**: 长程一致性、多视图一致性、多次推理稳定性（定性为主）

### 实现细节（论文披露部分）

- **重建**: 共享 visual backbone（论文未指明具体型号）+ MLP head；可微高斯光栅化
- **生成**: DiT backbone + [[VAE]] vision encoder + umT5 text encoder
- **训练阶段**: Bi-pretrain → Teacher Forcing → ODE Distillation → DMD
- **推理硬件**: NVIDIA H20，单视 0.19 s/frame，三视 0.46 s/frame
- **采样步数**: K=4（ODE 蒸馏后），相比 teacher 50 步约 12× 加速

### 关键定性观察

1. **稀疏查询 vs 稠密像素高斯**：前者参数更少但表征能力反而更强（聚合了多视图信息）。
2. **ODE 蒸馏不掉点**：4 步 vs 50 步 teacher 的 FID/FVD 几乎不变，验证蒸馏路径有效。
3. **DMD 是长程关键**：未做 DMD 时长程视频在 20 秒后明显漂移；做完后 1 分钟仍稳。
4. **跨数据集零样本能力**：Waymo 训练，nuScenes 零样本超过多数 fine-tune 方法，证明 sparse query 表征不是 dataset-specific 的。

---

## 批判性思考

### 优点

1. **三大瓶颈一次性解决**：优化时间、推理步数、重建/生成解耦——这三件事工业界喊了三年，本文一次给出系统级答案。
2. **稀疏查询是真正的"驾驶专属"设计**：相比通用域稠密 GS，稀疏 query 在 outdoor unbounded scene 下更鲁棒、参数更省。
3. **三阶段蒸馏是工程亮点**：Teacher Forcing → ODE → DMD，每一步都对应一个具体痛点，路径设计很清晰。
4. **量产指标过关**：0.19 s/frame 是真正能塞进数据闭环 pipeline 的延迟。
5. **零样本跨域**：nuScenes ZS 直接打过别人 FT，强泛化。

### 局限性

1. **没开源代码**：项目页只有 demo，没仓库链接——工业向技术报告的通病，复现难度高。
2. **细节缺失**：backbone 型号、训练数据规模、超参（$\lambda$、KL 权重）几乎都没披露，更像内部技术报告对外发布。
3. **公式 14 的 DMD 表述偏简化**：原始 DMD 论文里 KL 是通过两组 score model 估计的，本文写成 $D_{KL}(p_\varphi\|p_{data})$ 略含糊。
4. **没和 [[CausVid]]/[[Vid2World]] 这类通用 causal video diffusion 做对比**：仅与驾驶 domain 内方法比，方法路线创新性的相对性难以评估。
5. **联合训练成本未提**：双阶段联合微调需要多少 GPU·hour 没说，工业可复制性存疑。
6. **多视一致性的来源仍主要靠几何先验**：如果遇到 ego 第一次进入的"完全未观测区域"，纯 generation 模式下一致性还能维持多久？文中没明确测过这种 worst case。

### 潜在改进方向

1. **加入 [[Diffusion Forcing]] 替代 Teacher Forcing**：在 token 级别独立加噪，可能进一步减少 train-test gap。
2. **WorldRec 引入语义/类别先验**：现在的高斯只编码 RGB+几何，可以补 occupancy / semantic 多任务。
3. **WorldGen 引入 [[LongCat-Video]] / [[Wan2.2]] 这类强通用先验**：当前从零训 DiT，初始化为开源大规模视频生成器可能再省一大半算力。
4. **闭环 RL 微调**：用下游 planner / perception loss 做 reward，把 WM 调向 task-aware。
5. **3D occupancy + 高斯混合表征**：现在的高斯无法显式编码"可行驶区域"。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [ ] 训练细节完整（仅披露阶段顺序，无超参/数据规模）
- [x] 数据集可获取（Waymo、nuScenes 公开，内部数据不公开）

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: WorldRec 的底层表征
- [[DiT]]: WorldGen 的 backbone
- [[Rectified Flow]]: 双向预训练损失
- [[Teacher Forcing]]: Stage 2a 因果初始化
- [[ODE Distillation]]: Stage 2b 步数蒸馏
- [[DMD]]: Stage 2c 闭合训测 gap
- [[probability-flow ODE]]: ODE 蒸馏的数学基础

### 对比

- [[STORM]] / [[DGGT]]: 前馈重建路线 SOTA，被本文 WorldRec 全面超过
- [[MagicDrive]] / [[Vista]] / [[Epona]]: 驾驶视频生成 SOTA，被 WorldGen 在 FVD/速度/长度上压制
- [[CausVid]] / [[Vid2World]]: 通用 causal video diffusion，与 WorldGen 共享思路但未直接比较

### 方法相关

- [[Feed-Forward 重建]]: WorldRec 的范式
- [[Causal Fine-tuning]]: WorldGen 的微调路径
- [[Sparse Scene Query]]: 重建端的关键设计
- [[Joint World Model]]: 本文核心范式

### 硬件/数据相关

- [[nuScenes]]: 主要评测数据集
- [[Waymo]]: 主要训练/评测数据集

---

## 速查卡片

> [!summary] Xiaomi Auto World Model (JointWM)
> - **核心**: 前馈稀疏 3DGS 重建 + 4 步因果 DiT 视频生成，双向耦合
> - **方法**: WorldRec（稀疏查询聚合 → MLP 解高斯）+ WorldGen（Bi-pretrain → TF → ODE-Distill → DMD）+ 高斯渲染图回灌生成
> - **结果**: Waymo 28.48 PSNR（重建 SOTA），nuScenes FVD 64.97 / 0.19 s/frame（生成 SOTA），重建 10s vs 4h（1400× 加速）
> - **代码**: 未开源；项目页 <https://JointWM.github.io>

---

*笔记创建时间: 2026-05-27*
