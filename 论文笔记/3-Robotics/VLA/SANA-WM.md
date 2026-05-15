---
title: "SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer"
method_name: "SANA-WM"
authors: [Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, Enze Xie]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, linear-attention, diffusion-transformer, camera-control, minute-scale, long-video]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.15178v1
created: 2026-05-15
---

# 论文笔记：SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA, MIT, HKUST, HKU |
| 日期 | May 2026 |
| 项目主页 | https://nvlabs.github.io/Sana/WM/ |
| 对比基线 | [[CogVideoX]], [[Wan2.2]], Matrix-Game 3.0, HY-WorldPlay, LingBot-World, Infinite-World |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15178) / [HTML](https://arxiv.org/html/2605.15178v1) / [Project](https://nvlabs.github.io/Sana/WM/) |

---

## 一句话总结

> SANA-WM 用 2.6B 参数的[[混合线性扩散变换器]]在单卡 H100 上生成 60 秒 720p 可控相机轨迹的世界模型视频，吞吐量比同类开源基线高 36 倍。

---

## 核心贡献

1. **混合线性扩散 Transformer 架构**: 将 15 个 [[Frame-wise Gated DeltaNet|帧级 GDN]] 模块与 5 个 [[softmax 注意力]] 模块交错堆叠，在 D×S 空间 token 数下保持训练稳定，长序列推理内存近似恒定。
2. **双分支相机控制**: 在潜帧速率上用 [[Ray-Local UCPE|射线局部 UCPE]] 注入粗粒度位姿，在原始帧速率上用 [[Plücker 射线表示]] 注入细粒度运动，6-DoF 旋转误差降至 4.5°（比 LingBot-World 的 10.47° 提升一倍以上）。
3. **两阶段生成 + 长视频精炼器**: 阶段一生成低保真长视频，阶段二用[[截断 σ 流匹配]]训练的 17B 精炼器抑制长时漂移，60 秒 ΔIQ 仅 0.31（基线 HY-WorldPlay 高达 25.88）。
4. **可复现的数据-训练管线**: 在 ~213K 公开视频片段、64 卡 H100 上 15 天完成训练，附带基于 [[VIPE]] + [[Pi3X]] + [[MoGe-2]] 的相机姿态标注引擎与 3DGS 增强流程。

---

## 问题背景

### 要解决的问题

[[世界模型]] 需要从单张参考图与相机轨迹合成**分钟级、720p、相机精确可控**的视频，作为机器人与具身智能体的可交互模拟器。然而现有[[视频扩散模型]]在长视频任务下面临三大瓶颈：

1. 全局 [[softmax 注意力]] 的 KV cache 在分钟级序列下显存爆炸；
2. 纯 [[线性注意力]] 与 [[潜空间世界模型]] 在长时滚动中精度迅速漂移；
3. 6-DoF 相机条件难以注入到[[扩散变换器]]中，导致轨迹误差累积。

### 现有方法的局限

- **CogVideoX / Wan2.2** 等通用 T2V/I2V 模型未原生支持相机条件，且 KV cache O(T²) 增长，60 秒生成需 8 卡 H100。
- **Matrix-Game 3.0、HY-WorldPlay** 等行业基线参数量 5-14B，单卡推理不可行，吞吐量低（0.6-3.1 视频/小时）。
- **Infinite-World** 仅 1.3B 但分辨率 480p，长时漂移严重（ΔIQ +6.72）。
- 现有相机条件方法（Plücker only、PRoPE）单独使用时位姿误差仍 >6°，难以满足导航类下游应用。

### 本文的动机

作者发现：(a) **线性注意力**适合长序列但状态会因空间 token 数 S 爆炸；(b) **softmax 注意力**精度高但代价大；二者按 15:5 交错可同时保留长程精确召回与显存常数；同时，相机控制必须**显式分离粗几何与细运动**：粗几何决定潜空间整体取向，细运动决定逐像素射线方向。

---

## 方法详解

### 模型架构

SANA-WM 采用**混合线性扩散 Transformer** 架构，基于 SANA-Video 拓展：

- **输入**: 参考帧 $I_0$、文本指令 $l$、6-DoF 相机轨迹 $\{P_t\}_{t=1}^T$、文本/视频/位姿三路 token
- **Backbone**: 20 个 transformer block，$d_{\text{model}}=2240$，20 注意力头，单头维度 $D=112$
- **核心模块**:
  - 15 个 [[Frame-wise Gated DeltaNet|帧级 GDN]] block 负责长序列扫描
  - 5 个 [[softmax 注意力]] block（位置 $\{3, 7, 11, 15, 19\}$）保留精确召回
  - [[Ray-Local UCPE|射线局部 UCPE]] 粗分支 + [[Plücker 射线表示]] 细分支
- **VAE**: [[LTX-2 VAE]]，128 潜通道，比 Wan2.1-VAE 小 8 倍
- **输出**: 961 帧潜表示（60 秒 @ 16 FPS）
- **总参数**: 2.6B（精炼器另加 17B）

### 核心模块

#### 模块1: 帧级门控 DeltaNet（Frame-wise GDN）

**设计动机**: 标准 [[线性注意力]] 在 D×S 空间 token 下状态范数随 S 增长，长序列训练不稳。本文将 token 级扫描改为**潜帧级扫描**，每帧维护一个 D×D 状态矩阵，并引入 [[Delta 规则]] 与衰减门 $\gamma_t$。

**具体实现**:
- 把 SANA-Video 的累积 [[线性注意力]] 替换为帧级 GDN
- 状态递推 $S_t \in \mathbb{R}^{D \times D}$ 同时支持遗忘（$\gamma_t$）与残差修正（$\beta_t$）
- 用 $1/\sqrt{D \cdot S}$ 的代数缩放使 $\|M_t\|_2 \le \gamma_t \le 1$，保证训练稳定
- 自定义 Triton 内核为 GDN scan/gate 提供 1.5-2.0× 加速

#### 模块2: 周期性 Softmax 注意力

**设计动机**: 纯 GDN 在长程精确召回（如场景重访）上仍有误差，需要少量[[softmax 注意力]]作为"锚点"。

**具体实现**:
- 在 20 层中均匀插入 5 层 softmax，每 4 层一次，覆盖网络浅-中-深
- 推理时配合 attention sink token + 局部时间窗，支持 [[chunk-causal]] 部署
- 块因果窗口长度 < 60 帧，显存占用仍接近常数

#### 模块3: 双分支相机控制

**设计动机**: 粗几何（每帧整体位姿）与细运动（每像素射线）的尺度不同，应在不同分辨率注入。

**粗分支 — Ray-Local UCPE（潜帧速率）**:
- 用相机内参反投影像素，构建世界系→射线系的 $4 \times 4$ 变换矩阵 $D_i$
- 注意力头向量分裂为几何通道（应用 $D_i$）+ 标准 [[RoPE]] 通道
- 共享 $\text{GDN}_{\text{cam}}$ 门，输出经零初始化投影叠加到主分支

**细分支 — Plücker 原始帧混合（原始帧速率）**:
- 计算逐像素 [[Plücker 射线表示]] $\rho_{r,p} = (d_{r,p}, o_r \times d_{r,p}) \in \mathbb{R}^6$
- 8 个原始帧（每 VAE 步长）打包成 48 通道张量
- 3D patch embedder + 零初始化投影注入到 self-attention 后

#### 模块4: 两阶段生成 + 长视频精炼器

**Stage 1**: 2.6B 主模型生成 720p 长视频潜表示（低保真但相机精确）。
**Stage 2**: 17B 长视频精炼器在 [[截断 σ 流匹配]] 框架下读取 stage-1 潜表示并细化纹理。该精炼器把 LTX-2.3 短视频先验扩展到 60s，60 秒 ΔIQ 从 +3.73 降至 +1.17。

#### 模块5: Context-Parallel 训练

**设计动机**: 961 帧序列单卡放不下，跨卡分片时 GDN 的循环状态需要严格一致。

**具体实现**:
- 将帧索引 $I_p = \{pT/P, \ldots, (p+1)T/P - 1\}$ 切给 $P$ 张 GPU
- 每分片计算转移合成 $C_p$ 与输入合成 $H_p$，仅 all-gather 紧凑的 $D \times D$ 矩阵
- 通过独占前缀合成精确恢复初始状态，通信 $O(P)$
- 配合 halo 交换共享 $K-1$ 帧边界，支持时间卷积

---

## 关键公式

### 公式1: [[线性注意力|累积线性注意力（SANA-Video 基线）]]

$$
\tilde{O}_t^{LA} = \left( \sum_{\tau=0}^{t} V_\tau \phi(K_\tau)^\top \right) \phi(Q_t) = (A_{t-1}^{LA} + V_t \phi(K_t)^\top) \phi(Q_t)
$$

**含义**: SANA-Video 用 ReLU 核 $\phi$ 把 softmax 注意力线性化，状态 $A_t^{LA}$ 在时间维上累积。但 D×S 维度下状态范数会随空间 token 数 S 爆炸。

**符号说明**:
- $Q_t, K_t, V_t \in \mathbb{R}^{D \times S}$: 每头的 query/key/value
- $\phi(\cdot) = \text{ReLU}(\cdot)$: 核函数
- $A_t^{LA} \in \mathbb{R}^{D \times D}$: 累积分子状态
- $S = H_\ell W_\ell$: 每帧空间 token 数

### 公式2: [[Frame-wise Gated DeltaNet|帧级 GDN 递推]]

$$
\begin{aligned}
S_t &= S_{t-1} M_t + U_t \\
M_t &= \gamma_t (I - \hat{K}_t \beta_t \hat{K}_t^\top) \\
U_t &= V_t \beta_t \hat{K}_t^\top \\
O_t &= S_t \hat{Q}_t
\end{aligned}
$$

**含义**: 把 token 级扫描升级为帧级扫描，引入衰减门 $\gamma_t$（遗忘旧帧）与更新门 $\beta_t$（局部残差修正）。每帧仅更新一次 $S_t$，把整帧空间 token 视为同步事件。

**符号说明**:
- $S_t \in \mathbb{R}^{D \times D}$: 帧 $t$ 的循环状态
- $\gamma_t \in (0, 1]$: 帧级衰减
- $\beta_{t,s} \in [0, 1]$: 每 token 更新强度
- $\hat{K}_t, \hat{Q}_t$: 归一化的 key/query
- $O_t \in \mathbb{R}^{D \times S}$: 当前帧输出

### 公式3: [[空间稳定的键归一化]]

$$
\hat{K}_t = \bar{K}_t \cdot \frac{1}{\sqrt{D \cdot S}}, \quad \bar{K}_t = \text{ReLU}(\text{RMSNorm}(K_t))
$$

**含义**: 通过代数缩放保证 $\text{tr}(\hat{K}_t \beta_t \hat{K}_t^\top) \le 1$，进而 $\|M_t\|_2 \le \gamma_t \le 1$，使状态范数不会因空间 token 数 S 增大而爆炸。

**符号说明**:
- $D$: 单头维度
- $S$: 空间 token 数（$H_\ell W_\ell$）
- $\bar{K}_t$: RMSNorm + ReLU 后的 key
- $\text{tr}(\cdot)$: 矩阵迹

### 公式4: [[Ray-Local UCPE|射线局部 UCPE 相机条件]]

$$
\begin{aligned}
\tilde{Q}_i^c &= (D_i^\top \oplus \text{RoPE}_i) Q_i^c \\
(\tilde{K}_i^c, \tilde{V}_i^c) &= (D_i^{-1} \oplus \text{RoPE}_i) (K_i^c, V_i^c) \\
O_i^c &= (D_i \oplus \text{RoPE}_i^{-1}) \, \text{GDN}_{\text{cam}}(\tilde{Q}^c, \tilde{K}^c, \tilde{V}^c)_i
\end{aligned}
$$

**含义**: 把每个像素的 query/key/value 拆成几何通道（应用 $4 \times 4$ 射线变换 $D_i$）与外观通道（应用 [[RoPE]]）。注意力在射线局部坐标系内计算，输出再变回世界系。

**符号说明**:
- $D_i \in \mathbb{R}^{4 \times 4}$: 世界系→射线系变换矩阵
- 上标 $c$: 相机分支专用张量
- $\oplus$: 分块对角合成（几何通道 ⊕ 外观通道）
- $\text{GDN}_{\text{cam}}$: 共享门的相机分支 GDN

### 公式5: [[Plücker 射线表示]]

$$
\rho_{r,p} = (d_{r,p}, o_r \times d_{r,p}) \in \mathbb{R}^6
$$

**含义**: 每条相机光线用方向向量 + 矩量向量（位置叉乘方向）表示，构成 6D 几何特征，可在原始帧速率上提供细粒度相机条件。

**符号说明**:
- $d_{r,p}$: 像素 $p$ 在帧 $r$ 上的单位射线方向
- $o_r$: 帧 $r$ 的相机中心
- $\times$: 三维叉乘

### 公式6: [[截断 σ 流匹配|截断 σ 流匹配（精炼器损失）]]

$$
\begin{aligned}
x_1 &= (1 - \sigma_{\text{start}}) x_l + \sigma_{\text{start}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \\
\alpha &= \sigma_t / \sigma_{\text{start}}, \quad x_t = (1 - \alpha) x_h + \alpha x_1 \\
v^* &= (x_1 - x_h) / \sigma_{\text{start}} \\
\mathcal{L}_{\text{refiner}} &= \mathbb{E}_{\sigma_t, \epsilon} \| v_\theta(x_t, \sigma_t, c) - v^* \|_2^2
\end{aligned}
$$

**含义**: 把 stage-1 潜表示 $x_l$ 加少量噪声作为流的起点 $x_1$，让网络学习从 $x_1$ 到高保真目标 $x_h$ 的速度场 $v^*$。$\sigma_{\text{start}} = 0.909375$ 保证训练只覆盖 stage-1 输出附近的窄噪声带。

**符号说明**:
- $x_l$: stage-1 输出潜变量
- $x_h$: 高保真目标潜变量
- $v_\theta$: 17B 精炼器
- $c$: 条件（文本 + 相机）
- $\alpha \in (0, 1]$: 插值系数

### 公式7: [[Context-Parallel 训练|上下文并行合成]]

$$
S_{\text{end}}^{(p)} = S_{\text{start}}^{(p)} C_p + H_p, \quad C_p = \prod_{t \in I_p} M_t
$$

$$
\bar{S}_0 = 0, \quad \bar{S}_{p+1} = \bar{S}_p C_p + H_p, \quad S_{\text{start}}^{(p)} = \bar{S}_p
$$

**含义**: 把 961 帧切给 $P$ 卡，每卡本地计算转移合成 $C_p$ 与输入合成 $H_p$，再通过独占前缀合成精确恢复任意分片的初始状态，使得跨卡训练与单卡数学等价，通信量仅 $O(P)$ 个 $D \times D$ 矩阵。

**符号说明**:
- $I_p = \{pT/P, \ldots, (p+1)T/P - 1\}$: 第 $p$ 卡的帧索引
- $C_p$: 第 $p$ 卡的状态转移
- $H_p$: 第 $p$ 卡的输入累积
- $\bar{S}_p$: 第 $p$ 卡的左前缀状态

### 公式8: [[相机位姿评估指标|位姿评估指标]]

$$
\begin{aligned}
\text{RotErr} &= \frac{1}{T} \sum_{t=1}^{T} \frac{180}{\pi} \arccos\left( \text{clip}\left( \frac{\text{tr}(R_t^\top \tilde{R}_t) - 1}{2}, -1, 1 \right) \right) \\
\text{TransErr} &= \frac{1}{T} \sum_{t=1}^{T} \| t_t - \tilde{t}_t \|_2 \\
\text{CamMC} &= \frac{1}{T} \sum_{t=1}^{T} \| P_t - \tilde{P}_t \|_F
\end{aligned}
$$

**含义**: 用 [[Pi3X]] 从生成视频中恢复相机轨迹 $\tilde{P}_t = [\tilde{R}_t | \tilde{t}_t]$ 后与真值对比：旋转误差（度）、平移误差（米）、整体一致性（Frobenius 范数）。

**符号说明**:
- $R_t, t_t$: 真值旋转/平移
- $\tilde{R}_t, \tilde{t}_t$: Pi3X 恢复的对齐位姿
- $\| \cdot \|_F$: Frobenius 范数

---

## 关键图表

### Figure 1: Teaser / 任务定义

![Figure 1](https://arxiv.org/html/2605.15178v1/x1.png)

**说明**: SANA-WM 输入单张图像 + 6-DoF 相机轨迹，输出 720p、60 秒、高保真的世界视频，单卡 H100 可运行。展示室内、户外、第一人称等多种场景。

### Figure 2: 模型架构

![Figure 2](https://arxiv.org/html/2605.15178v1/x2.png)

**说明**: 主干由 15 个 [[Frame-wise Gated DeltaNet|帧级 GDN]] 与 5 个 [[softmax 注意力]] 交错组成；文本、视频、位姿 token 通过粗分支 [[Ray-Local UCPE]] 与细分支 [[Plücker 射线表示]] 注入相机条件；零初始化投影保证预训练权重不被破坏。

### Figure 3: 数据构造管线

![Figure 3](https://arxiv.org/html/2605.15178v1/x3.png)

**说明**: 公开视频 → [[VIPE]] 标注度量尺度相机位姿 → 3DGS 增强（FCGS + DiFix3D）→ 多级过滤 + Qwen3.5 VLM 标注。最终输出 212,975 个相机精确对齐的训练片段。

### Figure 4: 基准轨迹模板

![Figure 4](https://arxiv.org/html/2605.15178v1/x4.png)

**说明**: 1 分钟评估基准包含 Simple/Hard 两档 BEV 轨迹，覆盖直行、回环、转头、爬升等运动模式。

### Figure 5: 60 秒 Hard 轨迹定性对比

![Figure 5](https://arxiv.org/html/2605.15178v1/x5.png)

**说明**: 与 4 个开源基线在 4 段 60 秒 Hard 视频上的可视化对比，绿色边框为 SANA-WM。可见基线在长时滚动后出现纹理崩坏与轨迹偏离，SANA-WM 保持稳定。

### Figure 6: 训练稳定性消融

![Figure 6](https://arxiv.org/html/2605.15178v1/x6.png)

**说明**: 公式 3 的 $1/\sqrt{D \cdot S}$ 缩放使 GDN 梯度收敛；无缩放或 $1/\sqrt{D}$ 缩放的变体在中后期发散。

### Figure 7: 效率消融

![Figure 7](https://arxiv.org/html/2605.15178v1/x7.png)

**说明**: (a) 60 秒单卡推理延迟分解：扩散主干、精炼器、VAE 解码各占比；(b) H100 上序列长度 vs 显存/延迟曲线——纯 softmax 在 ~30s 就 OOM，混合架构在 960 帧仍 ~6.5 GB。

### Figure 8: 精炼器消融

![Figure 8](https://arxiv.org/html/2605.15178v1/x8.png)

**说明**: 无精炼器（仅 stage-1）的输出在 30s 后出现明显模糊；加入 17B 精炼器后细节稳定到 60s。

### Figure 9: 基准首帧示例

![Figure 9](https://arxiv.org/html/2605.15178v1/x9.png)

**说明**: 4 个场景类别（室内、城市、自然、第一人称）的代表性首帧条件图，由 Nano Banana Pro 生成。

### Figure 10: 基准轨迹分布

![Figure 10](https://arxiv.org/html/2605.15178v1/x10.png)

**说明**: Simple 与 Hard 轨迹的地面投影与高度剖面，Hard 集包含强烈的高度变化与回环。

### Figure 11: 额外 Hard 轨迹定性结果

![Figure 11](https://arxiv.org/html/2605.15178v1/x11.png)

**说明**: 更多 Hard 轨迹下与基线的对比，SANA-WM（绿框）保持高保真与轨迹一致性。

### Figure 12: 生成视频的 3D 重建

![Figure 12](https://arxiv.org/html/2605.15178v1/x12.png)

**说明**: 用 Pi3X 从 3 段 SANA-WM 生成视频中重建 3D 结构，验证几何一致性——若视频内部不一致，重建会失败。

### Table 1: 训练数据总览

| 来源 | 类型 | 时长 | 片段数 | 位姿来源 |
|------|------|------|--------|----------|
| SpatialVID-HQ | 真实 | 10s | 158,369 | VIPE + Pi3X/MoGe-2 |
| DL3DV | 真实 | 10s | 5,691 | GT pose + Pi3X |
| DL3DV GS Refined | 合成 | 60s | 14,881 | GT pose + Pi3X |
| OmniWorld | 合成 | 60s | 1,720 | VIPE + GT depth |
| Sekai Game | 合成 | 60s | 3,560 | GT pose + Pi3X |
| Sekai Walking-HQ | 真实 | 60s | 9,767 | VIPE + Pi3X/MoGe-2 |
| MiraData | 真实 | 60s | 18,987 | VIPE + Pi3X/MoGe-2 |
| **Total** | | | **212,975** | |

**说明**: 仅使用公开数据，总规模 ~213K 片段，60 秒长片段占 1/4，足以训练分钟级世界模型。

### Table 2: 1 分钟基准定量对比（Simple-Trajectory）

| Method | Param | Res | #G | RotErr↓ | TransErr↓ | CamMC↓ | Overall↑ | Mem(GB)↓ | Tput↑ |
|--------|-------|-----|----|---------|-----------|--------|----------|----------|-------|
| Infinite-World | 1.3B | 480p | 1 | 16.55 | 1.98 | 2.08 | 79.18 | 53.5 | 5.9 |
| LingBot-World | 14B+14B | 480p | 8 | 10.47 | 2.01 | 2.05 | 81.82 | 454.1 | 0.6 |
| HY-WorldPlay | 8B | 480p | 8 | 17.89 | 2.36 | 2.45 | 68.82 | 215.5 | 1.1 |
| Matrix-Game 3.0 | 5B | 720p | 8 | 12.96 | 1.83 | 1.92 | 78.53 | 106.2 | 3.1 |
| **SANA-WM** | **2.6B** | **720p** | **1** | **7.59** | **1.59** | **1.63** | **79.29** | **51.1** | **24.1** |
| **SANA-WM + refiner** | **2.6B+17B** | **720p** | **1** | **4.50** | **1.39** | **1.41** | **80.62** | **74.7** | **22.0** |

**说明**: SANA-WM 在单卡 720p 下相机精度全面领先；加精炼器后旋转误差 4.5° 为全场最佳。吞吐量 24 vid/h 是 LingBot-World 的 40 倍。

### Table 2（续）: Hard-Trajectory

| Method | RotErr↓ | TransErr↓ | CamMC↓ | Overall↑ |
|--------|---------|-----------|--------|----------|
| Infinite-World | 41.31 | 2.49 | 2.84 | 79.51 |
| LingBot-World | 18.99 | 1.65 | 1.81 | 81.89 |
| HY-WorldPlay | 35.46 | 2.34 | 2.64 | 70.46 |
| Matrix-Game 3.0 | 18.79 | 1.67 | 1.82 | 78.79 |
| **SANA-WM** | **10.02** | **1.66** | **1.72** | **79.60** |
| **SANA-WM + refiner** | **8.34** | **1.39** | **1.44** | **81.89** |

**说明**: Hard 轨迹下旋转误差降低更明显——基线翻倍恶化（如 Infinite-World 16°→41°），SANA-WM 仅 7.59°→10.02°。

### Table 3: 渐进式训练消融（VBench-I2V，5 秒）

| Model | Attention | Tokenizer | Quality↑ | I2V↑ | Total↑ | Mem(GiB)↓ | Lat(ms)↓ | Tput↑ |
|-------|-----------|-----------|----------|------|--------|-----------|----------|-------|
| Sana-Video | 累积线性 | Wan 2.1 / 480p | 0.7683 | 0.9073 | 0.8378 | 8.90 | 1266.6 | 0.79 |
| + LTX2 VAE | 累积线性 | LTX2 / 720p | 0.7697 | 0.9082 | 0.8390 | 5.40 | 371.7 | 2.69 |
| + Hybrid attn. | GDN + softmax | LTX2 / 720p | 0.7834 | 0.9226 | 0.8530 | 5.68 | 433.2 | 2.31 |

**说明**: LTX2 VAE 把延迟从 1266ms 降到 372ms（3.4×），混合注意力进一步将质量从 0.839 提升到 0.853。

### Table 4: 相机条件消融（OmniWorld）

| Camera Encoding | FVD↓ | RotErr↓ | TransErr↓ | CamMC↓ |
|-----------------|------|---------|-----------|--------|
| No control | 348.93 | 16.93 | 0.2347 | 0.4937 |
| Plücker only | 339.45 | 16.02 | 0.2340 | 0.4742 |
| PRoPE | 326.70 | 6.29 | 0.1857 | 0.2629 |
| UCPE only | 314.88 | 7.73 | 0.1350 | 0.2453 |
| **UCPE + Plücker** | **320.80** | **6.21** | **0.1162** | **0.2047** |

**说明**: 粗分支 UCPE 与细分支 Plücker 互补——单独使用时 RotErr 仍 >6°，组合后位姿误差全面最佳。

### Table 5: 精炼器消融（60 秒基准）

| Split | Refiner | AQ↑ | IQ↑ | DD↑ | Overall↑ | RotErr↓ | TransErr↓ | CamMC↓ | IQ_{50-60}↑ | ΔIQ↓ |
|-------|---------|-----|-----|-----|----------|---------|-----------|--------|-------------|------|
| Simple | LTX-2.3 原版 | 39.73 | 38.16 | 0.00 | 71.37 | 8.65 | 2.32 | 2.35 | 35.70 | 3.73 |
| Simple | Ours (long-video) | 58.05 | 72.12 | 61.25 | 80.62 | 4.50 | 1.39 | 1.41 | 72.21 | 1.17 |
| Hard | LTX-2.3 原版 | 40.70 | 37.17 | 0.00 | 71.16 | 27.38 | 2.29 | 2.52 | 33.69 | 4.65 |
| Hard | Ours (long-video) | 56.67 | 71.38 | 91.25 | 81.89 | 8.34 | 1.39 | 1.44 | 73.03 | 0.31 |

**说明**: 长视频精炼器把 60 秒末段画质（IQ_{50-60}）从 33.69 拉到 73.03，长时漂移（ΔIQ）从 4.65 降到 0.31。

### Table 6: 各数据集质量过滤阈值

| Dataset | VMAF Motion | UniMatch | DOVER | Color Sat. | Scene Cuts | VLM Entity | VLM Quality |
|---------|-------------|----------|-------|------------|------------|------------|-------------|
| OmniWorld | [0.5, 100] | [3, 100] | [0.35, 1.0] | — | — | [0, 10] | [0.5, 1.5] |
| Sekai Game | [0.5, 50] | [3, 80] | [0.25, 1.0] | — | — | [0, 10] | [0.5, 1.5] |
| Sekai Walking | [0.5, 50] | [3, 50] | [0.35, 1.0] | [0, 180] | — | [0, 25] | [0.5, 1.5] |
| MiraData | [0.5, 50] | [3, 80] | [0.4, 1.0] | [0, 180] | ≤1 | — | — |
| DL3DV-GS | [6, 50] | [3, 80] | [0.4, 1.0] | [0, 180] | ≤1 | — | — |
| SpatialVID | [0.5, 50] | [3, 80] | [0.35, 1.0] | [0, 180] | — | [0, 10] | [0.5, 1.5] |

**说明**: 多源数据采用差异化过滤阈值——合成数据放宽运动阈值，真实数据收紧场景切换约束。

### Table 7: 训练课程与超参

| 参数 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|------|---------|---------|---------|---------|
| 目的 | Frame-wise GDN | Hybrid Attention | Minute-Scale + CamCtrl | SFT |
| 数据 | SANA-Video SFT | SANA-Video SFT | SANA-WM data | ~50K 高质片段 |
| 片段时长 | 5s | 5s | 1 min | 1 min |
| Batch/GPU | 1 | 1 | 0.5 | 0.5 |
| CP Size | – | – | 2 | 2 |
| 全局 Batch | 64 | 64 | 32 | 32 |
| 学习率 | 5e-5 | 5e-5 | 1e-5 | 1e-5 |
| 训练步数 | 30K | 30K | 31K | 10K |
| 计算预算 | ~2.75 天 | ~2 天 | ~8 天 | ~2.5 天 |

**说明**: 4 阶段课程：先适配 GDN，再开启混合注意力，再扩展到分钟级 + 相机控制，最后高质量微调。

### Table 8: 回环记忆与时间稳定性（Hard）

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | IQ_{0-10}↑ | IQ_{50-60}↑ | ΔIQ↓ |
|--------|-------|-------|--------|------------|-------------|------|
| Infinite-World | 12.04 | 0.248 | 0.617 | 73.79 | 69.63 | +4.16 |
| LingBot-World | 14.08 | 0.332 | 0.436 | 73.66 | 73.09 | +0.58 |
| HY-WorldPlay | 13.72 | 0.328 | 0.654 | 70.21 | 44.33 | +25.88 |
| Matrix-Game 3.0 | 12.17 | 0.317 | 0.556 | 69.24 | 68.92 | +0.32 |
| SANA-WM | 14.10 | 0.327 | 0.469 | 72.58 | 69.49 | +3.09 |
| **SANA-WM + refiner** | **14.80** | **0.312** | **0.458** | **73.34** | **73.03** | **+0.31** |

**说明**: PSNR 14.80 为可见方法最高，HY-WorldPlay 在 60 秒后 IQ 暴跌 25.88，凸显长视频精炼器的价值。

### Table 9: 双向 vs 自回归 Stage-1（Hard）

| Mode | RotErr↓ | TransErr↓ | CamMC↓ | Overall↑ | PSNR↑ | LPIPS↓ | ΔIQ↓ |
|------|---------|-----------|--------|----------|-------|--------|------|
| 双向（Bidir.） | 3.17 | 1.08 | 1.09 | 80.18 | 13.78 | 0.432 | +2.13 |
| 自回归（AR） | 10.02 | 1.66 | 1.72 | 79.60 | 14.10 | 0.469 | +3.09 |

**说明**: 双向变体在位姿精度上显著领先（3.17° vs 10°），但 AR 更适合在线滚动部署；论文主表用 AR 报告以匹配实际使用场景。

### Table 10: VBench-I2V 各维度消融分

| Model | CM↑ | IS↑ | IB↑ | SC↑ | BC↑ | MS↑ | DD↑ | AQ↑ | IQ↑ |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Sana-Video | 0.4755 | 0.9312 | 0.9545 | 0.8267 | 0.8841 | 0.9613 | 0.9976 | 0.5738 | 0.6376 |
| + LTX2 VAE | 0.3942 | 0.9309 | 0.9622 | 0.8346 | 0.9048 | 0.9560 | 0.9902 | 0.5395 | 0.6636 |
| + Hybrid attention | 0.4343 | 0.9450 | 0.9693 | 0.8564 | 0.9114 | 0.9611 | 0.9602 | 0.5649 | 0.6765 |

**说明**: 切到 LTX2 VAE 时 CM/AQ 略降但 IB/SC 显著上升，混合注意力进一步推高语义一致性。

### Table 11: 资产与工具清单

| 资产/工具 | 用途 | 许可 |
|-----------|------|------|
| SpatialVID-HQ | 真实视频训练源 | CC-BY-NC-SA 4.0 |
| DL3DV-10K | 静态场景 + GT 位姿 + 3DGS 增强 | DL3DV 自定义条款 |
| OmniWorld | 合成 / 游戏数据 + 相机验证 | CC-BY-NC-SA 4.0 |
| Sekai | 游戏与步行视频 | 公开发布 |
| MiraData | 长真实视频源 | GPL-3.0 |
| VIPE | 相机位姿标注引擎 | Apache-2.0 |
| Pi3X / Pi3 | 位姿/深度恢复 | BSD-3 / 权重 CC BY-NC 4.0 |
| MoGe-2 | 度量尺度深度先验 | MIT/Apache 类 |
| FCGS | 3D Gaussian Splatting 重建 | 公开研究代码 |
| DiFix3D | 3DGS 渲染视频细化 | NVIDIA 非商业 |
| Qwen3.5 VLM | 内容过滤与字幕 | Apache-2.0 |
| Nano Banana Pro | 基准首帧生成 | Google/Gemini |
| LTX-2 / LTX-2.3 | LTX2 VAE 与长视频精炼器 | LTX-2 社区许可 |

**说明**: 全部组件均使用公开 / 非商业研究许可，可复现性高。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| SpatialVID-HQ | 158K | 真实，10s | 主训练 |
| DL3DV + GS Refined | 20.5K | 真实 + 合成长片段 | 训练 |
| OmniWorld / Sekai / MiraData | 34K | 长视频 60s | 训练 + 评估 |
| 自建 1-min Benchmark | 80 场景 × Simple/Hard | 4 类首帧 × 多轨迹 | 评估 |
| VBench-I2V | 标准 5s 基准 | 9 维度量 | 消融 |

### 实现细节

- **Backbone**: SANA-Video（2.6B）+ LTX2 VAE
- **精炼器**: 17B 长视频精炼器（截断 σ 流匹配训练）
- **优化器**: AdamW + BF16 + 梯度裁剪 0.5
- **Batch**: 全局 32-64，CP size = 2
- **训练硬件**: 64 张 H100
- **总训练时间**: ~15 天
- **推理硬件**: 单张 H100 完整 / 单张 RTX 5090 蒸馏版（34s/clip）

### 可视化结果

- **Figure 5 / 11** 显示 60 秒 Hard 轨迹下基线（Matrix-Game 3.0、HY-WorldPlay）出现物体漂移与纹理崩坏，SANA-WM 全程保持几何稳定。
- **Figure 12** 通过 Pi3X 对生成视频做 3D 重建，得到的点云连贯，验证了输出的 3D 一致性。
- **Figure 7(b)** 显示纯 softmax 在 ~30 秒即 OOM，混合架构在 960 帧下显存仍稳定在 ~6.5 GB。

---

## 批判性思考

### 优点

1. **首个分钟级单卡 720p 世界模型**: 64 H100 训练 + 单卡推理，门槛低，36× 吞吐量优势显著。
2. **理论严谨的注意力设计**: $1/\sqrt{D \cdot S}$ 的代数缩放有谱半径分析支撑，是少见的把工程稳定性与数学保证结合的工作。
3. **粗-细双分支相机控制**: UCPE + Plücker 把位姿信息按时空尺度分离注入，4.5° 旋转误差在 60s 长度下是 SOTA。
4. **长视频精炼器范式**: 截断 σ 流匹配让 17B 短视频先验"嫁接"到 60 秒视频上，ΔIQ 降到 0.31。
5. **完整可复现**: 数据来源、过滤阈值、训练课程全部公开，对开源社区贡献度高。

### 局限性

1. **未上机器人下游任务**: 论文未把世界模型接入策略学习或 [[基于模型的强化学习]] 做闭环评估，与 [[Dreamer 4]]、[[Vid2World]] 等工作的"模拟器价值"差距未量化。
2. **动态物体交互弱**: 训练数据以静态场景 + 自由相机为主，缺乏机器人手臂、人物等显式可交互对象，对操作类 [[世界模型]] 而言迁移性存疑。
3. **AR 模式精度下降明显**: 双向 vs AR 的 RotErr 从 3.17° → 10°（Hard），说明 chunk-causal 部署仍有较大精度代价。
4. **精炼器代价高**: 17B 额外参数在端侧不现实，蒸馏版（RTX 5090, 34s/clip）质量损失论文未详细报告。
5. **数据集合规性参差**: SpatialVID-HQ、OmniWorld 使用 CC-BY-NC-SA 限制商用。

### 潜在改进方向

1. **接入策略学习**: 把 SANA-WM 作为 dream rollout 环境用于 [[策略]] 训练，量化 ΔSR 增益。
2. **动作条件注入**: 在 Plücker 之外加入末端执行器位姿 / 物体姿态条件，向 [[CtrlWorld]]、[[DAWN]] 这类显式动作可控世界模型靠拢。
3. **小型化精炼器**: 用模型蒸馏 + LoRA 把 17B 精炼器压缩到 <5B。
4. **MoE 化 GDN**: 在 GDN 的状态矩阵 $S_t$ 上加入 [[MoE]] 路由，提高 capacity 利用率。

### 可复现性评估
- [x] 代码开源（项目主页声明）
- [x] 预训练模型（项目主页声明 2.6B 开源）
- [x] 训练细节完整（Stage 1-4 超参齐全）
- [x] 数据集可获取（全部公开数据）

---

## 关联笔记

### 基于
- [[扩散变换器]]: 主干架构源自 DiT
- [[Flow Matching]]: 训练与精炼器均基于流匹配范式
- [[Wan2.2]]: VAE 与 backbone 设计参考
- [[CogVideoX]]: I2V 任务的前置代表工作

### 对比
- [[CtrlWorld]]: 同样追求精确相机控制的世界模型
- [[DAWN]]: 长视频驱动世界模型
- [[EA-WM]]: 动作-视觉对齐的机器人世界模型
- [[CogVideoX]]: 标准 T2V/I2V baseline
- [[Wan2.2]]: 通用视频生成 backbone

### 方法相关
- [[线性注意力]]: GDN 派生基础
- [[Frame-wise Gated DeltaNet]]: 核心创新
- [[Ray-Local UCPE]]: 粗相机分支
- [[Plücker 射线表示]]: 细相机分支
- [[截断 σ 流匹配]]: 精炼器训练目标
- [[Context-Parallel 训练]]: 分布式训练
- [[chunk-causal]]: 在线滚动推理
- [[LTX-2 VAE]]: 高压缩潜空间

### 硬件/数据相关
- [[VIPE]]: 相机位姿标注引擎
- [[Pi3X]]: 长序列位姿/深度恢复
- [[MoGe-2]]: 度量尺度深度先验
- [[VBench]]: 视频生成基准

---

## 速查卡片

> [!summary] SANA-WM
> - **核心**: 2.6B 混合线性 DiT + 双分支相机控制 + 17B 长视频精炼器
> - **方法**: 15 GDN + 5 softmax 交错 / UCPE + Plücker / 截断 σ 流匹配
> - **结果**: 60s 720p 单卡 H100，RotErr 4.5°，吞吐量 24 vid/h（36× 基线）
> - **代码**: https://nvlabs.github.io/Sana/WM/

---

*笔记创建时间: 2026-05-15*
