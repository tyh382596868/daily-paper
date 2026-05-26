---
title: "GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation"
method_name: "GAF"
authors: [Ying Chai, Litao Deng, Ruizhi Shao, Jiajun Zhang, Kangchen Lv, Liangjun Xing, Xiang Li, Hongwen Zhang, Yebin Liu]
year: 2026
venue: ICRA 2026
tags: [world-model, 3d-gaussian-splatting, robot-manipulation, 4d-representation, diffusion-policy, dynamic-scene-modeling]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2506.14135v5
created: 2026-05-25
---

# 论文笔记：GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学、北京师范大学、Shadow AI |
| 日期 | June 2025 (v1) / May 2026 (v5) |
| 项目主页 | https://ChaiYing1.github.io/projects/GAF/ |
| 代码仓库 | https://github.com/ChaiYing1/GAF |
| 对比基线 | [[Diffusion Policy]] / [[Act3D]] / [[ManiGaussian]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.14135) / [Code](https://github.com/ChaiYing1/GAF) |

---

## 一句话总结

> GAF 把 [[3D Gaussian Splatting|3D 高斯]] 扩展成带可学习运动属性的 4D 表征，从两张未标注视图同时输出当前重建、未来预测和初始动作，再用动作-视觉对齐的去噪框架精修，把策略学习的范式从 V-A / V-3D-A 推进到 [[V-4D-A 范式|V-4D-A]]。

---

## 核心贡献

1. **[[V-4D-A 范式]]**: 提出"视觉 → 4D 动态表征 → 动作"的新范式，相比已有的 V-A（直接 RGB→动作）和 V-3D-A（静态 3D→动作），首次在显式 3D 几何中加入时间维度，让动作推断建立在动态感知而非静态几何之上。
2. **[[Gaussian Action Field]] 表征**: 将 [[3D Gaussian Splatting|3DGS]] 的每个高斯增加可学习位移属性 $\Delta\mu$，统一表达当前场景、未来场景和夹爪运动，仅需 ICP 即可从场景流抽取初始动作。
3. **前馈式 4D 重建网络**: 基于 [[MASt3R]] 初始化的 [[ViT|Vision Transformer]] 直接吃两张未标注视图，由三个专门 head 同时输出高斯中心、高斯外观和运动位移，**不需要 SfM / 相机位姿**。
4. **[[动作-视觉对齐去噪|Action-Vision-Aligned Denoising]]**: 把 ICP 得到的初始动作渲染回多视图图像，与噪声动作一起送入扩散去噪器精修，使最终动作既受任务奖励驱动又受场景几何约束。
5. **强实证结果**: RLBench 9 任务平均成功率 60.4%，超 Diffusion Policy 15.7%，超 Act3D 7.3%，超 ManiGaussian 10.3%；场景重建 PSNR +11.5 dB，未来预测 PSNR +10.5 dB；真实 Franka 平台 5 任务平均 38/50 成功。

---

## 问题背景

### 要解决的问题

视觉驱动的机器人操作策略需要精准感知场景，但现有范式难以同时满足三点：

1. **3D 空间理解**：纯 2D 策略缺乏度量几何，对位姿变化鲁棒性差
2. **时间动态建模**：静态 3D 表征不知道场景下一步会如何演化
3. **几何 → 动作的可解释映射**：从视觉表征直接回归动作过于隐式，缺乏几何约束

### 现有方法的局限

- **V-A (Vision-to-Action) 范式**：如 [[Diffusion Policy]]、RT-2 等，直接 RGB→动作映射，缺乏显式 3D [[世界模型]]
- **V-3D-A (Vision-to-3D-to-Action) 范式**：如 [[Act3D]]、PerAct、RVT 等使用静态 3D 表征（体素 / 点云 / NeRF），但无法捕捉场景的时序演化
- **基于高斯的世界模型**：如 [[ManiGaussian]] 引入 3D 高斯做世界建模，但仍把动作建模与 3D 重建解耦

### 本文的动机

如果机器人能够同时"看见现在的 3D 场景"和"预见 Δt 之后的 3D 场景"，那么从这两组高斯之间的位移就能直接读出夹爪应该如何运动——这把策略问题转化为 4D 场景演化预测问题，让动作有了显式的几何根基。这就是 [[V-4D-A 范式|V-4D-A]] 的核心思想。

---

## 方法详解

### 模型架构

<!-- 严禁 ASCII 流程图，使用结构化 Markdown 列表 -->

GAF 采用 **前馈 4D 高斯重建 + 动作-视觉对齐去噪** 的两阶段架构：

- **输入**: 当前时刻两张未标注图像 $\{I_v^t, k_v^t\}_{v=1}^V$（$V=2$，含相机内参 $k_v^t$）
- **几何 Backbone**: [[ViT|Vision Transformer]]，权重用 [[MASt3R]] 初始化，提取无需相机位姿的 3D-aware 特征
- **三个专门 Head**:
  - $h_{center}$: 预测每个像素对应的高斯中心 $\mu_j^t \in \mathbb{R}^3$
  - $h_{feature}$: 预测高斯外观属性 $f_j = \{c_j, \sigma_j, r_j, s_j\}$（颜色、不透明度、旋转、尺度）
  - $h_{motion}$: 预测每个高斯从 $t$ 到 $t+\Delta t$ 的位移 $\Delta\mu_j^{t\to t+\Delta t}$
- **三组输出**:
  - 当前高斯场 $\{\mu_j^t, f_j\}$：可渲染当前视图
  - 未来高斯场 $\{\mu_j^t + \Delta\mu_j, f_j\}$：可渲染 $\Delta t$ 之后的视图
  - 初始动作 $a_{init}$：用 [[ICP]] 从夹爪相关高斯的位移抽取的刚体变换序列
- **动作精修器**: [[动作-视觉对齐去噪]] 框架，把 $a_{init}$ 渲染回多视图作为条件，对动作做扩散去噪
- **输出**: 8 步动作块 $a_{t:t+7}$（位置 + 旋转 + 夹爪开合）

### 核心模块

#### 模块 1: [[Gaussian Action Field]]（4D 表征）

**设计动机**: 在 [[3D Gaussian Splatting|3DGS]] 静态外观属性之外，给每个高斯增加一条可学习的位移属性 $\Delta\mu$，让单一表征同时承载"几何 + 外观 + 运动"三类信息。机器人夹爪、被操物体、背景的运动差异都会体现在 $\Delta\mu$ 的分布中。

**具体实现**: 把 GAF 视作一个连续函数 $\mathcal{F}_\Theta$（详见公式 1），输入是高斯索引和时间 $t$，输出包含位置 $\mu$、位移 $\Delta\mu$、外观特征 $f$。

#### 模块 2: 前馈式动态高斯重建

**设计动机**: 已有 4D 高斯方法多依赖 SfM 或已知位姿，对未标定的机器人 RGB 流不友好。这里用 [[MASt3R]] 的 3D 几何先验绕开 SfM。

**具体实现**:
- ViT 特征通过 $h_{center}$ / $h_{feature}$（合称 $\mathcal{H}_{Gauss}$）生成像素对应的高斯集合，共 $V \times H \times W$ 个高斯（见公式 2）
- 平行地通过 $h_{motion}$ 输出位移场（见公式 3）
- 使用 [[3D Gaussian Splatting|高斯泼溅]] 渲染（见公式 4）
- 同时监督当前帧和 $t+\Delta t$ 未来帧（见公式 5 的复合损失）

#### 模块 3: 初始动作抽取（[[ICP]]）

**设计动机**: 一旦能预测未来高斯的位置，夹爪上的高斯位移就直接对应夹爪的刚体变换 $T^{t\to t+\Delta t}$。

**具体实现**: 在所有"属于夹爪"的高斯子集上做 [[ICP]] 求解最优刚体变换（公式 6），把变换插值成 8 步动作序列 $a_{init}$。

#### 模块 4: [[动作-视觉对齐去噪]]

**设计动机**: 初始动作 $a_{init}$ 由几何驱动，但忽略任务语义；要把它当作"几何先验"再用扩散去噪精修，让最终动作既贴合场景几何又满足任务目标。

**具体实现**:
- 把候选动作 $a$ 通过相机投影渲染成夹爪掩码 $R^c$（公式 7）：把当前夹爪点云在世界系下移动到 $a$，再投影回每个相机
- 多视图 $R^c$ 与原始观测拼接作为条件，对带噪动作 $a + \epsilon$ 做去噪
- 联合监督：去噪方向、噪声估计、夹爪开合状态（公式 8）
- 训练用 [[DDIM]] 50 步，推理只跑 3 步，单次 inference < 0.3 秒

---

## 关键公式

### 公式 1: [[Gaussian Action Field|GAF 连续表征]]

$$
\mathcal{F}_\Theta : \{g(x), t\} \mapsto \{\mu, \Delta\mu, f\}
$$

**含义**: GAF 是一个参数化连续函数 $\mathcal{F}_\Theta$，输入高斯索引 $g(x)$ 和时间 $t$，输出高斯位置、位移和外观特征三元组——这是论文最核心的表征定义。

**符号说明**:
- $g(x)$: 高斯索引（按像素索引）
- $t$: 当前时间步
- $\mu \in \mathbb{R}^3$: 当前 3D 位置
- $\Delta\mu \in \mathbb{R}^3$: 时间窗口内的位移
- $f = \{c, \sigma, r, s\}$: 外观参数（颜色 / 不透明度 / 旋转 / 尺度）

### 公式 2: 高斯参数预测 head

$$
\mathcal{H}_{Gauss}\!\left( \text{ViT}(\{I_v^t, k_v^t\}_{v=1}^V) \right) = \{\mu_j^t, c_j^t, \sigma_j^t, r_j^t, s_j^t\}_{j=1}^{V \times H \times W}
$$

**含义**: $\mathcal{H}_{Gauss} = \{h_{center}, h_{feature}\}$ 把 [[ViT]] 特征解码为每个像素对应的一个高斯，覆盖 $V$ 视图、$H \times W$ 像素。

**符号说明**:
- $I_v^t$: 第 $v$ 视图、第 $t$ 时刻的图像
- $k_v^t$: 对应相机内参（仅内参，不需要外参）
- $\mu_j^t$: 像素 $j$ 在 $t$ 时刻的 3D 位置
- $c_j, \sigma_j, r_j, s_j$: 颜色、不透明度、旋转、尺度

### 公式 3: 运动预测 head

$$
h_{motion}\!\left( \text{ViT}(\{I_v^t, k_v^t\}_{v=1}^V) \right) = \{\Delta\mu_j^{t \to t+\Delta t}\}_{j=1}^{V \times H \times W}
$$

**含义**: 独立的运动 head 从同一组 ViT 特征预测每个高斯在窗口 $[t, t+\Delta t]$ 的 3D 位移，把 3DGS 扩展为 4D。

**符号说明**:
- $\Delta\mu_j^{t \to t+\Delta t} \in \mathbb{R}^3$: 第 $j$ 个高斯的位移向量

### 公式 4: [[3D Gaussian Splatting|高斯泼溅渲染]]

$$
C(p) = \sum_{i=1}^N \alpha_i c_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

**含义**: 沿像素 $p$ 的视线对所有可见高斯做前向 alpha-合成，得到该像素颜色。这是 3DGS 标准的可微渲染公式，论文用它分别渲染当前帧（用 $\mu$）和未来帧（用 $\mu + \Delta\mu$）。

**符号说明**:
- $C(p)$: 像素 $p$ 的渲染颜色
- $\alpha_i$: 第 $i$ 个高斯在像素 $p$ 处的 2D 投影密度
- $c_i$: 高斯颜色（球谐展开）
- $N$: 沿视线的有效高斯数

### 公式 5: GAF 训练损失

$$
\mathcal{L}_{GAF} = \mathcal{L}_{LPIPS}^{t} + \mathcal{L}_{MSE}^{t} + \mathcal{L}_{LPIPS}^{t+\Delta t} + \mathcal{L}_{MSE}^{t+\Delta t}
$$

**含义**: 对当前帧和未来帧分别施加 [[MSE]] + [[LPIPS]] 的双重图像监督，把几何保真度和感知相似度都纳入。注意论文**没有动作 GT 监督**——动作信号完全经由视觉重建反向传播到位移属性。

**符号说明**:
- $\mathcal{L}_{MSE}^{t}, \mathcal{L}_{MSE}^{t+\Delta t}$: 当前 / 未来帧的像素 [[MSE]] 损失
- $\mathcal{L}_{LPIPS}^{t}, \mathcal{L}_{LPIPS}^{t+\Delta t}$: 当前 / 未来帧的 [[LPIPS]] 感知损失

### 公式 6: 初始动作的 [[ICP]] 抽取

$$
T^{t \to t+\Delta t} = \arg\min_{T \in SE(3)} \sum_{k \in \text{gripper}} \| T(\mu_k) - (\mu_k + \Delta\mu_k) \|^2
$$

**含义**: 在属于夹爪的高斯子集上求解最优刚体变换 $T \in SE(3)$，把"3D 位移场"压缩为"夹爪刚体动作"，再线性插值出 8 步初始动作序列 $a_{init}$。

**符号说明**:
- $T^{t \to t+\Delta t}$: 夹爪的 $SE(3)$ 刚体变换
- $k \in \text{gripper}$: 属于夹爪部分的高斯索引集合（通过预定义 mask）
- $\mu_k$: 当前位置
- $\mu_k + \Delta\mu_k$: 预测的未来位置

### 公式 7: 夹爪位姿渲染（动作可视化）

$$
R^c = \text{Render}\!\left( T_{w2c} \cdot T_g^w \cdot a, \, K^c \right)
$$

**含义**: 把候选动作 $a$ 视为夹爪在世界系下的目标位姿，先把夹爪模型变换到该位姿、再用相机外参 $T_{w2c}$ 和内参 $K^c$ 投影到第 $c$ 个相机，得到一张夹爪掩码图。这张图与原始观测拼接后作为去噪条件。

**符号说明**:
- $a$: 候选动作（位姿）
- $T_g^w$: 夹爪在世界系下的当前位姿
- $T_{w2c}$: 世界到相机的外参变换
- $K^c$: 第 $c$ 个相机的内参
- $R^c$: 第 $c$ 视图下的夹爪渲染图像

### 公式 8: [[动作-视觉对齐去噪|去噪精修损失]]

$$
\mathcal{L}_{refine} = L_1(D, D^{gt}) + L_1(\epsilon, \epsilon^{gt}) + \text{BCE}(g, g^{gt})
$$

**含义**: 动作扩散去噪的联合监督：去噪方向回归（$D$）+ 噪声预测回归（$\epsilon$）+ 夹爪开合二分类（$g$），把连续位姿与离散夹爪状态同步优化。

**符号说明**:
- $D, D^{gt}$: 预测 / 真值的去噪方向（即每步位姿增量）
- $\epsilon, \epsilon^{gt}$: 预测 / 真值的噪声
- $g, g^{gt}$: 预测 / 真值的夹爪开合状态（二值）
- $L_1$: L1 损失，$\text{BCE}$: 二元交叉熵

---

## 关键图表

### Figure 1: V-A / V-3D-A / V-4D-A 三种范式对比

![Figure 1](https://arxiv.org/html/2506.14135v5/x1.png)

**说明**: 论文最核心的概念图。上排是 V-A 范式（RGB→Action，无显式 3D），中排是 V-3D-A（先重建静态 3D 再回归动作），下排是本文提出的 [[V-4D-A 范式|V-4D-A]]：先建动态 4D 高斯场，再从场景流抽取动作。直观体现了"加上时间维度"的必要性。

### Figure 2: GAF 重建模块总览

![Figure 2](https://arxiv.org/html/2506.14135v5/x2.png)

**说明**: 前馈 4D 重建子网络的详细结构。两张未标注图像 + 内参输入 [[ViT]]（[[MASt3R]] 初始化），分流到 $h_{center}$、$h_{feature}$、$h_{motion}$ 三个 head。前两个组合成当前高斯 $\{\mu, f\}$，加上位移 $\Delta\mu$ 得到未来高斯，再渲染回 $t$ 和 $t+\Delta t$ 两帧做监督。

### Figure 3: 操作流水线（动作-视觉对齐去噪）

![Figure 3](https://arxiv.org/html/2506.14135v5/x3.png)

**说明**: 整体 manipulation pipeline。GAF 重建 → 夹爪高斯子集 [[ICP]] → 初始动作 $a_{init}$ → 把候选动作渲染回多视图（公式 7）→ 与噪声动作拼接送入扩散去噪 → 输出最终 8 步动作块。

### Figure 4: 重建与未来预测定性对比

![Figure 4](https://arxiv.org/html/2506.14135v5/x4.png)

**说明**: 当前帧重建（左半）与未来帧预测（右半）的 [[ManiGaussian]] vs GAF 定性对比。GAF 的重建明显更锐利、未来帧的夹爪位置预测更准。这也是 +11.5 dB PSNR 提升的视觉解释。

### Figure 5: 去噪框架消融

![Figure 5](https://arxiv.org/html/2506.14135v5/x5.png)

**说明**: 去掉 [[动作-视觉对齐去噪]] 后的失败案例。仅靠 ICP 抽取的初始动作 $a_{init}$ 在重建误差较大的区域会偏离目标（如夹爪未对准物体），而加入去噪精修后能纠正偏差。

### Figure 6: 空间泛化（22 任务 grid sampling）

![Figure 6](https://arxiv.org/html/2506.14135v5/x6.png)

**说明**: 在 22 个 RLBench 任务上做工作空间网格采样，比较 GAF 与 [[Diffusion Policy]] 在边界 / 角落位置的成功率热图。GAF 在远离训练分布的位置仍能保持成功率，验证显式 3D 几何带来的空间外推能力。

### Figure 7: 真实世界实验设置

![Figure 7](https://arxiv.org/html/2506.14135v5/x7.png)

**说明**: 真实平台配置：Franka Emika Panda + Panda Hand 夹爪，3 台 RealSense D435i（2 外部 + 1 腕部）。展示相机布置和工作空间，对应 Table IV 的 5 个真实任务。

### Figure 8: 真实世界重建与未来预测

![Figure 8](https://arxiv.org/html/2506.14135v5/x8.png)

**说明**: 真机部署中，GAF 同时渲染出当前帧高斯和未来帧高斯——可视化展示其在 sim-to-real 后仍能稳定预测夹爪几何运动，这是模型在真实场景下成功执行 push button / close door 等任务的关键。

### Table I: RLBench 9 任务成功率（%）

| Method | Toilet Seat Down | Open Grill | Close Grill | Close Fridge | Phone On Base | Lift Lid Up | Close Microwave | Push Button | Close Laptop | **Avg.** |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [[Diffusion Policy]] | 39 | 20 | 36 | 16 | 23 | 81 | 62 | 61 | 64 | 44.7 |
| [[Act3D]] | 60 | 22 | 41 | 47 | 26 | 91 | 73 | 60 | 58 | 53.1 |
| [[ManiGaussian]] | 34 | 24 | 38 | 41 | 30 | 97 | 67 | 63 | 57 | 50.1 |
| Ours w/o GAF | 57 | 16 | 49 | 27 | 31 | 94 | 79 | 58 | 51 | 51.3 |
| **Ours with GAF** | **71** | **26** | **55** | 42 | **35** | **100** | **85** | 61 | **69** | **60.4** |

**说明**: GAF 在 9 个任务的 7 个上取得最优。w/ GAF 相比 w/o GAF 的 +9.1% 提升直接验证了 4D 表征的有效性。值得注意的是"Lift Lid Up"取得 100% 成功率。

### Table II: 场景重建与未来预测（vs [[ManiGaussian]]）

| 指标 | 当前帧 | 未来帧 |
|------|:-:|:-:|
| **[[PSNR]]** (↑) | **+11.5385 dB** | **+10.5311 dB** |
| **[[SSIM]]** (↑) | **+0.3864** | **+0.3856** |
| **[[LPIPS]]** (↓) | **-0.5574** | **-0.5757** |

**说明**: 在 3 个测试任务上对当前 / 未来视图合成全面碾压 ManiGaussian，未来帧 PSNR 提升幅度（+10.5）与当前帧（+11.5）几乎相当，证明 GAF 的位移预测确实捕捉到了真实运动而不是简单复制当前帧。

### Table III: 多任务联合训练成功率（%）

| Method | Toilet Seat Down | Close Microwave | Lift Lid Up | Close Laptop | **Average** |
|--------|:-:|:-:|:-:|:-:|:-:|
| [[Diffusion Policy]] | 55 (+16) | 50 (-12) | 39 (-42) | 18 (-46) | 40.5 (-21.0) |
| **GAF** | 59 (-12) | **85** (+0) | **57** (-43) | **79** (+10) | **70.0** (-11.25) |

**说明**: 4 任务联合训练。括号内是相对单任务的变化。GAF 平均仅下降 11.25%，而 DP 下降 21%，显示 GAF 的 3D 几何先验对多任务干扰更鲁棒。Close Laptop 上 GAF 甚至**反向提升** +10%。

### Table IV: 真实世界 5 任务成功率（10 次试验）

| Task | Push Button | Close Door | Open Door | Pick Cup | Place Apple | **Total** |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| GAF | **10/10** | 8/10 | 7/10 | 7/10 | 6/10 | **38/50** |

**说明**: Franka 真实平台 sim-to-real 部署。最简单的 Push Button 100% 成功，最难的 Place Apple（涉及精细抓取 + 放置）6/10。失败主要源自缺乏力反馈导致的抓取问题，而非视觉感知。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RLBench | 9 任务 × 20 demo | 仿真，CoppeliaSim，含未见过的位姿测试集 | 训练 + 主对比 |
| RLBench-22 | 22 任务 | grid sampling 工作空间 | 空间泛化测试 |
| Franka 真实数据 | 5 任务 | RealSense D435i ×3 | sim-to-real 真机部署 |

### 实现细节

- **Backbone**: [[ViT]]，权重用 [[MASt3R]] 初始化
- **输入分辨率**: 仿真 128 × 128，真机 1280 × 720
- **优化器与训练**: 80k iterations，batch size 16
- **去噪步数**: 训练 50 步 [[DDIM]]，推理 3 步
- **硬件**: 单卡 RTX A800，约 24 小时训练完成
- **推理时延**: 8 步动作预测 < 0.3 秒

### 可视化结果

- 在干扰物较多的桌面上，GAF 的未来高斯能正确预测目标物体被夹爪触发的运动模式
- 重建结果中夹爪边缘明显比 ManiGaussian 锐利（参见 Figure 4）
- 多任务训练下没有出现 [[灾难性遗忘]] 现象

---

## 批判性思考

### 优点

1. **范式贡献**：[[V-4D-A 范式|V-4D-A]] 概念清晰，把"4D 世界建模"和"动作预测"统一到同一表征下，对后续工作有提纲挈领的作用
2. **无 SfM 依赖**：用 [[MASt3R]] 先验绕开相机外参标定，对未标定真实场景非常友好
3. **可解释性强**：初始动作来自 [[ICP]] 几何匹配，不是黑盒回归，易于排查失败原因
4. **强重建提升**：+11.5 dB PSNR 是非常显著的数字，说明运动 head 的引入确实改善了场景几何，而非仅仅做了"未来预测"
5. **去噪步数极少**：推理仅 3 步去噪，远低于一般 [[Diffusion Policy]] 的 50–100 步，部署友好

### 局限性

1. **依赖夹爪 mask**：[[ICP]] 抽取初始动作需要知道哪些高斯属于夹爪，文中通过预定义 mask 实现，对新机器人形态需要重新定义
2. **位移监督形式间接**：损失里没有 3D flow 的显式监督（公式 5），位移属性靠 $t+\Delta t$ 帧的渲染误差反传，可能在纹理稀疏区域学不准
3. **单时间窗口**：只预测一个 $\Delta t$ 步的未来，对长时序任务可能不够；论文用插值得到 8 步动作的做法回避了多步预测问题
4. **缺乏力 / 触觉融合**：真实任务失败案例（Pick Cup、Place Apple）多源于缺乏力反馈，但纯视觉框架无法直接解决
5. **离散夹爪状态**：用 BCE 监督夹爪开合，未考虑连续夹持力——对柔软物体可能不友好

### 潜在改进方向

1. **多时间窗口 GAF**：同时预测 $t+\Delta t_1, t+\Delta t_2, \dots$，让动作 head 直接从多步未来抽取
2. **自动夹爪检测**：用 SAM2 / FoundationPose 自动分割夹爪，去掉手工 mask 依赖
3. **加入 3D scene flow 显式监督**：在仿真中有 GT scene flow，可作为辅助损失提高位移预测精度
4. **与 [[VLA]] 结合**：把语言指令作为额外条件，扩展到任务级泛化
5. **结合触觉/力觉**：把 tactile 信号作为去噪条件，缓解抓取阶段的失败

### 可复现性评估

- [x] 代码开源（GitHub: ChaiYing1/GAF）
- [x] 项目主页齐全
- [x] 训练细节完整（80k iter / batch 16 / RTX A800 / 24h）
- [x] 数据集可获取（RLBench 开源）
- [ ] 真实世界数据未公开

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: 基础 3D 表征
- [[MASt3R]]: 提供 ViT backbone 的 3D-aware 初始化权重
- [[Diffusion Policy]]: 动作扩散去噪的范式基础

### 对比

- [[Diffusion Policy]]: 纯 V-A 范式基线
- [[Act3D]]: V-3D-A 静态 3D 基线
- [[ManiGaussian]]: 同样基于高斯的世界模型基线（动作建模与重建解耦）
- [[GaussianDream]]: 同期的前馈式高斯世界模型工作，但定位是 [[VLA]] 插件而非独立策略

### 方法相关

- [[Gaussian Action Field]]: 本文核心表征
- [[V-4D-A 范式]]: 本文提出的策略学习新范式
- [[动作-视觉对齐去噪]]: 动作精修核心模块
- [[ICP]]: 初始动作抽取算法
- [[DDIM]]: 扩散采样器
- [[ViT]]: 骨干网络
- [[Flow Matching]]: 同类去噪范式可对比

### 评估相关

- [[PSNR]] / [[SSIM]] / [[LPIPS]]: 重建质量度量
- [[世界模型]]: 上位概念
- [[Embodied AI]]: 上位概念

---

## 速查卡片

> [!summary] GAF (ICRA 2026)
> - **核心**: 给 3D 高斯加上可学习位移 $\Delta\mu$ 变成 4D 表征，从场景流抽取动作再扩散精修
> - **方法**: V-4D-A 范式 = 前馈 4D 高斯重建（ViT + 三个 head）+ ICP 初始动作 + 多视图渲染条件下的动作扩散去噪
> - **结果**: RLBench 9 任务 60.4%（+15.7% over DP），重建 PSNR +11.5 dB，真机 5 任务 38/50
> - **代码**: https://github.com/ChaiYing1/GAF

---

*笔记创建时间: 2026-05-25*
