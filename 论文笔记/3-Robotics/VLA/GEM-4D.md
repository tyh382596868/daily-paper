---
title: "GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation"
method_name: "GEM-4D"
authors: [Kaichen Zhou, Yuzhen Chen, Fangneng Zhan, Hang Hua, Grace Chen, Xinhai Chang, Ao Qu, Yilun Du, Zhuang Liu, Paul Pu Liang, Mengyu Wang]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, robot-manipulation, geometry-distillation, flow-matching, inverse-dynamics]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.22882v1
created: 2026-05-25
---

# 论文笔记：GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Harvard AI and Robotics Lab; MIT Media Lab & EECS; Princeton CS; MIT-IBM Watson AI Lab |
| 日期 | May 2026 |
| 项目主页 | https://anonymous-submission-20.github.io/gem.github.io/ |
| 对比基线 | [[TesserAct]] / [[CogVideoX]] / [[Wan2.2]] / Geometry Forcing |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22882) / [HTML](https://arxiv.org/html/2605.22882v1) / Code: 暂未公开 |

---

## 一句话总结

> GEM-4D 把冻结的 4D 几何基础模型当作"对应一致性教师"，通过双流 [[Flow Matching]] 把几何特征蒸馏进视频 DiT 的中间表示，单流推理零额外开销，把 Droid 真机操作成功率从 61% 拉到 81%。

---

## 核心贡献

1. **原理层面**：形式化了"几何监督即对应监督"的关系——证明 4D 几何基础模型的特征本质上是密集帧间对应表示，对它做特征蒸馏等价于在表示层加 [[对应一致性]] 正则。
2. **架构层面**：提出**双流 Flow Matching**，[[几何 DiT]] 仅以视频骨干的中间特征 $\mathbf{m}_t$ 作为唯一的场景条件（[[Asymmetric Conditioning|非对称条件]]），强迫视频骨干编码几何信息，推理时直接丢弃几何分支。
3. **系统层面**：设计 [[Adaptive Inverse Dynamic System|AIDS 逆动力学系统]]，把"对应一致"的视频翻译成可执行 6-DoF 轨迹，在真实 Droid 上 +20 个百分点（61%→81%），RLBench 7 任务平均 63–82%。

---

## 问题背景

### 要解决的问题

视频世界模型（[[UniPi]]、[[TesserAct]] 等）能从一句指令生成看上去合理的未来帧，但作为机器人 planner 时存在致命问题：**像素级合理 ≠ 几何一致**。要从生成视频里提取动作，必须保证"同一 3D 表面点的像素在帧间稳定地对应"——而当前视频扩散模型完全无法保证这一点。

### 现有方法的局限

帧间对应由相机运动、场景深度、物体运动三者共同决定（见公式 1）。直接做像素重建损失对这三者全部不可知：

- **TesserAct / 3DFlowAction**：直接在输出空间增加深度/法向量/3D flow 头，需要为几何额外建模、改架构、增推理成本。
- **像素级损失**：物体非刚性形变、接触漂移、深度跳变在视觉上不易察觉，却根本性破坏动作提取。
- **Geometry Forcing**：做特征对齐但只用静态深度先验，与机器人操作的动态场景不匹配。

### 本文的动机

如果有现成的 [[4D 几何基础模型]]（[[PAGE-4D]]、[[Depth Anything V3]]、[[VGGT]]、[[DUSt3R]] 系列）能编码密集帧间对应，**为什么要在输出空间复刻它，而不是直接把它的特征作为表示级正则注入视频骨干？** 这样推理时几何分支可以丢弃，零额外成本。

---

## 方法详解

### 模型架构

<!-- 见 Figure 2 -->

GEM-4D 在训练时维持两个并行的 [[Flow Matching]] 过程，推理时只用视频分支：

- **输入**：语言指令 $c$ + 首帧观测 $\mathbf{I}_0$ + 噪声视频潜变量 $\mathbf{z}_t$
- **视频 DiT**：[[扩散变换器|DiT]] 骨干 $E_\theta^{\text{vid}}$ 抽取中间特征 $\mathbf{m}_t$，输出头 $U_\theta^{\text{vid}}$ 预测视频速度场 $\mathbf{v}_\theta^{\text{vid}}$（公式 3）
- **几何 DiT**：参数 $\psi$，输入噪声几何潜变量 $\mathbf{g}_t$，**唯一**的场景条件就是视频骨干的 $\mathbf{m}_t$（[[Asymmetric Conditioning|非对称]]：几何 DiT 读 $\mathbf{m}_t$，但梯度反传迫使 $\mathbf{m}_t$ 编码几何）
- **几何监督源**：冻结的 [[PAGE-4D]] / [[Depth Anything V3]] / [[VGGT]] 抽取 $\mathbf{g}_0$，作为几何 flow matching 的目标
- **输出**：视频潜变量 → VAE 解码为 RGB 视频，再交给 [[AIDS|Adaptive Inverse Dynamic System]] 转成 6-DoF 动作

### 核心模块

#### 模块1: 视频流匹配 (Video Flow Matching)

**设计动机**：采用 [[Flow Matching]] 的线性轨迹便于把"几何监督等价于在中间表示上加约束"这一性质严格写出来（公式 7 的梯度分解）。

**具体实现**：
- 视频潜变量 $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$，目标速度 $\mathbf{v}^* = \mathbf{z}_1 - \mathbf{z}_0$
- 损失：见公式 2
- DiT 输出中间特征 $\mathbf{m}_t$ 在中层抽出（公式 3），既送给速度头，也送给几何 DiT 作为条件

#### 模块2: 几何蒸馏分支 (Geometry Distillation)

**设计动机**：4D 几何基础模型在大规模真实视频上学到的特征本身就是"密集对应表示"——它的特征空间天然编码相机位姿、深度、scene flow 三大对应因素。让视频骨干学会预测这种特征，相当于在 $\mathbf{m}_t$ 上加了对应一致性正则。

**具体实现**：
- 冻结的几何 FM 抽特征 $\mathbf{g}_0 = G(\{\mathbf{I}_t\}_{t=0}^T) \in \mathbb{R}^{T \times \frac{H}{P} \times \frac{W}{P} \times C}$（公式 4）
- 噪声几何潜变量 $\mathbf{g}_t = (1-t)\mathbf{g}_0 + t\mathbf{g}_1$，由 [[几何 DiT]] $\mathbf{v}_\psi^{\text{geo}}(\mathbf{g}_t, t, \mathbf{m}_t)$ 拟合
- **关键设计**：几何 DiT 只看 $\mathbf{m}_t$，不直接看像素或文本——这强迫梯度通过 $\mathbf{m}_t$ 倒灌回视频骨干（公式 7）

#### 模块3: 联合训练目标

**设计动机**：单一权重 $\alpha$ 调节"看起来真实"与"几何一致"的折中。

**具体实现**：
- 公式 6：$\mathcal{L} = \mathcal{L}_{\text{FM}}^{\text{vid}} + \alpha \mathcal{L}_{\text{FM}}^{\text{geo}}$
- 梯度分解（公式 7）显示：几何项对 $\theta$ 的影响完全通过共享表示 $\mathbf{m}_t$ 传递，无需修改视频输出空间
- 推理时直接丢弃几何 DiT，与原始视频 DiT 同样的 FLOPs

#### 模块4: [[Adaptive Inverse Dynamic System|AIDS 逆动力学系统]]

<!-- 见 Figure 3 -->

把"对应一致"的视频 → 可执行 6-DoF 轨迹，分 4 步：

**Step 1 — 3D Scene Grounding**：用 [[Qwen3.5-VL]] + [[SAM-2]] 从指令和深度图生成目标物体掩膜与末端执行器掩膜 $\mathcal{M}_{\text{ee}}$，用 [[FoundationPose]] 把 EE CAD 模型对齐到点云，得到初始位姿 $(\mathbf{R}_{\text{ee}}^0, \mathbf{T}_{\text{ee}}^0) \in SE(3)$。

**Step 2 — Dual-Criterion Confidence-Gated Tracker**：用 [[CoTracker3]] 沿 $\mathcal{M}_{\text{ee}}$ 采样的密集 keypoint 在生成视频上做跟踪，监控两个信号——保留比例 $s_t$ 与帧间变化 $\Delta s_t$（公式 8）。区分**渐进漂移**（$s_t$ 缓慢低于阈值 $\tau$）与**突发崩溃**（$\Delta s_t$ 急降超过 $\delta$），分别用"重锚定 tracker"与"用 Qwen3.5-VL 重新定位 EE mask"两种策略修复（公式 9）。

**Step 3 — Geometry-Kinematics Pose Fallback**：每帧用 [[FoundationPose]] 出 6-DoF 位姿 + 置信度 $\kappa_t$（公式 10）。低置信度或位姿跳变过大（公式 11）时拒收 FP 估计，转用**点云质心反投影**恢复平移、用 [[Slerp|球面线性插值]] 在前后可信帧之间恢复旋转。

**Step 4 — Grasp Insertion & Action Synthesis**：从恢复的 EE 轨迹挑出离目标物体最近的参考位姿 $(\mathbf{R}_{\text{ref}}, \mathbf{T}_{\text{ref}})$，用 [[GraspGen]] 在物体点云上枚举候选抓取，按公式 12 的加权位姿偏差排序选最优。把最佳抓取插入轨迹中作为中间目标，再做平滑 + 逆运动学得到关节动作序列 $\{\mathbf{a}_t\}_{t=0}^{N-1}$。

---

## 关键公式

### 公式1: [[相机投影|帧间对应像素投影]]

$$
\mathbf{p}_{t+1} \sim \mathbf{K}\left[\mathbf{R}_{t \to t+1}\,\mathbf{D}(\mathbf{p}_t)\,\mathbf{K}^{-1}\mathbf{p}_t + \mathbf{T}_{t \to t+1} + \Delta\mathbf{X}_t\right]
$$

**含义**：同一 3D 点在 $t \to t+1$ 帧的像素对应由**相机内参 + 相机外参 + 深度 + 场景流**四者共同决定。像素级损失对这四者全部不可知，因此根本不能强制对应——这是论文整套方法的理论起点。

**符号说明**:
- $\mathbf{p}_t, \mathbf{p}_{t+1}$：$t, t+1$ 帧像素齐次坐标
- $\mathbf{K} \in \mathbb{R}^{3\times 3}$：相机内参
- $\mathbf{R}_{t\to t+1}, \mathbf{T}_{t\to t+1}$：帧间相机相对位姿
- $\mathbf{D}(\mathbf{p}_t)$：像素深度
- $\Delta\mathbf{X}_t$：[[场景流]]（物体相对相机运动）

### 公式2: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{\text{FM}}^{\text{vid}} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t}\left[\|\mathbf{v}_\theta^{\text{vid}}(\mathbf{z}_t, t, c) - \mathbf{v}^*(\mathbf{z}_t, t)\|_2^2\right]
$$

**含义**：标准视频 [[Flow Matching]]，回归从噪声到数据的解析速度场，$c$ 是语言条件。

**符号说明**:
- $\mathbf{z}_0$：标准高斯噪声潜变量
- $\mathbf{z}_1$：真实视频潜变量（VAE 编码）
- $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$
- $\mathbf{v}^* = \mathbf{z}_1 - \mathbf{z}_0$：目标速度
- $c$：语言指令嵌入

### 公式3: [[视频 DiT 中间特征|视频 DiT 解耦]]

$$
\mathbf{m}_t = E_\theta^{\text{vid}}(\mathbf{z}_t, t, c), \qquad \mathbf{v}_\theta^{\text{vid}} = U_\theta^{\text{vid}}(\mathbf{m}_t)
$$

**含义**：把视频 DiT 拆成 backbone $E_\theta$ + head $U_\theta$，**中间表示 $\mathbf{m}_t$ 成为视频与几何两个目标的共享枢纽**。

**符号说明**:
- $E_\theta^{\text{vid}}$：DiT 主干（mid-level layer 之前）
- $U_\theta^{\text{vid}}$：速度输出头（mid-level layer 之后）
- $\mathbf{m}_t$：中层特征图，几何蒸馏的注入点

### 公式4: [[4D 几何基础模型|几何特征图]]

$$
\mathbf{g}_0 = G(\{\mathbf{I}_t\}_{t=0}^T) \in \mathbb{R}^{T \times \frac{H}{P} \times \frac{W}{P} \times C}
$$

**含义**：冻结的几何 FM（PAGE-4D / Depth Anything V3 / VGGT）一次性吃下整段视频，输出 $T$ 帧 × patch 网格 × $C$ 通道的密集几何特征——本质上是密集对应表示。

**符号说明**:
- $G$：冻结的 4D 几何基础模型
- $T$：视频帧数；$H, W$：图像分辨率；$P$：patch size；$C$：特征通道
- $\mathbf{g}_0$：作为几何 Flow Matching 的目标分布

### 公式5: [[几何蒸馏|几何流匹配损失]]

$$
\mathcal{L}_{\text{FM}}^{\text{geo}} = \mathbb{E}\left[\|\mathbf{v}_\psi^{\text{geo}}(\mathbf{g}_t, t, \mathbf{m}_t) - \mathbf{v}^*_{\text{geo}}(\mathbf{g}_t, t)\|_2^2\right]
$$

**含义**：[[几何 DiT]] 学习从噪声到几何特征 $\mathbf{g}_0$ 的速度场，**唯一的场景条件是视频骨干的 $\mathbf{m}_t$**——这是非对称信息流的关键。

**符号说明**:
- $\mathbf{g}_t = (1-t)\mathbf{g}_0' + t\mathbf{g}_0$：几何潜变量插值
- $\mathbf{v}_\psi^{\text{geo}}$：几何速度网络，参数 $\psi$（仅训练用）
- $\mathbf{m}_t$：从公式 3 来的视频中间特征
- 注意：几何 DiT **不接收** $\mathbf{z}_t$ 或 $c$ 作为输入

### 公式6: 联合训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{FM}}^{\text{vid}} + \alpha\,\mathcal{L}_{\text{FM}}^{\text{geo}}
$$

**含义**：单超参 $\alpha$ 折中外观保真度与几何一致性。

**符号说明**:
- $\alpha$：损失权重（实践中调到几何项与视频项同量级）

### 公式7: [[梯度分解|几何监督的梯度路径]]

$$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{\text{FM}}^{\text{vid}} + \alpha \cdot \frac{\partial \mathcal{L}_{\text{FM}}^{\text{geo}}}{\partial \mathbf{m}_t} \cdot \frac{\partial \mathbf{m}_t}{\partial \theta}
$$

**含义**：因为视频与几何两个损失**共享中间表示 $\mathbf{m}_t$**，几何项对视频骨干参数 $\theta$ 的影响完全通过 $\frac{\partial \mathbf{m}_t}{\partial \theta}$ 这条路径传递——视频输出空间**不变**，但 $\mathbf{m}_t$ 被强制编码几何因素。

**符号说明**:
- $\theta$：视频 DiT 参数
- 第二项即"对应一致性正则"在表示层的具体形式

### 公式8: [[Dual-Criterion Confidence-Gated Tracker|双判据跟踪信号]]

$$
s_t = \frac{|\mathcal{V}_t|}{|\mathcal{V}_{t_0}|} \in [0, 1], \qquad \Delta s_t = s_t - s_{t-1}
$$

**含义**：两个统计量同时刻画 tracker 健康状态——$s_t$ 是 anchor 保留率（总体趋势），$\Delta s_t$ 是帧间变化（突发信号），二者分离"渐进漂移"与"瞬时崩溃"两种失败模式。

**符号说明**:
- $\mathcal{V}_{t_0} \subseteq \mathcal{M}_{\text{ee}}$：首帧采样的 anchor keypoint 集合
- $\mathcal{V}_t \subseteq \mathcal{V}_{t_0}$：第 $t$ 帧仍可靠跟踪的子集
- $s_t \to 0$ 意味着 tracker 完全失败

### 公式9: [[Confidence-Gated Tracker|双分支修复策略]]

$$
\hat{\mathcal{M}}_{\text{ee}}^{\,t} = \begin{cases}
\text{re-anchor tracker at } t, & \text{if } s_t < \tau \\
\mathrm{Qwen3.5\text{-}VL}(I_t, c), & \text{if } \Delta s_t < -\delta \\
\mathcal{M}_{\text{ee}}^{\,t}, & \text{otherwise}
\end{cases}
$$

**含义**：渐进漂移用"重采样新 anchor"补救；突发崩溃则直接用 [[Qwen3.5-VL]] 重新做语义级 EE 定位；其余情况维持当前 mask。

**符号说明**:
- $\tau \in (0,1)$：保留率阈值
- $\delta > 0$：帧间下降阈值
- $I_t$：第 $t$ 帧 RGB；$c$：指令文本

### 公式10: [[FoundationPose|FoundationPose 位姿估计]]

$$
(\mathbf{R}_{\text{ee}}^t, \mathbf{T}_{\text{ee}}^t, \kappa_t) = \mathrm{FoundationPose}\bigl(\mathbf{I}_t, \mathbf{D}_t, \mathcal{M}_{\text{ee}}^{\,t}, \mathrm{CAD}\bigr)
$$

**含义**：给每帧的 RGB + 深度 + EE 掩膜 + CAD 模型，输出 $SE(3)$ 位姿与置信度。

**符号说明**:
- $\mathbf{D}_t$：每帧深度（真实场景由 [[Depth Anything V3]] 估计）
- $\kappa_t \in [0,1]$：FP 自带置信度

### 公式11: 时间一致性拒收准则

$$
\|\mathbf{T}_{\text{ee}}^t - \mathbf{T}_{\text{ee}}^{t-1}\|_2 > \epsilon_t, \quad d_{\text{geo}}(\mathbf{R}_{\text{ee}}^t, \mathbf{R}_{\text{ee}}^{t-1}) > \epsilon_R
$$

**含义**：当 $\kappa_t < \kappa^*$ 时启动二次检查——若平移跳跃超过 $\epsilon_t$ 或旋转测地距超过 $\epsilon_R$，拒收 FP 估计，转用质心反投影 + Slerp 兜底。

**符号说明**:
- $d_{\text{geo}}$：$SO(3)$ 上的测地距（旋转向量夹角）
- $\epsilon_t, \epsilon_R$：平移/旋转跳跃容忍阈值
- $\kappa^*$：FoundationPose 置信度接受阈值

### 公式12: [[Grasp Selection|抓取候选评分]]

$$
\mathbf{T}_{\text{grasp}}^* = \arg\min_{\mathbf{T}_{\text{grasp}}^{(i)}} \Bigl(\lambda_t\,\|\mathbf{t}_{\text{grasp}}^{(i)} - \mathbf{t}_{\text{ref}}\|_2 + \lambda_R\,d_{\text{geo}}(\mathbf{R}_{\text{grasp}}^{(i)}, \mathbf{R}_{\text{ref}})\Bigr)
$$

**含义**：从 [[GraspGen]] 输出的 $M$ 个抓取候选里，按与参考位姿（最接近目标物体的 EE 位姿）的加权偏差排序选最优——平移误差与旋转测地距按 $\lambda_t : \lambda_R$ 折中。

**符号说明**:
- $(\mathbf{R}_{\text{ref}}, \mathbf{T}_{\text{ref}})$：从恢复轨迹中挑出的"接近物体"的参考位姿
- $\{(\mathbf{R}_{\text{grasp}}^{(i)}, \mathbf{T}_{\text{grasp}}^{(i)})\}_{i=0}^M$：[[GraspGen]] 在物体点云上枚举的候选
- $\lambda_t, \lambda_R$：平移/旋转权重

---

## 关键图表

### Figure 1: Teaser / 任务示例

![Figure 1](https://arxiv.org/html/2605.22882v1/x1.png)

**说明**：给定指令与初始观测，GEM-4D 预测保持几何一致性的未来帧。与 baseline（左）相比，GEM-4D（右）生成的场景演化在结构上更真实、对象边界更紧致。

### Figure 2: GEM-4D 架构

![Figure 2](https://arxiv.org/html/2605.22882v1/x2.png)

**说明**：训练阶段，[[扩散变换器|Video DiT]] 预测噪声视频潜变量的速度，其中间特征 $\mathbf{m}_t$ 作为唯一场景条件喂给 Geometry DiT 预测几何速度。这种**双流耦合 + 非对称条件**强迫视频骨干编码几何结构。推理阶段**只用视频分支**，与原始视频 DiT 同样的计算成本。

### Figure 3: Adaptive Inverse Dynamic System

![Figure 3](https://arxiv.org/html/2605.22882v1/x3.png)

**说明**：给定生成视频，[[AIDS]] 通过四步抽出机器人策略——(1) 用 [[Qwen3.5-VL]]+[[SAM-2]]+[[FoundationPose]] 做 3D scene grounding；(2) 用 [[CoTracker3]] + 双判据置信门跟踪 EE keypoints；(3) FoundationPose 失败时用质心反投影 + [[Slerp]] 兜底位姿；(4) 用 [[GraspGen]] 选最优抓取插入轨迹，逆运动学得动作。

### Figure 4: Generated Frames to Arm Action / 真机执行示例

![Figure 4](https://arxiv.org/html/2605.22882v1/x4.png)

**说明**：从初始观测出发，经 GEM-4D 预测的未来帧，最终落到 UF 机械臂的实际动作执行。展示从生成视频到真机操作的完整流水线。

### Figure 5: 4D 场景生成定性对比

![Figure 5](https://arxiv.org/html/2605.22882v1/x5.png)

**说明**：在 Droid（真实）和 RLBench（合成）上的 4D 场景生成对比。**上**：ground truth；**中**：[[TesserAct]] 的 RGB 看着合理但深度不一致——机械臂区域明显扭曲、Droid 上深度带有噪声条纹；**下**：GEM-4D 保留跨帧几何结构，深度更干净、物体边界更紧致。

### Figure 6: 真机滚动展示

![Figure 6](https://arxiv.org/html/2605.22882v1/x6.png)

**说明**：从左到右：ground truth 视频、GEM-4D 生成 RGB、反投影 3D 点云。在未见背景与真实条件下，模型仍能生成真实且几何相干的滚动结果，支持迁移到 UF Arm 操作。

### Table 1: 4D 场景生成定量对比

测试 RGB 重建（FVD/SSIM/PSNR）+ 深度重建（AbsRel/$\delta_1$/$\delta_2$）+ 点对应（Chamfer/$\delta^{vis}_{avg}$）。D=数据集，R=Real (Droid)，S=Synthetic (RLBench)。

| D | Method | FVD↓ | SSIM↑ | PSNR↑ | AbsRel↓ | $\delta_1$↑ | $\delta_2$↑ | Chamfer↓ | $\delta^{vis}_{avg}$↑ |
|---|--------|------|-------|-------|---------|-------------|-------------|----------|----------------------|
| R | [[CogVideoX]] | 35.56 | 75.91 | 20.18 | 22.33 | 68.32 | 83.17 | 0.2670 | 66.22 |
| R | [[Wan2.2]] 2.2-14B | 33.43 | 76.24 | 20.70 | 21.39 | 71.18 | 84.35 | 0.2349 | 68.18 |
| R | [[TesserAct]] | 33.28 | 75.66 | 20.08 | 22.07 | 66.80 | 82.60 | 0.2630 | 67.14 |
| R | Geometry-Forcing | 33.17 | 76.12 | 20.53 | 21.96 | 69.74 | 83.83 | 0.2443 | 67.97 |
| R | **GEM-4D** | **31.82** | **82.05** | **21.11** | **20.13** | **78.19** | **88.21** | **0.2001** | **71.23** |
| S | [[CogVideoX]] | 40.21 | 75.51 | 20.03 | 15.41 | 70.99 | 92.90 | 0.2913 | 58.32 |
| S | [[Wan2.2]] 2.2-14B | 49.20 | 73.01 | 19.87 | 17.81 | 67.07 | 90.16 | 0.1762 | 61.99 |
| S | [[TesserAct]] | 41.97 | 76.72 | 19.71 | 16.02 | 69.26 | 93.03 | 0.1813 | 61.15 |
| S | Geometry-Forcing | 34.06 | 77.92 | 19.48 | 15.34 | 68.96 | 92.80 | 0.1488 | 60.84 |
| S | **GEM-4D** | **27.94** | **80.27** | **23.36** | **14.11** | **74.13** | **95.01** | **0.0702** | **68.18** |

**说明**：GEM-4D 在两类数据集、三组指标（外观/几何/对应）上**全面占优**，尤其 RLBench 上 Chamfer 距离从次优 0.1488 直接降到 0.0702（-53%），证明几何蒸馏对几何指标的提升远大于外观指标。

### Table 2: 任务成功率对比（真机 + 仿真）

左半：Droid 真机任务（三个真实环境 AUTOLab/CLVR/RAIL），人评打分；右半：RLBench 7 个任务，重放生成轨迹的成功率。

| Method | Droid AUTOLab | Droid CLVR | Droid RAIL | RLBench Lift Numbered Block | Put Rubbish In Bin | Reach Target | Lamp On | Pick Up Cup | Slide Block To Target | Solve Puzzle |
|--------|--------------|------------|-----------|----------------------------|--------------------|--------------| --------|-------------|----------------------|--------------|
| [[CogVideoX]] | 49 | 64 | 39 | - | - | - | - | - | - | - |
| [[TesserAct]] | 58 | 65 | 59 | 21 | 0 | 2 | 36 | 49 | 18 | 33 |
| **GEM-4D (Ours)** | **75** | **83** | **87** | **78** | **75** | **82** | **67** | **81** | **80** | **63** |

**说明**：真机三环境平均从 60.7% → 81.7%（+21 pp），论文摘要里的 "61%→81%" 即三环境平均。RLBench 上 TesserAct 在 Put Rubbish In Bin / Reach Target 上几乎是 0%（因 inverse dynamics 不开源，作者用自己的 AIDS 接它），GEM-4D 七任务全部 >60%。

### Table 3: 消融研究

替换或去除几何分支，看外观与几何指标的变化。

| Domain | Method | FVD↓ | SSIM↑ | PSNR↑ | AbsRel↓ | $\delta_1$↑ | $\delta_2$↑ | Chamfer↓ |
|--------|--------|------|-------|-------|---------|-------------|-------------|----------|
| Real | [[CogVideoX]] (微调 baseline) | 35.56 | 75.91 | 20.18 | 22.33 | 68.32 | 83.17 | 0.2670 |
| Real | [[Wan2.2]] 2.2-14B | 33.43 | 76.24 | 20.70 | 21.39 | 71.18 | 84.35 | 0.2349 |
| Real | GEM-4D (Dep) | 32.91 | 78.58 | 20.75 | 20.89 | 74.60 | 86.67 | 0.2229 |
| Real | GEM-4D (VGGT) | 33.68 | 75.89 | 20.64 | 21.73 | 71.03 | 83.80 | 0.2370 |
| Real | **GEM-4D (PAGE-4D)** | **31.82** | **82.05** | **21.11** | **20.13** | **78.19** | **88.21** | **0.2001** |

**关键发现**:
1. **纯微调不够**：CogVideoX 微调 baseline 的几何指标完全跟不上——必须有几何蒸馏。
2. **直接预测深度（Dep）次优**：把几何分支替换为预测深度速度场也能提升（比 baseline 好），但比预测密集几何特征差——说明特征级监督比深度像素级监督信息量更大。
3. **VGGT 反而轻微降效**：[[VGGT]] 主要在静态/准静态场景上训练，与机器人动态操作场景不匹配；[[PAGE-4D]] 这种能解耦静/动的几何模型才合适。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[ManiSkill3]] | 仿真 | 多任务机器人操作 | 训练 |
| [[RLBench]] | 仿真，7 评测任务 | 真值深度可得 | 训练 + 评测（780 unseen samples） |
| [[Bridge]] | 真实操作数据 | 大规模真实演示 | 训练 |
| [[RT-1]] | 真实操作数据 | Google 机器人数据 | 训练 |
| [[Droid]] | 真实，3 实验室 | AUTOLab/CLVR/RAIL | 评测（400 unseen samples，深度用 Depth Anything V3 估，对应用 CoTracker3 估） |

### 基线方法

1. **[[CogVideoX]]**：DiT 架构的图生视频模型（35M 视频预训练），作者在自己训练集上微调。
2. **[[Wan2.2]] 2.2-14B**：基于时空 VAE 的图生视频模型，同样微调。
3. **[[TesserAct]]**：基于 CogVideoX 的 4D 世界模型，联合预测深度 + 法向量（最强 4D 基线）。
4. **Geometry Forcing**：用表示对齐方式注入几何先验。
5. **未直接对比**：RoboTransfer（多视图几何约束，被 GEM-4D 作者认为定位不同）。

### 评测协议

- **4D Scene Prediction (4.1)**：外观 (FVD/SSIM/PSNR) + 几何 (AbsRel/$\delta_1$/$\delta_2$ 用 DA3 估深度) + 对应 (Chamfer / $\delta^{vis}_{avg}$ 用 CoTracker3)。
- **Embodied Action Planning (4.2)**：
  - 真机 Droid：15 名参与者人评（任务成功 + 物体/机械臂形变 + 指令跟随）
  - 仿真 RLBench：把生成视频走 AIDS 转动作，重放到模拟器看成功率
- **Qualitative**：UF Arm 真机滚动（未见背景）

### 实现细节

论文未公开具体超参，但可推断：
- **Backbone**：CogVideoX 类 [[扩散变换器]]（与 baseline 同骨干，公平对比）
- **几何监督源**：默认 [[PAGE-4D]]，消融实验也试过 [[Depth Anything V3]]、[[VGGT]]
- **训练数据**：ManiSkill3 + RLBench + Bridge + RT-1 混合
- **辅助工具链**：[[Qwen3.5-VL]] (VLM)、[[SAM-2]] (分割)、[[FoundationPose]] (6-DoF 位姿)、[[CoTracker3]] (点跟踪)、[[GraspGen]] (抓取)、[[Depth Anything V3]] (真实场景深度)
- **推理成本**：与 baseline 视频 DiT **完全相同**（几何 DiT 被丢弃）

---

## 批判性思考

### 优点

1. **理论清晰**：公式 7 的梯度分解把"几何监督等价于表示层正则"写得很干净，比 [[TesserAct]] 那种"再加个输出头"思路在工程上更优雅。
2. **零推理成本**：训练用辅助分支、推理只留主分支——这套范式应该可以推广到很多视频/世界模型的特征蒸馏场景。
3. **真机收益大**：Droid 三环境平均 +21 pp，且 RAIL 环境从 59→87（+28 pp）证明在难场景下提升更明显，说明几何一致性确实是机器人 planning 的关键瓶颈。
4. **AIDS 工程价值**：[[Dual-Criterion Confidence-Gated Tracker|双判据门控]] + [[Geometry-Kinematics Pose Fallback|位姿兜底]] 是把"生成视频 → 真机动作"打通的实际配方，组件均用现成模型（CoTracker3 / FoundationPose / GraspGen / SAM-2 / Qwen3.5-VL），可复用性强。
5. **数据多样性**：训练集横跨 ManiSkill3 + RLBench + Bridge + RT-1，覆盖仿真+真实+多机器人形态。

### 局限性

1. **未提及训练成本**：与冻结的大型几何 FM 联合训练应该相当昂贵，但论文未公布 GPU 时数与模型大小。
2. **AIDS 流水线脆弱**：依赖 SAM-2、FoundationPose、CoTracker3、GraspGen、Qwen3.5-VL 五个外部模型，任一环节失败都可能放大误差；论文虽设计了 fallback，但根本无法解决 EE CAD 模型不可获取的场景（强假设）。
3. **VGGT 退化只给出一句解释**：作者只说"VGGT 偏向静态场景"，未在动态视频上做更细的消融（如不同帧数、不同动作幅度），证据偏弱。
4. **真机评测靠人评**：Droid 人评虽然有 15 名参与者，但与"实际机械臂执行成功"仍有差距；只有 UF Arm 上做了真机闭环，规模小。
5. **几何监督仅用 PAGE-4D 充分发挥**：换 [[VGGT]] / Depth 都不如 PAGE-4D，说明这套方法的天花板**强依赖几何 FM 的质量**——一旦没有匹配的动态几何 FM，方法收益就有限。
6. **未与显式 3D Flow 方法对比**：3DFlowAction 这条线没出现在 baseline 表，论文以"输出空间不同"为由跳过，缺乏直接对比。

### 潜在改进方向

1. **几何分支自蒸馏**：用 GEM-4D 自身做几个 step 后的中间特征当 teacher，可能进一步降低对外部 FM 的依赖。
2. **AIDS 端到端化**：现在 AIDS 是模块化串联，每步都可能失败；可考虑训一个统一的视觉→动作 transformer 把 4 步合并。
3. **多视角扩展**：当前公式 1 假设单相机，多相机/腕部相机场景下对应公式更复杂，可与 [[WristWorld]] 思路结合。
4. **在线适应几何 FM**：训练时也微调一下几何 FM 的低层 LoRA 适配机器人场景，可能比纯冻结更好。

### 可复现性评估

- [ ] 代码开源（项目页是匿名提交版，未给代码 URL）
- [ ] 预训练模型（未提）
- [ ] 训练细节（无 GPU/时长/学习率/batch size）
- [x] 数据集可获取（ManiSkill3/RLBench/Bridge/RT-1/Droid 均公开）
- [x] 推理工具链全部开源（SAM-2 / FoundationPose / CoTracker3 / GraspGen / Depth Anything V3 / Qwen3.5-VL）

---

## 关联笔记

### 基于

- [[Flow Matching]]：基础训练范式，论文用线性插值轨迹
- [[扩散变换器|DiT]]：视频/几何骨干架构
- [[CogVideoX]] / [[Wan2.2]]：候选的视频 DiT 主干
- [[PAGE-4D]] / [[VGGT]] / [[Depth Anything V3]] / [[DUSt3R]] 系列：几何监督源
- [[FoundationPose]] / [[CoTracker3]] / [[GraspGen]] / [[SAM-2]] / [[Qwen3.5-VL]]：AIDS 工具链

### 对比

- [[TesserAct]]：最强 4D 基线，在输出空间预测深度+法向量；GEM-4D 改在**表示层**做蒸馏，几何指标全面超越
- [[UniPi]]：早期把视频生成当 planner 的代表，对应一致性问题最早被它揭示
- [[3DFlowAction]]：用 3D flow 当动作表示，与 GEM-4D 思路正交但未直接对比
- [[CogVideoX]] / [[Wan2.2]]：纯视频生成基线，无几何约束

### 方法相关

- [[对应一致性]]：本文的核心目标
- [[几何蒸馏]]：本文提出的训练范式
- [[Asymmetric Conditioning|非对称条件]]：几何 DiT 只读不写的关键设计
- [[Adaptive Inverse Dynamic System|AIDS]]：视频→动作的工程方案
- [[Dual-Criterion Confidence-Gated Tracker|双判据置信门跟踪]]：tracker 健康监控
- [[Geometry-Kinematics Pose Fallback|位姿几何-运动学兜底]]：FP 失败修复

### 硬件/数据相关

- [[Droid]]：真机评测数据
- [[RLBench]]：仿真评测平台
- [[ManiSkill3]] / [[Bridge]] / [[RT-1]]：训练数据
- UF Arm：真机部署平台

---

## 速查卡片

> [!summary] GEM-4D
> - **核心**: 把冻结 4D 几何 FM 当"对应一致性教师"，通过双流 Flow Matching 蒸馏到视频 DiT 中间表示，推理零额外成本
> - **方法**: 非对称条件（几何 DiT 只读视频 $\mathbf{m}_t$）+ 联合损失 $\mathcal{L}_{\text{FM}}^{\text{vid}} + \alpha\mathcal{L}_{\text{FM}}^{\text{geo}}$ + AIDS 4 步管道
> - **结果**: Droid 真机 61%→81%（+20 pp），RLBench 7 任务 63–82%，4D 场景指标全面超越 TesserAct
> - **代码**: 暂未公开（项目页 anonymous-submission-20.github.io/gem.github.io/）

---

*笔记创建时间: 2026-05-25*
