---
title: "MAD: Mapping-Aware World Models for Agile Quadrotor Flight"
method_name: "MAD"
authors: [Xinhong Zhang, Runqing Wang, Yunfan Ren, Ding Yu, Boyu Zhou, Jian Sun, Fang Deng, Jie Chen, Gang Wang]
year: 2026
venue: arXiv
tags: [world-model, quadrotor, agile-flight, occupancy-grid-map, vision-based-navigation, model-based-rl, sim-to-real]
zotero_collection: 9-无人机
image_source: online
arxiv_html: https://arxiv.org/html/2606.04534v1
created: 2026-06-07
---

# 论文笔记：MAD: Mapping-Aware World Models for Agile Quadrotor Flight

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 北京理工大学 / 同济大学 / 南方科技大学 / 哈尔滨工业大学 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[DreamerV3]]、[[PPO]]、[[SHAC]]、EGO-Planner、YOPO |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04534) / [PDF](https://arxiv.org/pdf/2606.04534) / [HTML](https://arxiv.org/html/2606.04534v1) |

---

## 一句话总结

> 把 [[DreamerV3]] 的"重建深度图"自监督换成"重建 [[占用栅格图|OGM]] + [[可见性栅格图|VGM]]"，迫使 [[循环状态空间模型|RSSM]] 隐状态显式编码局部几何与观测历史，让四旋翼在杂乱场景中以 5 m/s 真机飞行。

---

## 核心贡献

1. **几何感知的世界模型 MAD**: 用 [[占用栅格图|OGM]] / [[可见性栅格图|VGM]] + 本体感知状态替代深度图重建作为自监督目标，让 [[循环状态空间模型|RSSM]] 隐状态显式编码局部几何、可见性历史、本体运动，输出可解释的占据/可见性预测。
2. **GPU 并行 OGM/VGM 构造模块**: 在可微仿真器 [[DiffAero]] 内实现，单卡可达 $4.84 \times 10^8$ voxels/s 的监督吞吐量、$1.21 \times 10^5$ env-steps/s 的交互速率，使训练在数小时内完成。
3. **统一三种策略学习范式**: 同一份 MAD 编码器可以分别接入 [[DreamerV3]] 风格的 imagination rollout（MAD-Dreamer）、冻结作 feature extractor 喂 [[PPO]]（MAD-PPO）或 [[SHAC]]（MAD-SHAC），且能从导航迁移到[[Drone Racing|无人机竞速]]任务。
4. **Sim-to-Real 真机验证**: 在 RealSense D435i + ASUS NUC 的紧凑四旋翼上部署 ONNX 模型（~4 ms 推理），室内最高 3.10 m/s、室外密林最高 5.05 m/s、仿真最高 9.66 m/s。

---

## 问题背景

### 要解决的问题

在杂乱场景中实现敏捷视觉自主飞行：仅靠机载[[Depth Camera|深度相机]]和本体感知，让四旋翼记住哪些区域已被观测、推断附近占据空间、在视野受限和实时性紧约束下行动。

### 现有方法的局限

1. **经典模块化栈**（[[VIO]] + 局部建图 + 轨迹规划 + 跟踪控制如 EGO-Planner）: 接口之间引入延迟、误差累积、巨大的工程开销，对密林、室内走廊等场景泛化差。
2. **端到端 RL 策略**（如 YOPO、纯 [[PPO]]）: 把感知-控制联合优化，但单一策略缺乏对"已占据 / 已观测 / 未知"空间的内置理解，难诊断、跨任务迁移差。
3. **现有空中世界模型**（Dream to Fly、SkyDreamer）: 强调时序预测或任务回报，没有把对**安全飞行最关键的局部几何结构**作为表征显式约束。

### 本文的动机

[[Embodied AI|具身智能]]中真正决定安全的不是像素，而是"哪些 voxel 是占据的、哪些是已观测的"。如果把 [[循环状态空间模型|RSSM]] 的自监督目标从"重建深度图"换成"重建机器人为中心的 [[占用栅格图|OGM]] / [[可见性栅格图|VGM]]"，隐状态自然就被迫编码这些几何与可见性信息——既可解释，又直接对接避障。

---

## 方法详解

### 模型架构

MAD 整体由 **GPU 并行栅格构造 + 几何感知 RSSM + 多种策略头** 组成：

- **输入**:
  - 深度图 $w_t \in \mathbb{R}^{18 \times 32}$
  - 本体感知向量 $d_t \in \mathbb{R}^9$ = 截断目标方向速度 $v^{tar}$ ⊕ 当前速度 $v$ ⊕ 体坐标系 z 轴方向 $u_z$
- **编码器**:
  - Image encoder $q_\psi^{img}$：轻量 CNN
  - Sensory encoder $q_\psi^{snsr}$：MLP
- **RSSM Core**: 沿用 [[DreamerV3]] 设计的确定性 $h_t$ + 离散随机 $z_t$
- **解码器（关键改动！）**: 不解码深度图，改解码 **OGM、VGM、本体状态、reward、continue flag**
- **策略头**: 三种可选 — MAD-Dreamer（[[Latent Dynamics Rollout|想象 rollout]] 学策略）、MAD-PPO（冻结 MAD 喂 [[PPO]]）、MAD-SHAC（冻结 MAD 喂 [[SHAC]]）
- **部署**: 整套导出为单个 ONNX 模型，机载推理 ~4 ms

### 核心模块

#### 模块 1: 机器人为中心的 [[占用栅格图|OGM]] 与 [[可见性栅格图|VGM]]

**设计动机**: OGM 编码"哪里有障碍"，VGM 编码"哪里被观测过"，二者分离才能避免模型在未观测区域"幻觉"占据，且与避障语义直接对齐。

**具体实现**:
- 局部网格沿 x/y/z 方向跨度 $[-3, 5] \times [-4, 4] \times [-2, 2]$ m，voxel 边长 $l = 0.4$ m，共 **20 × 20 × 10 = 4000** 个 voxel。
- **OGM 生成**: 在所有 voxel 中心并行做 point-in-primitive 测试（cube / sphere），GPU 上极快。
- **VGM 生成**:
  1. 当前帧把深度像素按相机内外参反投影为射线，被射中的 voxel 标记可见；
  2. 把上一帧的可见 voxel 变换到当前 robocentric 坐标系；
  3. **可见性 re-marking**: 每个旧可见 voxel 中心展开为 8 个 anchor 点，若任一 anchor 落在当前某 voxel 内，该 voxel 即继承可见标记（见 Figure 3b）。

#### 模块 2: 几何感知 [[循环状态空间模型|RSSM]]

**设计动机**: 不再重建 ego-centric 图像（视角变化导致难拟合），而是重建一个**朝向无关**的局部几何场，迫使隐状态承载长程几何记忆。

**具体实现**:
- 时序模型 $h_t = f_\psi(h_{t-1}, z_{t-1}, a_{t-1})$ 维持长期记忆
- 后验 $z_t \sim q_\psi(z_t | h_t, x_t)$ 注入当前观测
- 先验 $\hat{z}_t \sim p_\psi(\hat{z}_t | h_t)$ 在没有观测时也能预测下一隐状态
- 所有 decoder 都是 MLP（OGM/VGM 输出 4000 维 Bernoulli logits）
- 关键 trick: 占据重建只在**可见 voxel 上**计算 BCE（公式 12 第一项的 $\wedge g_t^{vis}$），避免对未观测区域伪监督

#### 模块 3: 多种策略学习范式

**MAD-Dreamer**: 从 replay buffer 抽 4 步上下文初始化隐状态，autoregressive 想象 16 步，actor π_θ(h_t, z_t) 在想象 rollout 上做 REINFORCE 风格梯度更新，critic 预测 bootstrapped λ-returns 的离散分布，$\gamma = 0.997$。

**MAD-PPO / MAD-SHAC**: 把 MAD 当作冻结的 feature extractor，每步前向得到 $(h_t, z_t)$ 喂给 actor；critic 走 privileged state；reward decoder 关闭，目标速度直接喂策略以实现任务无关训练。

---

## 关键公式

### 公式 1: 四旋翼[[质点动力学|点质量动力学]]

$$
\dot{p} = v, \quad \dot{v} = a + g - d v, \quad \dot{a} = \lambda (u - a)
$$

**含义**: 用三阶平移动力学近似四旋翼短时避障行为，把姿态/电机响应吸收进 $\lambda$ 一阶滤波。

**符号说明**:
- $p, v$: 世界系位置 / 速度
- $a$: 推力诱导加速度
- $g$: 重力
- $d$: 线性阻力系数
- $u$: 控制器下发的加速度命令
- $\lambda$: 姿态跟踪带宽近似

### 公式 2: 目标方向截断速度

$$
v^{tar} = \frac{p^{tar} - p}{\max\left(\|p^{tar} - p\| / v_{max},\ 1\right)}
$$

**含义**: 把"指向目标"的速度向量截断到 $v_{max}$，作为策略的方向引导。

**符号说明**:
- $p^{tar}$: 目标位置
- $v_{max}$: 最大允许速度

### 公式 3: 总 reward

$$
r = r_{srvl} + \alpha_{coll} r_{coll} + \alpha_{vel} r_{vel} + \alpha_{pos} r_{pos} + \alpha_{z} r_{z} + \alpha_{app} r_{app} + \alpha_{avoid} r_{avoid} + \alpha_{jerk} r_{jerk}
$$

**含义**: 8 项加权组合的 [[Reward Shaping|reward shaping]]，权重为 $\alpha_{coll}=5,\ \alpha_{vel}=0.06,\ \alpha_{pos}=0.5,\ \alpha_z=0.2,\ \alpha_{app}=0.6,\ \alpha_{avoid}=0.3,\ \alpha_{jerk}=0.005$。

### 公式 4: reward 各分量

$$
\begin{aligned}
r_{srvl} &= 0.3 \\
r_{coll} &= -\mathbb{I}(\text{collision}) \\
r_{jerk} &= -\|u - a\|_2^2 \\
r_{vel} &= -\text{SmoothL1}(\|v^{tar} - v\|_2, 0) \\
r_{z} &= e^{-|p_z - p^{tar}_z|} \\
r_{pos} &= e^{-\|p - p^{tar}\|_2}
\end{aligned}
$$

**含义**: 存活奖励、碰撞惩罚、jerk 惩罚（控制平滑）、速度跟踪、高度对齐、位置接近。

### 公式 5: 每个障碍物的"接近"奖励

$$
r_{app, i} = \begin{cases} -\|v^{app}_i\|_2 \cdot e^{-\|p^{rel}_i\|_2}, & \text{if } (p^{rel}_i)^\top v^{app}_i > 0 \\ 0, & \text{otherwise} \end{cases}
$$

**含义**: 速度分解为接近 + 规避两分量，若接近分量指向障碍则按距离指数衰减惩罚。

### 公式 6: 每个障碍物的"规避"奖励

$$
r_{avoid, i} = \begin{cases} \|v^{avoid}_i\|_2 \cdot e^{-\|p^{rel}_i\|_2}, & \text{if } (p^{rel}_i)^\top v^{app}_i > 0 \\ 0, & \text{otherwise} \end{cases}
$$

**含义**: 同上但奖励"侧向规避"分量。

### 公式 7: 选取最危险障碍

$$
r_{app} = r_{app, k}, \quad r_{avoid} = r_{avoid, k}, \quad k = \arg\min_i r_{app, i}
$$

**含义**: 只对最危险的障碍打 app/avoid 分量，避免被远处障碍稀释。

### 公式 8: [[循环状态空间模型|RSSM]] 三件套

$$
\begin{aligned}
h_t &= f_\psi(h_{t-1}, z_{t-1}, a_{t-1}) &\quad &\text{(sequence)} \\
z_t &\sim q_\psi(z_t | h_t, x_t) &\quad &\text{(posterior)} \\
\hat{z}_t &\sim p_\psi(\hat{z}_t | h_t) &\quad &\text{(prior)}
\end{aligned}
$$

**含义**: 时序确定性状态 + 离散随机潜变量的标准 [[DreamerV3]] RSSM。

### 公式 9: 双模态编码器

$$
x_t^{img} = q_\psi^{img}(w_t), \quad x_t^{snsr} = q_\psi^{snsr}(d_t)
$$

**含义**: 深度图走 CNN、本体状态走 MLP，融合作为后验 $q_\psi$ 的输入。

### 公式 10: 五个并行解码器

$$
\begin{aligned}
\hat{g}_t^{occ} &\sim p_\psi^{occ}(h_t, z_t) &\quad &\text{(OGM)} \\
\hat{g}_t^{vis} &\sim p_\psi^{vis}(h_t, z_t) &\quad &\text{(VGM)} \\
\hat{d}_t &\sim p_\psi^{snsr}(\hat{d}_t | h_t, z_t) &\quad &\text{(proprio)} \\
\hat{r}_t &\sim p_\psi^{rew}(\hat{r}_t | h_t, z_t) &\quad &\text{(reward)} \\
\hat{c}_t &\sim p_\psi^{term}(\hat{c}_t | h_t, z_t) &\quad &\text{(continue)}
\end{aligned}
$$

**含义**: 最关键的创新在前两项—直接预测 4000 维 OGM 和 VGM。

### 公式 11: 世界模型总损失

$$
\mathcal{L}(\psi) \doteq \mathbb{E}_{q_\psi}\left[\sum_{t=1}^{T} \beta_{recon} \mathcal{L}_{recon}(\psi) + \beta_{dyn} \mathcal{L}_{dyn}(\psi) + \beta_{rep} \mathcal{L}_{rep}(\psi)\right]
$$

**含义**: reconstruction + dynamics-KL + representation-KL 三项加权，权重 $\beta_{recon}=1, \beta_{dyn}=0.5, \beta_{rep}=0.1$。

### 公式 12: 重建损失（核心创新）

$$
\mathcal{L}_{recon} \doteq \text{BCE}(\hat{g}_t^{occ} \wedge g_t^{vis},\ g_t^{occ} \wedge g_t^{vis}) + \text{BCE}(\hat{g}_t^{vis}, g_t^{vis}) - \ln p_\psi(d_t | z_t, h_t) - \ln p_\psi(r_t | z_t, h_t) - \ln p_\psi(c_t | z_t, h_t)
$$

**含义**: 占据预测的 [[Binary Cross-Entropy|BCE]] **只在可见 voxel 上**计算（前两项的 $\wedge g_t^{vis}$），避免对未观测区域伪监督；可见性预测、本体、reward、continue 走标准负对数似然。

**符号说明**:
- $\wedge$: voxel-wise 逻辑与，作为可见性掩码
- 各项默认为 voxel-wise 求和后取批次平均

### 公式 13: Dynamics KL

$$
\mathcal{L}_{dyn} = D_{KL}\left[\text{sg}(q_\psi(z_t | h_t, x_t))\ \|\ p_\psi(\hat{z}_t | h_t)\right]
$$

**含义**: 让先验**追上**后验，posterior 用 stop-gradient（$\text{sg}$）冻住。

### 公式 14: Representation KL

$$
\mathcal{L}_{rep} = D_{KL}\left[q_\psi(z_t | h_t, x_t)\ \|\ \text{sg}(p_\psi(\hat{z}_t | h_t))\right]
$$

**含义**: 反向：让后验**靠近**先验，避免后验编码过多观测细节，prior 用 $\text{sg}$。

### 公式 15: Actor / Critic

$$
a_t \sim \pi_\theta(h_t, z_t), \quad v_t = V_\theta(h_t, z_t) \approx \mathbb{E}_{\pi_\theta, p_\psi}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k}\right]
$$

**含义**: 策略和价值函数都直接条件于 RSSM 隐状态 $(h_t, z_t)$；actor 是参数化 Gaussian，critic 输出 bootstrapped $\lambda$-returns 上的 categorical 分布。

---

## 关键图表

### Figure 1: 系统概览与真实部署

![Figure 1](https://arxiv.org/html/2606.04534v1/x1.png)

**说明**: (a) MAD 预测的占据（紫）/ 可见性（绿）栅格叠加在飞行场景上，明显能看到 robocentric 网格随姿态对齐。(b) 真机在杂乱室内用 Intel RealSense D435i 自主飞行，连续预测 OGM/VGM 作为内部表征。

### Figure 2: MAD 整体训练 + 部署管线

![Figure 2](https://arxiv.org/html/2606.04534v1/x2.png)

**说明**: (a) [[DiffAero]] 内 GPU 并行生成 OGM/VGM 监督，深度图 + 本体状态喂 MAD 学 $(h_t, z_t)$。(b) 两种策略训练范式：左为 imagination-based 的 MAD-Dreamer，右为 MAD 冻结作 encoder 的 MAD-PPO / MAD-SHAC。(c) 部署时 MAD + actor 合并导出 ONNX，深度 + VIO 状态进，PX4 控制命令出。

### Figure 3: OGM/VGM 的几何定义与可见性 re-marking

![Figure 3](https://arxiv.org/html/2606.04534v1/x3.png)

**说明**: (a) 机器人为中心的 OGM（占据）与 VGM（已观测）voxel 三维栅格示意。(b) 时刻 $k+1$ 的某 voxel 若包含来自时刻 $k$ 可见 voxel 的 8 个 anchor 点之一，则继承可见性标记—解决相机运动后可见区域如何"搬运"的问题。

### Figure 4: MAD 训练管线细节

![Figure 4](https://arxiv.org/html/2606.04534v1/x4.png)

**说明**: 每一步 encoder 把 $w_t, d_t$ 编码为 $x_t$，RSSM 输出 $h_t, z_t$；五个 decoder 并行重建 OGM / VGM / 本体 / reward / continue。所有目标共享同一 latent，强制信息高度聚合。

### Figure 5: 训练曲线（视觉导航）

![Figure 5](https://arxiv.org/html/2606.04534v1/x5.png)

**说明**: 88 seeds 上 mean episode return 与 success rate。MAD-PPO / MAD-SHAC / MAD-Dreamer 收敛到更高 success rate（>90%）和回报，初期收敛略慢——作者归因为隐状态需"先学会空间记忆"。

### Figure 6: 重建质量（消融）

![Figure 6](https://arxiv.org/html/2606.04534v1/x6.png)

**说明**: 关闭 sensory encoder 后只看 depth 输入下的速度 / 姿态估计误差。深色柱：sensory decoder 反传梯度到全模型；浅色柱：仅训 decoder。换成 OGM/VGM 监督后速度误差从约 1.5 m/s 降到 0.7 m/s，姿态余弦相似度也显著提升——印证"重建几何能逼模型推断本体运动"。

### Figure 7: 任务迁移学习曲线（无人机竞速）

![Figure 7](https://arxiv.org/html/2606.04534v1/x7.png)

**说明**: 把导航上训好的 MAD 冻结迁移到竞速任务作 feature extractor。MAD-PPO 早期就超过从头训的 PPO-Vision，最终甚至追上具有 privileged gate 位置的 PPO-State——证明 MAD 学到的几何表征任务无关。

### Figure 8: 无人机竞速可视化

![Figure 8a](https://arxiv.org/html/2606.04534v1/imgs/racing_env_1.jpeg)
![Figure 8b](https://arxiv.org/html/2606.04534v1/x8.png)

**说明**: (a) [[DiffAero]] 内的视觉竞速场景。(b) MAD-PPO 顶视轨迹，穿过窄门时姿态平滑、机动激进。

### Figure 9: MAD 想象 rollout 可视化

![Figure 9](https://arxiv.org/html/2606.04534v1/x9.png)

**说明**: 上：深度图（prior 预测 / posterior 重建 / GT）；中：占据图；下：可见性图。即使 MAD 并不重建深度，prior rollout 出的深度也"够用"，关键是占据预测在已观测区准确。

### Figure 10: Gazebo 森林导航综合对比

![Figure 10a](https://arxiv.org/html/2606.04534v1/x10.png)
![Figure 10b](https://arxiv.org/html/2606.04534v1/x11.png)
![Figure 10c](https://arxiv.org/html/2606.04534v1/x12.png)

**说明**: (a) YOPO / EGO-Planner / [[PPO]] / MAD-Dreamer 的代表轨迹。(b) 峰值飞行速度 vs 期望最大速度（5–10 m/s）× 环境稀疏度（10–30 m²/障碍）。(c) success rate 热力图：MAD-Dreamer 在所有稀疏度上都 >90%，基线在 20–80% 之间。**密林（9 m²/障碍）**下：MAD 峰值 4.95 m/s、路径 58.47 m；EGO-Planner 4.01 m/s；YOPO 3.71 m/s。

### Figure 11: 室内走廊轨迹

![Figure 11](https://arxiv.org/html/2606.04534v1/x13.png)

**说明**: 5 m × 40 m 走廊，MAD-Dreamer 完赛 **8.36 s**（EGO-Planner 14.33 s / YOPO 16.02 s），峰值 6.37 m/s。YOPO 在户外训练的迁移性差，MAD 的栅格表征跨域稳定。

### Figure 12: 动态环境零样本规避

![Figure 12](https://arxiv.org/html/2606.04534v1/x14.png)

**说明**: 没在动态场景训过，部署到含行人的仿真里仍能反应性规避——支持"空间结构化表征即使无显式动态建模也具反应安全性"的论点。

### Figure 13: 真机室内飞行

![Figure 13a](https://arxiv.org/html/2606.04534v1/x15.png)

**说明**: (a) 杂乱场景峰值 **2.06 m/s**；(b) 粗糙障碍场景峰值 **3.10 m/s**。轨迹平滑、无碰撞。

### Figure 14: 真机户外密林

![Figure 14](https://arxiv.org/html/2606.04534v1/x16.png)

**说明**: 持续风扰下的密林（稀疏度 ~10 m²/树），MAD 全程峰值 **5.05 m/s**，验证 sim-to-real。

> **注**: 论文中无显式的数据表格，所有量化结果都以图表（learning curves / success rate 热力图 / 速度对比）形式呈现，已在上述 Figure 中覆盖。

---

## 实验

### 数据集 / 训练环境

| 环境 | 用途 | 备注 |
|------|------|------|
| [[DiffAero]] | MAD 训练 + 策略训练 | GPU 并行可微仿真，包含 GPU-parallel OGM/VGM 构造模块 |
| Gazebo + PX4 SITL | 闭环大规模仿真 | 森林环境 + 5×40 m 走廊 + 动态行人 |
| 真实四旋翼 | sim-to-real 验证 | 室内杂乱 + 户外密林 |

### 实现细节

- **OGM/VGM 分辨率**: 20 × 20 × 10 = 4000 voxel，边长 0.4 m，跨度 $[-3, 5] \times [-4, 4] \times [-2, 2]$ m
- **深度图分辨率**: 18 × 32
- **本体状态维度**: $d_{sens} = 9$
- **损失权重**: $\beta_{recon} = 1, \beta_{dyn} = 0.5, \beta_{rep} = 0.1$
- **MAD-Dreamer**: 上下文长度 4 步，想象 horizon 16 步，$\gamma = 0.997$，replay ratio 4
- **训练硬件**: Intel Ultra 9-285K CPU + NVIDIA RTX 5090 GPU，训练数小时
- **Seeds**: 主实验 88 seeds，迁移实验 4 seeds
- **GPU 栅格吞吐**: $4.84 \times 10^8$ voxels/s，$1.21 \times 10^5$ env-step/s
- **机载硬件**: 250 mm 轴距、1.10 kg、T-Motor F60 PRO IV 2550 KV、推重比 4.2、NxtPX4v2 飞控、ASUS NUC (i7-1260p) 机载计算
- **相机**: RealSense D435i，30 Hz，FOV 87°×58°，量程 ~5 m
- **部署**: ONNX 模型，机载推理延迟 ~4 ms

### 关键定量结果

| 场景 | 指标 | MAD | 基线最佳 |
|------|------|------|----------|
| 密林（9 m²/障碍）仿真 | 峰值速度 | 4.95 m/s | EGO-Planner 4.01 m/s |
| 走廊 5×40 m | 完赛时间 | 8.36 s | EGO-Planner 14.33 s |
| 走廊峰值速度 | — | 6.37 m/s | EGO-Planner 4.01 m/s |
| DiffAero 仿真上限 | — | **9.66 m/s** | — |
| 真机室内 | 峰值速度 | **3.10 m/s** | — |
| 真机户外密林 | 峰值速度 | **5.05 m/s** | — |

### 可视化结果

- Figure 9 显示 MAD 即便不被监督深度，prior rollout 的深度仍可解释，VGM/OGM 在可见区清晰对齐。
- Figure 12 显示对未训练的动态行人也能反应性规避——栅格表征自带"反应安全"。

---

## 批判性思考

### 优点

1. **方法层面优雅**: 把 Dreamer 系列从"重建像素"换成"重建几何 + 可见性"，对症下药（飞行安全直接关乎占据），且实现增量小、对接 [[DreamerV3]] 几乎无痛。
2. **可见性掩码细节**: 公式 12 的占据 BCE 用 $\wedge g_t^{vis}$ 屏蔽未观测区域，避免训出"幻觉占据"，对真机安全极重要。
3. **统一三种策略学习**: 同一 MAD 编码器可走 imagination / 冻结 + PPO / 冻结 + [[SHAC]]，提供工业级灵活性。
4. **完整 sim-to-real**: 真机 5 m/s 户外飞行 + ONNX 4 ms 延迟，闭环验证扎实。
5. **GPU 栅格构造**: $4.84 \times 10^8$ voxel/s 的并行实现是基础设施贡献，使大规模训练在小时级完成。

### 局限性

1. **OGM/VGM 内存开销大**: 4000 voxel × 长序列在显存上限制了 batch / horizon，论文也明确指出"limits training scale"。
2. **下游策略未显式利用解码出的地图**: 解码的 OGM/VGM 仅作为自监督目标"间接整形"隐状态，部署时策略仍是反应式的，没有"安全过滤器"显式利用预测占据做硬约束（未来工作方向）。
3. **质点动力学过简**: 公式 1 是 3-DoF 平移模型，姿态被吸收为 $\lambda$ 一阶滤波，对极限机动可能精度不够。
4. **依赖 robocentric local map**: 全局长程导航（如 km 级户外）需要拼接，纯局部表征会丢全局拓扑。
5. **栅格分辨率固定 0.4 m**: 对小障碍（细枝条、电线）容易"漏看"，文章未做小障碍消融。
6. **没有显式动态建模**: Figure 12 的动态行人规避是"反应性"，但若多个高速动态物则未必稳，不能与真显式预测竞速。

### 潜在改进方向

1. 把解码的 OGM/VGM 接入显式 [[MPC|MPC]] / [[CBF|安全过滤器]]，端到端 + 安全保证 hybrid。
2. 概率占据预测 + 不确定度门控，让策略对"模型自认为可见但实际不确定"区域更保守。
3. 多分辨率（octree-like）OGM 替代固定网格，节省内存同时保留细障碍。
4. 引入 object-level 结构（如点云分割），处理动态物体与可拆解场景。
5. 把 DiffAero 的可微梯度同时反传到 MAD（论文里 SHAC 已经做了一半），可能进一步提升 sample efficiency。

### 可复现性评估

- [ ] 代码开源（论文未给出明确仓库 URL）
- [ ] 预训练模型
- [x] 训练细节完整（loss 权重、网络架构、超参齐全）
- [x] 仿真器 DiffAero 已开源
- [x] 硬件配置详细到型号
- [ ] 数据集（仿真生成，可复现但需 DiffAero）

---

## 关联笔记

### 基于

- [[DreamerV3]]: RSSM 骨架与 imagination 学策略框架
- [[循环状态空间模型]]: 确定性 $h_t$ + 随机 $z_t$ 的标准 RSSM 公式

### 对比

- [[PPO]]: 端到端 RL 基线，纯视觉 PPO 性能远逊于 MAD-PPO
- [[SHAC]]: 可微仿真梯度的 RL 基线，加 MAD encoder 后显著提升
- 经典模块化栈（EGO-Planner / YOPO）: 显式建图 + 规划，MAD 用"隐式建图 + 端到端"打通

### 方法相关

- [[占用栅格图]]: 核心监督信号
- [[可见性栅格图]]: 核心监督信号
- [[Binary Cross-Entropy]]: 占据/可见性 voxel 级损失
- [[Latent Dynamics Rollout]]: imagination 学策略
- [[Reward Shaping]]: 8 项加权 reward
- [[基于模型的强化学习|MBRL]]: 方法范式归属
- [[部分可观测过程|POMDP]]: 问题形式化
- [[Smooth-L1 Loss]]: 速度跟踪损失

### 仿真与硬件相关

- [[DiffAero]]: GPU 可微飞行仿真器（基础设施）
- [[sim-to-real]]: 整篇论文的目标域
- [[VIO|视觉惯性里程计]]: 真机状态估计输入
- [[Depth Camera|RealSense D435i]]: 机载深度相机

---

## 速查卡片

> [!summary] MAD: Mapping-Aware World Models for Agile Quadrotor Flight
> - **核心**: Dreamer-style RSSM 的自监督目标从"重建深度图"换成"重建机器人为中心 OGM + VGM"
> - **方法**: GPU 并行栅格构造 + 5 路 decoder（OGM/VGM/proprio/reward/continue）+ MAD-Dreamer / MAD-PPO / MAD-SHAC 三种策略头
> - **结果**: 仿真 9.66 m/s、真机室内 3.10 m/s、户外密林 5.05 m/s；Gazebo 走廊 8.36 s（基线 14+ s）
> - **创新点**: 公式 12 用 $\wedge g_t^{vis}$ 把占据 BCE 限制在已观测 voxel 内，避免幻觉
> - **代码**: 未公开明确仓库；依赖开源 [[DiffAero]]

---

*笔记创建时间: 2026-06-07*
