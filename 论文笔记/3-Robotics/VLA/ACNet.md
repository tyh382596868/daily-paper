---
title: "Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models"
method_name: "ACNet"
authors: [Tiecheng Guo, Meng Guo]
year: 2026
venue: arXiv
tags: [vla, asynchronous-inference, action-chunking, delay-aware, parameter-efficient]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.25985v1
created: 2026-06-25
---

# 论文笔记：Action ControlNet

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | School of Advanced Manufacturing and Robotics, Peking University |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[RTC]]、Training-RTC、Naïve Async |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25985) |

---

## 一句话总结

> ACNet 是一个轻量级延迟感知适配器，将已执行的「延迟动作」编码为残差条件注入 VLA 的动作头，仅用 ~20% 的参数量在异步控制场景中达到与全量重训相当的平滑轨迹效果。

---

## 核心贡献

1. **问题建模**: 将推理延迟导致的轨迹抖动形式化为「块边界条件缺失」问题，提出以执行中的延迟动作后缀作为条件信号。
2. **ACNet 适配器**: 设计轻量级侧支路编码器 + 末层残差注入机制，无需修改预训练的感知-语言骨干，仅微调 ~20% 参数。
3. **跨平台验证**: 在 Kinetix、Meta-World MT50（50 任务）和真实 SO-ARM101 机器人三个平台上验证，成功率和轨迹平滑度均优于 Naïve Async 和 RTC，媲美全量 Training-RTC。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型在机器人操控中通常以「动作块」（[[Action Chunking]]）形式输出动作序列。推理本身需要时间（延迟 $d$ 步），若采用**同步**模式，机器人必须等待推理完成才能执行，造成控制空白；若采用**异步**模式，推理与执行并行，但下一个动作块是基于「已过时的观测」生成的，块与块之间的拼接处出现**不连续性、抖动和震荡修正**。

### 现有方法的局限

- **Naïve Async（朴素异步）**: 直接拼接动作块，边界处抖动严重。
- **RTC（运行时修复）**: 推理时做动作内插修复，但不改变模型，且推理延迟从 73ms 增加到 159ms，控制频率从 13.6Hz 降到 6.28Hz。
- **Training-RTC（延迟条件训练）**: 将延迟信号融入全量重训，效果最好，但需要重训练 100% 参数，成本高。

### 本文的动机

延迟问题的本质是**边界条件缺失**：模型不知道在它推理期间机器人实际执行了哪些动作，因而无法平滑过渡。如果只在动作头处引入这一「已执行动作后缀」作为残差修正，可以高效地修复边界不连续，同时冻结骨干网络，大幅降低适配成本。

---

## 方法详解

### 模型架构

**ACNet** 采用 **参数高效残差适配** 架构，将预训练 [[VLA]] 分解为感知-语言骨干（冻结）和动作头（局部解冻），再添加一个轻量级侧支路进行延迟感知残差注入：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$ + **延迟动作** $\mathbf{a}_t^{\mathrm{delay}}$（推理期间实际执行的动作后缀）
- **Backbone**: 预训练感知-语言骨干 $\mathcal{B}_\omega$（完全冻结）
- **Action Head**: [[DiT]]-style 动作专家（8 层，主体冻结，仅末层参与残差接收）
- **侧支路**: 轻量 Transformer 编码器 $\mathcal{E}_\varphi$ → 池化为条件向量 $\mathbf{c}_t$
- **注入机制**: 末层残差投影 $\mathcal{Z}_{\varphi,L}(\mathbf{c}_t)$ 直接加到动作头最后一层隐状态
- **输出**: 修正后的 [[Action Chunking|动作块]] $\hat{\mathbf{a}}_t \in \mathbb{R}^{H \times d_a}$
- **可训练参数**: ~20% 总参数量（侧支路编码器 + 末层投影头）

### 核心模块

#### 模块 1: 延迟动作编码器（Delay-Action Encoder）

**设计动机**: 在推理期间（$d$ 步），机器人已执行了前一块的部分动作，这些已执行动作是下一块边界条件的最精确信息源。通过将这段「延迟动作」编码为紧凑的条件向量，可告知动作头「从哪里接着来」。

**具体实现**:
- 将 $d$ 步延迟动作 $\mathbf{a}_t^{\mathrm{delay}} \in \mathbb{R}^{d \times d_a}$ 用可学习填充 token $\mathbf{p}_j$ 补齐到完整视野长度 $H$，得到 $\tilde{\mathbf{a}}_t^{\mathrm{delay}} \in \mathbb{R}^{H \times d_a}$
- 送入轻量 [[Transformer]] 编码器 $\mathcal{E}_\varphi$ 提取序列表示
- 通过全局池化压缩为条件向量 $\mathbf{c}_t \in \mathbb{R}^{d_c}$

#### 模块 2: 末层残差注入（Late Residual Injection）

**设计动机**: 对动作头各层的更新幅度分析（Figure 6）显示，Evo-1 的 8 层 [[DiT]]-style 动作专家中，**最后一层的激活幅度最大**，说明末层承担了最终动作细化任务。因此，在末层注入残差是最高效的修正位置。

**具体实现**:
- 对动作头第 $l$ 层隐状态 $\mathbf{h}_l$，注入残差 $\mathbf{h}_l' = \mathbf{h}_l + \mathcal{Z}_{\varphi,l}(\mathbf{c}_t)$
- 根据一阶 Taylor 近似，此注入等价于在输出空间做边界修正：$\hat{a}_t^{(d)} \approx g_L(\mathbf{h}_L) + J_L \mathbf{u}_L$
- 注入集合 $\mathcal{S} = \{L\}$（仅末块），最小化可训练参数量

---

## 关键公式

### 公式 1: [[VLA|VLA 基础输出]]

$$
\mathbf{a}_t = \mathcal{M}_\theta(o_t, l) = \{a_t^{(0)}, a_t^{(1)}, \cdots, a_t^{(H-1)}\}, \quad \mathbf{a}_t \in \mathbb{R}^{H \times d_a}
$$

**含义**: VLA 模型 $\mathcal{M}_\theta$ 在时刻 $t$ 基于观测 $o_t$ 和语言指令 $l$ 输出长度为 $H$ 的动作块。

**符号说明**:
- $o_t$: 当前时刻的视觉观测
- $l$: 语言任务指令
- $H$: 动作块视野长度（chunk horizon）
- $d_a$: 动作维度

### 公式 2: [[Action Chunking|视野分解]]

$$
H = d + e + r
$$

**含义**: 动作块视野 $H$ 分解为三段：推理延迟步数、重启间隔步数、可选未来后缀步数。

**符号说明**:
- $d$: 推理延迟（单位：执行步数）
- $e$: 重启间隔（relaunch interval，即执行几步后启动新推理）
- $r$: 可选未来后缀步数（$r \geq 0$）

### 公式 3: [[异步控制|延迟动作定义]]

$$
\mathbf{a}_t^{\mathrm{delay}} = \{a_{t-e}^{(e)}, \cdots, a_{t-e}^{(e+d-1)}\} \in \mathbb{R}^{d \times d_a}
$$

**含义**: 推理期间实际被执行的 $d$ 步动作切片，即从上一块第 $e$ 个动作起的 $d$ 步。这是边界修正的核心条件信号。

**符号说明**:
- $a_{t-e}^{(j)}$: 上一块（在 $t-e$ 时刻生成）的第 $j$ 个动作
- $d$: 推理延迟步数

### 公式 4: [[VLA|策略分解]]

$$
\mathcal{M}_\theta(o_t, l) = \mathcal{A}_\psi(\mathcal{B}_\omega(o_t, l))
$$

**含义**: 将 VLA 模型分解为感知-语言骨干 $\mathcal{B}_\omega$ 和动作头 $\mathcal{A}_\psi$，ACNet 只修改后者。

**符号说明**:
- $\mathcal{B}_\omega$: 感知-语言骨干（冻结），提取多模态特征
- $\mathcal{A}_\psi$: 动作头，将特征解码为动作序列

### 公式 5: [[参数高效微调|延迟感知策略目标]]

$$
\eta^* = \mathop{\mathrm{arg\,min}}_{\eta:|\eta| \ll |\theta|} \mathbb{E}\left[\mathcal{L}_{\mathrm{pred}}(\hat{\mathbf{a}}_t, \mathbf{a}_t^*) + \lambda \mathcal{L}_{\mathrm{bd}}(\hat{\mathbf{a}}_t, \tilde{\mathbf{a}}_t^{\mathrm{delay}})\right]
$$

**含义**: 在参数量远小于全量 $|\theta|$ 的约束下，联合优化预测损失和边界一致性损失，寻找最优适配器参数 $\eta$。

**符号说明**:
- $\eta$: ACNet 可训练参数（~20% 总量）
- $\mathbf{a}_t^*$: 目标动作序列（ground truth）
- $\lambda$: 边界一致性损失权重
- $\mathcal{L}_{\mathrm{pred}}$: 预测损失（flow matching 或扩散损失）
- $\mathcal{L}_{\mathrm{bd}}$: 边界一致性损失

### 公式 6: [[边界一致性约束|边界一致性损失]]

$$
\mathcal{L}_{\mathrm{bd}} = \|\hat{a}_t^{(d)} - a_{t-e}^{(e+d-1)}\|_2^2
$$

**含义**: 约束新动作块第 $d$ 个动作与延迟动作最后一步（$a_{t-e}^{(e+d-1)}$）在欧氏距离上一致，确保边界平滑过渡。

**符号说明**:
- $\hat{a}_t^{(d)}$: 新块第 $d$ 步（即边界处）的预测动作
- $a_{t-e}^{(e+d-1)}$: 延迟动作序列的最后一步（已执行）

### 公式 7: [[延迟动作编码|填充延迟动作]]

$$
\tilde{\mathbf{a}}_t^{\mathrm{delay}} = [a_{t-e}^{(e)}, \cdots, a_{t-e}^{(e+d-1)}, \mathbf{p}_d, \cdots, \mathbf{p}_{H-1}]
$$

**含义**: 用可学习填充 token $\mathbf{p}_j$ 将 $d$ 步延迟动作补齐到完整块长度 $H$，使编码器输入维度固定。

**符号说明**:
- $\mathbf{p}_j \in \mathbb{R}^{d_a}$: 第 $j$ 位置的可学习填充 token
- $d$: 实际延迟步数，$H$: 总视野长度

### 公式 8: [[延迟动作编码|条件向量生成]]

$$
\mathbf{c}_t = \mathrm{Pool}(\mathcal{E}_\varphi(\tilde{\mathbf{a}}_t^{\mathrm{delay}})) \in \mathbb{R}^{d_c}
$$

**含义**: 轻量 Transformer 编码器将填充后的延迟动作序列编码后全局池化，得到紧凑的条件向量 $\mathbf{c}_t$。

**符号说明**:
- $\mathcal{E}_\varphi$: 轻量 Transformer 编码器（ACNet 侧支路）
- $\mathrm{Pool}$: 全局平均池化
- $d_c$: 条件向量维度

### 公式 9: [[残差注入|末层残差更新]]

$$
\mathbf{h}_l' = \mathbf{h}_l + \mathcal{Z}_{\varphi,l}(\mathbf{c}_t)
$$

**含义**: 将条件向量 $\mathbf{c}_t$ 经线性投影后以残差形式加到动作头第 $l$ 层隐状态，修正动作预测边界。

**符号说明**:
- $\mathbf{h}_l$: 动作头第 $l$ 层隐状态
- $\mathcal{Z}_{\varphi,l}$: 线性投影层（ACNet 的末层注入参数）

### 公式 10: [[Flow Matching|Flow Matching 训练损失]]

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{d, \tau, \mathbf{x}_0, \mathbf{z}}\left[\left\|(\mathbf{x}_0 - \mathbf{z}) - v_{\psi, \eta^{\mathrm{ACNet}}}(\mathbf{x}_\tau, \tau, \mathbf{c}_{t,d})\right\|_2^2\right]
$$

**含义**: [[Flow Matching]] 目标下，ACNet 学习以延迟条件 $\mathbf{c}_{t,d}$ 为额外输入预测从噪声 $\mathbf{z}$ 到目标动作 $\mathbf{x}_0$ 的速度场。

**符号说明**:
- $\mathbf{x}_\tau = (1-\tau)\mathbf{z} + \tau\mathbf{x}_0$: 插值中间状态，$\tau \sim \mathrm{Beta}(2,2)$
- $\mathbf{z} \sim \mathcal{U}([-1,1]^{H \times d_a})$: 均匀分布噪声
- $\mathbf{x}_0 = \mathbf{a}_t^*$: ground-truth 动作
- $v_{\psi, \eta^{\mathrm{ACNet}}}$: 带 ACNet 参数的速度预测网络
- $\mathbf{c}_{t,d}$: 延迟条件向量（随机延迟 $d$ 采样）

### 公式 11: [[Jerk|轨迹抖动度量]]

$$
J_t = \frac{1}{3} \sum_{q \in \{x,y,z\}} |a_{t+2}^q - 2a_{t+1}^q + a_t^q|
$$

**含义**: 以步间加速度的二阶差分（即 jerk）衡量轨迹平滑程度，值越小表示动作越平滑。

**符号说明**:
- $a_t^q$: $t$ 时刻 $q$ 轴方向的动作值
- $J_t$: $t$ 时刻的抖动值（在 xyz 三轴上平均）

---

## 关键图表

### Figure 1: 总览与真实机器人对比

![Figure 1 Top: Async VLA + ACNet Overview](https://arxiv.org/html/2606.25985v1/head.png)

![Figure 1 Bottom: Real-world Rollout on SO-ARM101](https://arxiv.org/html/2606.25985v1/exp_clean_the_table.png)

**说明**: 上图展示异步 VLA 控制设置与 ACNet 的整体示意；下图为 SO-ARM101 平台上 clean-the-table 任务的真实异步展开对比。Naïve Async（上序列）在块过渡处接触点偏移导致任务失败；ACNet（下序列）维持更平滑的接触并成功完成任务。

### Figure 2: 同步 vs 异步执行对比

![Figure 2: Sync vs Async VLA Execution](https://arxiv.org/html/2606.25985v1/async_vla.png)

**说明**: 上方（同步）：推理完成后才执行，存在控制空白；下方（异步）：推理与执行并行，但下一块基于过时观测生成，块边界处出现动作不连续。这张图是整篇论文核心问题的可视化。

### Figure 3: ACNet 架构图

![Figure 3: ACNet Architecture](https://arxiv.org/html/2606.25985v1/model.png)

**说明**: 展示 ACNet 的整体架构。预训练感知-语言骨干（左侧，冻结）正常处理 $o_t$ 和 $l$；延迟动作 $\mathbf{a}_t^{\mathrm{delay}}$ 送入轻量侧支路编码器 $\mathcal{E}_\varphi$，条件向量 $\mathbf{c}_t$ 通过投影层以残差形式注入动作专家末层。主体参数冻结，仅侧支路和末层投影参与训练。

### Figure 4: 轨迹抖动（Jerk）对比

![Figure 4 Left: Jerk on nut-assembly-v3](https://arxiv.org/html/2606.25985v1/jerk_compare.png)

![Figure 4 Right: Jerk on plate-slide-back-v3](https://arxiv.org/html/2606.25985v1/jerk_compare2.png)

**说明**: 左图 nut-assembly-v3，右图 plate-slide-back-v3，均为 $H=50, d=10$ 设置。Naïve Async 在块过渡点（约第 25 步）出现明显 jerk 峰值；ACNet 在相同位置 jerk 轮廓更平坦，过渡更平滑。

### Figure 5: SO-ARM101 真实机器人展开对比

![Figure 5: SO-ARM101 Rollout Comparison](https://arxiv.org/html/2606.25985v1/exp_put_cube_to_the_box.png)

**说明**: put-blue-cube-into-box 任务的真实机器人操作序列对比。上方为 Naïve Async，下方为 ACNet。清晰显示 ACNet 在块过渡处的稳定性优势。

### Figure 6: 动作专家层敏感性分析

![Figure 6: Layer Sensitivity Analysis](https://arxiv.org/html/2606.25985v1/layer_sensitivity.png)

**说明**: Evo-1 动作专家 8 层中每层的平均更新幅度（$K=1000$ 步 flow-matching 积分）。末层（第 8 层）幅度最大，表明末层是动作预测最终细化的关键层，这是 ACNet 选择在末层注入残差的实验依据。

### Figure 7: 消融实验

![Figure 7: ACNet Ablation Studies](https://arxiv.org/html/2606.25985v1/ablation.png)

**说明**: (a) 注入位置消融：对比在不同层（1~8 层以及多层）注入残差的效果，末层（Layer 8）表现最佳；(b) 填充策略消融：对比可学习 padding token、零填充、随机噪声填充，可学习 padding 在所有延迟值下均优于其他方案。

### Table 1: Kinetix 基准测试（成功率）

| 方法 | d=0 | d=1 | d=2 | d=3 | d=4 | Avg (d>0) | 参数量 (%) |
|------|-----|-----|-----|-----|-----|-----------|------------|
| Naïve Async | 0.89 | 0.74 | 0.69 | 0.55 | 0.46 | 0.61 | 0 |
| RTC | 0.91 | 0.75 | 0.80 | 0.72 | 0.61 | 0.72 | 0 |
| Training-RTC | 0.89 | 0.88 | 0.83 | 0.79 | 0.70 | 0.80 | 100 |
| **ACNet (ours)** | **0.90** | **0.87** | **0.84** | **0.76** | **0.68** | **0.79** | **~20** |

**说明**: ACNet 以约 20% 的参数量达到 0.79 平均成功率，媲美 Training-RTC 的 0.80，远超 RTC（0.72）和 Naïve Async（0.61）。

### Table 2: Meta-World MT50 基准测试（成功率 + 延迟指标）

| 方法 | d=0 | d=5 | d=10 | d=15 | Avg | 延迟 (ms) | 控制频率 (Hz) |
|------|-----|-----|------|------|-----|-----------|---------------|
| Naïve Async | 0.80 | 0.71 | 0.70 | 0.70 | 0.70 | 73 | 13.6 |
| RTC | 0.79 | 0.72 | 0.72 | 0.71 | 0.71 | 159 | 6.28 |
| Training-RTC | 0.80 | 0.77 | 0.74 | 0.73 | 0.74 | 134 | 7.46 |
| **ACNet (ours)** | **0.81** | **0.76** | **0.74** | **0.73** | **0.74** | **91** | **11.0** |

**说明**: ACNet 成功率（0.74）匹配 Training-RTC，同时端到端推理延迟仅 91ms，控制频率 11.0Hz，相比 RTC（159ms，6.28Hz）有显著优势，近乎维持了 Naïve Async 的实时性。

### Table 3: SO-ARM101 真实机器人任务（10 次试验）

| 方法 | 放方块入盒 | 清洁桌面 | 总体 |
|------|-----------|---------|------|
| Naïve Async | 9/10 (90%) | 8/10 (80%) | 17/20 (85%) |
| **ACNet (ours)** | **10/10 (100%)** | **10/10 (100%)** | **20/20 (100%)** |

**说明**: 在接触丰富的真实场景下，ACNet 实现 100% 成功率，而 Naïve Async 在块过渡处接触点偏移导致 15% 失败率。

---

## 实验

### 数据集

| 平台 | 规模 | 特点 | 用途 |
|------|------|------|------|
| Kinetix | π₀ 策略，32 epoch | 仿真基准（含 RTC 设置复现） | 训练/测试 |
| Meta-World MT50 | 50 个操作任务，H=50 | 多任务仿真，RTX 4080 SUPER | 训练/测试 |
| SO-ARM101 | 50 次真实展开，10 epoch | 真实机器人，接触丰富任务 | 测试 |

### 实现细节

- **Backbone**: Evo-1（基于 π₀，[[DiT]]-style 8 层动作专家）
- **ACNet 预训练**: 1 epoch 标准预热 + 24 epoch 全量预训练
- **ACNet 适配**: 8 epoch 延迟条件微调（仅 ~20% 参数）
- **Meta-World 配置**: 预测视野 $H=50$，重启间隔 $e=25$，延迟 $d \in \{0, 5, 10, 15\}$
- **硬件**: RTX 4080 SUPER（Meta-World）、SO-ARM101 机器人平台
- **延迟采样**: 训练时随机采样延迟 $d$，使模型对不同延迟值具有鲁棒性

### 可视化结果

轨迹 jerk 分析（Figure 4）显示 ACNet 在块过渡点附近的 jerk 轮廓更平坦，两个仿真任务（nut-assembly-v3、plate-slide-back-v3）均验证了这一点。真实机器人展开（Figure 5）直观展示了清洁任务中的接触平滑性优势。

---

## 批判性思考

### 优点

1. **参数高效**: 仅训练 ~20% 参数即达到全量重训性能，适合资源受限的部署场景。
2. **实时性保持**: 端到端延迟 91ms，控制频率 11.0Hz，远优于 RTC（6.28Hz），接近 Naïve Async（13.6Hz）。
3. **方法通用**: 残差注入框架与具体动作头类型（扩散/Flow Matching）无关，理论上可推广至其他生成式动作模型。
4. **真实验证充分**: 不只在仿真上测试，SO-ARM101 100% 成功率提供了有力的真实机器人证据。

### 局限性

1. **延迟假设固定**: 目前假设延迟 $d$ 在训练时已知且相对稳定，对时变或不可预测延迟的泛化性未充分讨论。
2. **Evo-1 绑定**: 实验仅在 Evo-1（π₀ 衍生）骨干上验证，其他 VLA 架构（如 OpenVLA、RT-2）的适用性需要验证。
3. **仿真任务较简单**: Meta-World MT50 任务的物理复杂度远低于真实操作场景，真实多样化任务的泛化性待验证。
4. **边界损失权重 λ 敏感性**: 未见消融实验讨论 $\lambda$ 的选择对结果的影响。

### 潜在改进方向

1. 扩展到时变延迟（变长延迟 $d$）场景，使模型在推理时动态估计实际延迟并调整条件。
2. 将 ACNet 框架迁移到其他 VLA 骨干（OpenVLA、UniVLA 等），验证通用性。
3. 与 [[LoRA]] 等其他参数高效方法结合，进一步压缩适配成本。

### 可复现性评估

- [ ] 代码开源（论文未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（epoch、学习率、硬件均有说明）
- [x] 数据集可获取（Kinetix、Meta-World 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[π₀]]: ACNet 基于 Evo-1（π₀ 衍生），复用其感知-语言骨干和 DiT-style 动作专家
- [[Action Chunking]]: 论文的核心问题设置建立在动作块执行机制之上

### 对比

- [[RTC]]: 运行时修复基线，ACNet 在成功率和实时性上均优于 RTC
- [[Diffusion Policy]]: 生成式动作头的代表方法，ACNet 框架与之兼容

### 方法相关

- [[Flow Matching]]: ACNet 的训练损失采用 Flow Matching 目标
- [[DiT]]: 动作专家采用 DiT-style 架构，末层残差注入基于对各层激活幅度的分析
- [[参数高效微调]]: ACNet 属于参数高效适配方法，仅训练 ~20% 参数
- [[异步控制]]: 论文核心问题场景

### 硬件/数据相关

- [[Meta-World]]: 50 任务仿真 benchmark
- [[SO-ARM101]]: 真实机器人平台

---

## 速查卡片

> [!summary] Action ControlNet (ACNet)
> - **核心**: 将异步 VLA 的推理延迟问题建模为「边界条件缺失」，用已执行延迟动作作为残差条件修正动作头
> - **方法**: 轻量侧支路编码器 + 末层残差注入，~20% 可训练参数
> - **结果**: Kinetix avg 0.79 ≈ Training-RTC 0.80；MT50 avg 0.74，延迟 91ms，频率 11Hz；SO-ARM101 100%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-25*
