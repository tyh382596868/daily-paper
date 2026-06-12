---
title: "LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories"
method_name: "LabVLA"
authors: [Baochang Ren, Xinjie Liu, Xi Chen, Yanshuo Liu, Chenxi Li, Daqi Gao, Zeqin Su, Jintao Xing, Zirui Xue, Rui Li, Xiangyu Zhao, Shuofei Qiao, Minting Pan, Wangmeng Zuo, Lei Bai, Dongzhan Zhou, Ningyu Zhang, Huajun Chen]
year: 2026
venue: arXiv
tags: [vla, laboratory-automation, synthetic-data, flow-matching, action-tokenization, cross-embodiment, sim-to-real]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.13578
created: 2026-06-12
---

# 论文笔记：LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University / Shanghai AI Laboratory / Harbin Institute of Technology |
| 日期 | June 2026 |
| 项目主页 | [zjunlp.github.io/LabVLA](https://zjunlp.github.io/LabVLA) |
| 对比基线 | [[pi0-FAST]] / [[SmolVLA]] / [[UniVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13578) / [Code](https://github.com/zjunlp/LabVLA) / [Model](https://huggingface.co/zjunlp/LabVLA) |

---

## 一句话总结

> LabVLA 通过 RoboGenesis 仿真引擎合成实验室场景数据，结合 [[FAST]] 动作 tokenizer 预训练 + [[Flow Matching]] 后训练的两阶段 recipe，在 LabUtopia benchmark 上以 4B 参数实现 71.1%/70.0% ID/OOD 成功率，超越现有 VLA 基线。

---

## 核心贡献

1. **RoboGenesis 仿真引擎**: 三阶段自动化合成管道（资产生成 → Agentic 流程生成 → 数据导出），支持 16 种机器人平台，针对实验室场景生成跨具身训练数据
2. **LabEmbodied-Data 数据集**: 含 2,947 个标注 3D 资产（LabAssetLibrary）和 10,000 个程序化生成实验室场景（LabSceneLibrary），15 类结构化标注
3. **两阶段训练 Recipe**: [[FAST Action Tokenizer]] 预训练对齐动作 token + [[Flow Matching]] 后训练精化，配合 Knowledge Insulation 防止 [[Cross-Embodiment Learning|跨具身]] VLM 能力退化

---

## 问题背景

### 要解决的问题

AI 系统可以推理科学实验，但物理执行仍在其能力范围之外。实验室机器人自动化需要操作精密仪器（移液管、离心机、热循环仪）、处理透明液体、遵守固定实验协议——这些在现有 VLA 训练数据中几乎不存在。

### 现有方法的局限

- 现有 [[VLA]] 策略几乎全部在家用和桌面场景演示上训练，缺乏对实验室仪器、透明液体、固定协议工作流的曝光
- 直接采集真实实验室数据成本极高（专用设备、安全程序、领域专业知识）
- 主流机器人数据集（如 Open X-Embodiment）中几乎没有移液管、离心机等实验室专用工具
- [[Cross-Embodiment Learning|跨具身]] 微调往往导致 VLM backbone 能力退化（知识遗忘）

### 本文的动机

通过仿真合成专域数据，规避真实数据采集的高成本；同时设计 Knowledge Insulation 机制，保留 VLM 在微调中的通用能力。将实验室自动化确立为独立的 VLA 研究领域。

---

## 方法详解

### 模型架构

[[LabVLA]] 采用 **Backbone + DiT 动作专家** 双模块架构：

- **输入**: RGB 图像 $I_t^{1:V}$（多相机视角）+ 语言指令 $\ell$ + 机器人状态 $s_t$
- **VLM Backbone**: [[Qwen3-VL]]-4B-Instruct（2560 维隐层）
- **动作专家**: [[DiT]]（18 层，宽度 1024，8 注意力头）
- **动作表示**: [[Action Chunking]] $A_t^r = [a_t^r, \ldots, a_{t+K-1}^r]$
- **推理步数**: N=10 Euler 步 [[Flow Matching]]
- **总参数**: ~4B

### 核心模块

#### 模块 1: RoboGenesis 数据合成引擎

**设计动机**: 真实实验室数据采集成本极高，通过 [[Physics Simulator|物理仿真]] 合成替代

**三阶段管道**:
1. **环境构建**: 文本提示 → 参考图像（text-to-image）→ TRELLIS 2.0 三维重建 → LabAssetLibrary（2,947 资产）；场景装配验证 → LabSceneLibrary（10,000 场景）
2. **Agentic 流程生成**: Agent 规划器将自然语言指令分解为有序原子技能工作流；6 轴域随机化（光照、纹理、物理参数、相机、机器人、物体布局）
3. **LabEmbodied-Data 导出**: 成功率过滤；15 类标注提供者（多视角 RGB、机器人状态、动作、结构化注释）

**优势**: 支持 16 种机器人平台（单臂、双臂、移动操作臂），自动生成资产/场景/任务，是现有引擎中功能最全的（见 Table 1）。

#### 模块 2: 两阶段训练 Recipe

**Stage 1 — [[FAST Action Tokenizer|FAST]] 预训练**

对齐动作 tokenization 表示，序列为：

$$
X_{\text{pre}} = [v_t;\; c_t;\; y_t;\; z_{1:L_z}]
$$

其中 $v_t$ 为视觉 token，$c_t$ 为相机参数 token，$y_t$ 为语言指令 token，$z_{1:L_z}$ 为 FAST 动作 token 序列。

**Stage 2 — [[Flow Matching]] 后训练 + Knowledge Insulation**

DiT 以 [[Flow Matching]] 目标微调，同时通过 stop-gradient 机制（Knowledge Insulation）保护 VLM backbone 参数不被动作 loss 破坏。

#### 模块 3: Knowledge Insulation (KI)

**设计动机**: 直接用 flow matching loss 端到端训练会导致 VLM backbone 遗忘预训练获得的语言/视觉能力

**具体实现**:
- KI 序列将机器人状态 $p_t$ 替换视觉 token 作为额外条件
- 对 VLM 隐层输出施加 stop-gradient $\mathrm{sg}(\cdot)$，切断动作 loss 对 backbone 的梯度反传
- 联合优化 [[FAST]] autoregressive loss + [[Flow Matching]] loss + 辅助 [[Cross-Entropy]] loss

---

## 关键公式

### 公式 1: [[Action Chunking|动作块]]预测目标

$$
A_t^r = [a_t^r,\; \ldots,\; a_{t+K-1}^r] \in \mathbb{R}^{K \times d_r}
$$

**含义**: 每步预测未来 $K$ 帧动作块，$d_r$ 为机器人具身的动作维度

**符号说明**:
- $A_t^r$: 时刻 $t$ 的动作块，$r$ 表示机器人具身类型
- $K$: chunk 长度（时间范围）
- $d_r$: 动作维度（依具身而异）

### 公式 2: [[Qwen3-VL|VLM 隐状态]]映射

$$
H_\phi = f_\phi(I_t^{1:V},\; \ell) \in \mathbb{R}^{L_h \times d_{\text{vlm}}}
$$

**含义**: VLM backbone $f_\phi$ 将多视角图像和语言指令编码为隐状态序列

**符号说明**:
- $f_\phi$: VLM backbone（Qwen3-VL-4B）
- $I_t^{1:V}$: $V$ 个相机视角的 RGB 图像
- $\ell$: 语言指令
- $L_h$: 隐状态序列长度，$d_{\text{vlm}} = 2560$

### 公式 3: [[FAST Action Tokenizer|FAST]] 预训练序列

$$
X_{\text{pre}} = [v_t;\; c_t;\; y_t;\; z_{1:L_z}]
$$

**含义**: 预训练阶段输入序列的构成，将视觉、相机、语言、动作 token 拼接

**符号说明**:
- $v_t$: 视觉 token
- $c_t$: 相机参数 token
- $y_t$: 语言指令 token
- $z_{1:L_z}$: FAST 动作 token 序列（长度 $L_z$）

### 公式 4: [[FAST Action Tokenizer|FAST]] 预训练损失

$$
\mathcal{L}_{\text{FAST}} = -\frac{1}{\sum_{i=1}^{L_z} m_i} \sum_{i=1}^{L_z} m_i \log p_\phi(z_i \mid v_t, c_t, y_t, z_{<i})
$$

**含义**: 对动作 token 的 autoregressive 交叉熵损失，$m_i$ 为掩码以仅计算动作 token 位置

**符号说明**:
- $m_i \in \{0,1\}$: 动作 token 掩码
- $p_\phi$: VLM 参数化的 next-token 预测分布
- $z_{<i}$: 前序动作 token（自回归条件）

### 公式 5: VLM 预训练综合损失

$$
\mathcal{L}_{\text{VLM}} = \mathcal{L}_{\text{FAST}} + \sum_j \lambda_j \mathcal{L}_{\text{CE}}^{(j)}
$$

**含义**: FAST 损失加上多个辅助 [[Cross-Entropy]] 任务的加权组合

**符号说明**:
- $\lambda_j$: 第 $j$ 个辅助任务的权重
- $\mathcal{L}_{\text{CE}}^{(j)}$: 第 $j$ 个辅助分类/预测任务的交叉熵损失

### 公式 6: [[Flow Matching]] 正向插值

$$
X_\tau = \tau \bar{A}_t^r + (1-\tau)\varepsilon, \quad U_\tau = \bar{A}_t^r - \varepsilon
$$

**含义**: 在噪声 $\varepsilon$ 和目标动作块 $\bar{A}_t^r$ 之间线性插值构造训练轨迹，$U_\tau$ 为对应速度场

**符号说明**:
- $\tau \sim \mathrm{Beta}(1.0, 1.5)$，缩放至 $0.999\tilde{\tau}$: 插值时间步（Beta 分布偏向低 $\tau$ 值，强化噪声端训练）
- $\bar{A}_t^r$: 目标动作块（归一化后）
- $\varepsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声

### 公式 7: [[DiT]] 速度场预测

$$
V_\theta = g_\theta(X_\tau,\; \tau,\; q_t^r,\; \Pi(H_\phi))
$$

**含义**: DiT 动作专家 $g_\theta$ 以噪声动作、时间步、机器人状态和 VLM 隐状态为条件预测速度场

**符号说明**:
- $g_\theta$: DiT（18 层，1024 宽度）
- $q_t^r$: 机器人专有状态向量
- $\Pi(\cdot)$: 将 VLM 隐状态投影到 DiT 维度的线性层

### 公式 8: [[Flow Matching]] 训练损失（带掩码）

$$
\mathcal{L}_{\text{FM}} = \begin{cases}
S_M^{-1} \sum_{k,d} M^{\text{act}}_{k,d}\,(V_{\theta,k,d} - U_{\tau,k,d})^2 & S_M > 0 \\
0 & S_M = 0
\end{cases}
$$

其中 $S_M = \sum_{k,d} M^{\text{act}}_{k,d}$

**含义**: 仅对有效动作维度（掩码 $M^{\text{act}}$ 为 1 的位置）计算速度场预测的 MSE，屏蔽无效维度

**符号说明**:
- $M^{\text{act}}_{k,d} \in \{0,1\}$: 第 $k$ 步第 $d$ 维的有效动作掩码
- $V_{\theta,k,d}$: DiT 预测的速度场分量
- $U_{\tau,k,d}$: 真实速度场分量

### 公式 9: Knowledge Insulation 序列

$$
X_{\text{KI}} = [s_t;\; p_t;\; y_t;\; z_{1:L_z}]
$$

**含义**: KI 阶段用紧凑机器人状态 $p_t$ 替换图像 token，减少梯度传播路径

**符号说明**:
- $s_t$: 系统提示 token
- $p_t$: 机器人专有状态（紧凑表示，替换视觉输入）

### 公式 10–11: Knowledge Insulation Stop-Gradient

$$
H^{\text{KI}}_{\phi,p} = \mathrm{slice}_p(f_\phi(X_{\text{KI}}))
$$

$$
\tilde{H}^{\text{KI}}_{\phi,p} = \mathrm{sg}(H^{\text{KI}}_{\phi,p}), \quad V^{\text{KI}}_\theta = g_\theta(X_\tau,\; \tau,\; q_t^r,\; \Pi(\tilde{H}^{\text{KI}}_{\phi,p}))
$$

**含义**: 对 VLM 隐状态施加 stop-gradient $\mathrm{sg}(\cdot)$，使 flow matching loss 不反传到 VLM backbone

**符号说明**:
- $\mathrm{slice}_p(\cdot)$: 从 VLM 输出中切片出与 $p_t$ 对应的隐状态
- $\mathrm{sg}(\cdot)$: stop-gradient 算子（前向传值，反向梯度为 0）

### 公式 12: [[Knowledge Insulation]] 综合损失

$$
\mathcal{L}_{\text{KI}} = \alpha \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{FAST}} + \sum_j \lambda_j \mathcal{L}_{\text{CE}}^{(j)}, \quad \alpha = 10
$$

**含义**: 后训练阶段的联合损失，flow matching 项权重 $\alpha=10$ 主导，FAST autoregressive 项维持语言能力

**符号说明**:
- $\alpha = 10$: flow matching 损失权重（相对于 FAST 损失放大 10 倍）
- $\mathcal{L}_{\text{FAST}}$: FAST autoregressive 损失（维持 VLM 能力）
- $\mathcal{L}_{\text{CE}}^{(j)}$: 辅助任务损失

### 公式 13: [[Flow Matching]] 推理积分

$$
X_{\tau + \Delta\tau} = X_\tau + \Delta\tau \cdot g_\theta(X_\tau,\; \tau,\; q_t^r,\; \Pi(H_\phi)), \quad \Delta\tau = \frac{1}{N}
$$

**含义**: 以 Euler 方法在 $N=10$ 步内从噪声积分到动作块，推理时无需重新计算 VLM 特征

**符号说明**:
- $N = 10$: Euler 积分步数
- $\Delta\tau = 1/N$: 每步步长

---

## 关键图表

### Figure 1: LabVLA 框架概览

![Figure 1](https://arxiv.org/html/2606.13578v1/x1.png)

**说明**: LabVLA 整体框架，展示策略架构（左：Qwen3-VL backbone + DiT 动作专家）、LabEmbodied-Data 数据组成（中：多场景、多具身）、RoboGenesis 三阶段管道（右），以及 LabUtopia 覆盖的四类任务家族。

### Figure 2: RoboGenesis 数据合成管道

![Figure 2](https://arxiv.org/html/2606.13578v1/x2.png)

**说明**: RoboGenesis 三阶段流程。(1) 环境构建（左）：文本提示经 text-to-image + TRELLIS 2.0 三维重建填充 LabAssetLibrary，场景装配流程生成 LabSceneLibrary（底部条带）。(2) Agentic 流程生成（中）：Agent 规划器将自然语言指令分解为原子技能序列，叠加 6 轴域随机化。(3) LabEmbodied-Data 导出（右）：成功率过滤 + 15 类结构化标注提供者输出最终数据集。

### Figure 3: LabVLA 训练 Recipe

![Figure 3](https://arxiv.org/html/2606.13578v1/x3.png)

**说明**: 两阶段训练方案对比。左侧预训练阶段以 $\mathcal{L}_{\text{FAST}}$ 对齐动作 tokenization；右侧后训练阶段在 VLM 隐状态到 DiT 之间加入 stop-gradient（Knowledge Insulation），Flow Matching loss 仅更新 DiT 而不反传到 VLM backbone。

### Figure 4: LabUtopia 六类任务 Rollout 可视化

![Figure 4](https://arxiv.org/html/2606.13578v1/x4.png)

**说明**: LabUtopia 六类评测任务（Pick Up、Press Button、Open Door、Pour Liquid、Heat Beaker、Transport Beaker）的第三人称视角 rollout 截图。展示了从初始状态到完成的连续动作序列，体现实验室专用操作的精细度要求。

### Figure 5: 真实机器人实验平台

![Figure 5](https://arxiv.org/html/2606.13578v1/x5.png)

**说明**: Franka 平台真实机器人实验设置，工作区包含烧杯、锥形瓶、磁力搅拌器和加热板，用于评测 Shake Liquid、Pour Liquid、Magnetic Stir、Stopper Plug/Unplug 四类任务。

### Figure 6: 实验室 VLA 能力四层金字塔

![Figure 6](https://arxiv.org/html/2606.13578v1/x6.png)

**说明**: 从 Apprentice（学徒）到 Scientist（科学家）的四层能力金字塔，定义实验室 VLA 的能力阶梯：底层处理简单操作，顶层具备完整实验设计与执行能力。

### Table 1: 仿真引擎功能对比

| Engine | Robots | Auto Asset | Auto Scene | Auto Task | Domain Random. | Success QA | Annotations | Long-horizon | Lab Protocol |
|--------|--------|-----------|-----------|----------|---------------|-----------|-------------|-------------|--------------|
| RoboTwin 2.0 | 5 | — | ✓ | ✓ | ✓ | ✓ | — | — | — |
| RoboCasa 365 | 1 | — | ✓ | — | ✓ | ✓ | — | — | — |
| ManiSkill 3 | 23 | — | — | — | ✓ | ✓ | — | — | — |
| RLBench | 5 | — | — | — | ✓ | ✓ | — | — | — |
| RoboGen | 6 | — | ✓ | ✓ | — | ✓ | — | — | — |
| **RoboGenesis** | **16** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**表格说明**: RoboGenesis 是唯一支持所有 8 项功能的引擎，尤其在自动资产生成、结构化标注、长时域任务、实验室协议方面具有显著优势。

### Table 2: LabUtopia 评测结果（成功率 %）

#### In-Distribution (ID)

| Method | Size | Pick Up | Press Button | Open Door | Pour Liquid | Heat Beaker | Transport Beaker | Avg. |
|--------|------|---------|-------------|-----------|------------|------------|------------------|------|
| SmolVLA | <1B | 15.8 | 97.5 | 16.7 | 0.8 | 96.7 | 85.8 | 52.2 |
| X-VLA | <1B | 27.5 | 98.3 | 65.0 | 45.0 | 25.8 | 83.3 | 57.5 |
| GR00T N1.5 | 3B | 40.8 | 99.2 | 6.7 | 0 | 99.2 | 69.2 | 52.5 |
| π₀ | 3B | 21.7 | 92.5 | 51.6 | 37.5 | 90.0 | 86.7 | 63.3 |
| π₀.₅ | 3B | 38.3 | 60.0 | 55.8 | 29.2 | 40.8 | 90.0 | 52.4 |
| π₀-FAST | 3B | 16.7 | 37.5 | 17.5 | 5.8 | 3.3 | 20.8 | 16.9 |
| InternVLA-A1 | 3B | 25.8 | 93.3 | 38.3 | 2.5 | 82.5 | 67.5 | 51.7 |
| Wall-oss-flow | 4B | 11.7 | 54.2 | 0.83 | 0 | 0 | 29.2 | 16.0 |
| **LabVLA** | **4B** | **49.2** | **100** | **65.0** | **43.3** | **83.3** | **85.8** | **71.1** |

#### Out-of-Distribution (OOD)

| Method | Size | Pick Up | Press Button | Open Door | Pour Liquid | Heat Beaker | Transport Beaker | Avg. |
|--------|------|---------|-------------|-----------|------------|------------|------------------|------|
| SmolVLA | <1B | 11.7 | 99.2 | 18.3 | 1.67 | 98.3 | 89.2 | 53.1 |
| X-VLA | <1B | 27.5 | 99.2 | 59.2 | 25.0 | 39.2 | 67.5 | 52.9 |
| GR00T N1.5 | 3B | 33.3 | 92.5 | 8.3 | 0 | 99.2 | 66.7 | 50.0 |
| π₀ | 3B | 19.2 | 89.1 | 53.3 | 38.3 | 90.8 | 88.3 | 63.2 |
| π₀.₅ | 3B | 30.0 | 68.3 | 59.2 | 29.2 | 40.0 | 85.8 | 52.1 |
| π₀-FAST | 3B | 14.2 | 45.0 | 15.8 | 7.5 | 11.7 | 24.2 | 19.7 |
| InternVLA-A1 | 3B | 19.2 | 95.8 | 63.3 | 0.83 | 84.2 | 57.5 | 53.5 |
| Wall-oss-flow | 4B | 7.50 | 61.7 | 0 | 0 | 0 | 26.7 | 16.0 |
| **LabVLA** | **4B** | **48.3** | **98.3** | **65.8** | **34.2** | **87.5** | **85.8** | **70.0** |

**表格说明**: LabVLA 在 ID/OOD 均达到最高平均成功率（71.1%/70.0%），超越 π₀（63.3%/63.2%）约 7.8/6.8 pp。值得注意的是 π₀-FAST 在本 benchmark 上表现最差（16.9%/19.7%），表明纯 FAST 路线对实验室场景适应不足。

### Table 3: LabEmbodied-Data 跨模型迁移效果

| Method | Size | Pick Up | Open Door | Pour Liquid | Heat Beaker | Transport Beaker | Avg. | Δ |
|--------|------|---------|-----------|------------|------------|------------------|------|---|
| **In-Distribution** | | | | | | | | |
| X-VLA | <1B | 27.5 | 65.0 | 45.0 | 25.8 | 83.3 | 49.3 | — |
| X-VLA + LabEmbodied | <1B | 26.7 | 69.2 | 59.2 | 68.3 | 98.3 | 64.3 | **+15.0** |
| **Out-of-Distribution** | | | | | | | | |
| X-VLA | <1B | 27.5 | 59.2 | 25.0 | 39.2 | 67.5 | 43.7 | — |
| X-VLA + LabEmbodied | <1B | 31.7 | 63.3 | 65.0 | 65.0 | 90.0 | 63.0 | **+19.3** |

**表格说明**: 用 LabEmbodied-Data 微调 X-VLA，ID 提升 15.0 pp，OOD 提升 19.3 pp，验证数据集的跨模型迁移价值。

### Table 4: 真实机器人评测（Franka 平台，每设置 50 次 rollout）

| Task | Clutter | LabVLA | DreamZero | π₀.₅ |
|------|---------|--------|-----------|------|
| **Shake Liquid — In-domain** | ✗ | 92 | 90 | 92 |
| **Shake Liquid — In-domain** | ✓ | 86 | 84 | 80 |
| **Shake Liquid — Out-of-domain** | ✗ | 84 | 84 | 82 |
| **Shake Liquid — Out-of-domain** | ✓ | 80 | 80 | 78 |
| **Pour Liquid — In-domain** | ✗ | 86 | 88 | 82 |
| **Pour Liquid — In-domain** | ✓ | 78 | 80 | 74 |
| **Pour Liquid — Out-of-domain** | ✗ | 76 | 72 | 74 |
| **Pour Liquid — Out-of-domain** | ✓ | 72 | 70 | 68 |
| **Magnetic Stir — In-domain** | ✗ | 88 | 86 | 88 |
| **Magnetic Stir — In-domain** | ✓ | 80 | 84 | 80 |
| **Magnetic Stir — Out-of-domain** | ✗ | 80 | 78 | 82 |
| **Magnetic Stir — Out-of-domain** | ✓ | 74 | 80 | 76 |
| **Stopper Plug/Unplug — In-domain** | ✗ | 80 | 84 | 78 |
| **Stopper Plug/Unplug — In-domain** | ✓ | 76 | 76 | 72 |
| **Stopper Plug/Unplug — Out-of-domain** | ✗ | 80 | 78 | 70 |
| **Stopper Plug/Unplug — Out-of-domain** | ✓ | 70 | 72 | 64 |
| **Average — In-domain** | ✗ | **86.5** | 87.0 | 85.0 |
| **Average — In-domain** | ✓ | **80.0** | 81.0 | 76.5 |
| **Average — Out-of-domain** | ✗ | **80.0** | 78.0 | 77.0 |
| **Average — Out-of-domain** | ✓ | **74.0** | 75.5 | 71.5 |

**表格说明**: LabVLA 在 OOD 场景（无杂乱 80.0% vs. DreamZero 78.0%）表现出色，有杂乱 OOD 场景下（74.0% vs. 75.5%）略低于 DreamZero，整体与 DreamZero 持平并超越 π₀.₅。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LabAssetLibrary | 2,947 个 3D 资产 | 实验室专用设备（移液管、烧杯、离心机等），带语义标注 | 仿真场景构建 |
| LabSceneLibrary | 10,000 场景 | 程序化生成，6 轴域随机化 | 仿真训练环境 |
| LabEmbodied-Data | 多相机 RGB + 状态/动作 | 15 类标注，支持 16 机器人平台 | VLA 训练 |
| Open X-Embodiment | 大规模通用 | 跨具身异构数据 | 预训练辅助 |
| LabUtopia Benchmark | 6 任务类型 × ID/OOD | Pick/Press/Open/Pour/Heat/Transport | 评测 |

### 实现细节

- **VLM Backbone**: Qwen3-VL-4B-Instruct（隐层维度 2560）
- **DiT**: 18 层，宽度 1024，8 注意力头
- **Flow Matching 推理**: N=10 Euler 步
- **KI 权重**: $\alpha = 10$
- **真实机器人**: Franka（单臂），每设置 50 rollouts
- **对比方法**: SmolVLA、X-VLA、GR00T N1.5、π₀、π₀.₅、π₀-FAST、InternVLA-A1、Wall-oss-flow、DreamZero

### 可视化结果

LabVLA 在 Press Button（100% ID，98.3% OOD）和 Open Door（65.0% ID/OOD）上表现最强；Pour Liquid（43.3% ID，34.2% OOD）相对较弱，体现液体操作的固有难度。真实机器人 Franka 平台验证了从仿真到真实的迁移能力（最高 92%）。

---

## 批判性思考

### 优点

1. **填补领域空白**: 首次系统建立实验室自动化 VLA 数据基础设施（资产库 + 场景库 + benchmark），具有强烈的应用价值
2. **数据引擎可复用**: RoboGenesis 的三阶段管道独立于 LabVLA 模型，可为其他 VLA 项目提供实验室数据
3. **Knowledge Insulation 设计合理**: stop-gradient 机制有效防止跨具身微调的能力退化，且在 Table 3 中以 +15/+19 pp 验证了数据迁移价值
4. **跨具身支持广泛**: 16 种机器人平台覆盖当前主流硬件，具备通用性

### 局限性

1. **液体操作仍是瓶颈**: Pour Liquid ID 43.3%/OOD 34.2%，液体动力学仿真与真实差距大，sim-to-real 难度最高
2. **安全约束未建模**: 化学试剂、危险操作的安全约束暂未纳入训练目标，限制实际部署
3. **真实数据规模小**: 真实机器人评测仅覆盖 4 类任务，Franka 单一平台，真实世界泛化性有待验证
4. **协议适应性缺失**: 当前模型执行固定协议，缺乏面对意外情况的动态调整能力（能力金字塔仍在低层）

### 潜在改进方向

1. 引入液体动力学仿真（如 PhysX Fluid）提升液体操作的 sim-to-real 忠实度
2. 加入安全约束奖励或 [[CBF|控制屏障函数]] 实现安全感知操作
3. 扩展多臂协作任务（双臂操作、移动操作臂的组合）

### 可复现性评估

- [x] 代码开源（[GitHub](https://github.com/zjunlp/LabVLA)）
- [x] 预训练模型（[HuggingFace](https://huggingface.co/zjunlp/LabVLA)）
- [x] 训练细节完整（两阶段 recipe 完整描述）
- [ ] 完整数据集公开（LabEmbodied-Data 发布情况待确认）

---

## 关联笔记

### 基于

- [[FAST Action Tokenizer]]: Stage 1 预训练的动作 tokenizer
- [[Flow Matching]]: Stage 2 后训练的动作生成范式
- [[DiT]]: 动作专家骨干网络

### 对比

- [[pi0-FAST]]: 同为 FAST + flow matching，但缺乏实验室专域数据
- [[SmolVLA]]: <1B 小模型基线
- [[UniVLA]]: 跨具身通用 VLA 对比

### 方法相关

- [[Cross-Embodiment Learning]]: 支持 16 机器人平台的核心挑战
- [[Action Chunking]]: 动作块预测的基础表示
- [[RoboGenesis]]: 数据合成引擎
- [[Knowledge Distillation]]: KI 机制的相关概念（stop-gradient 防遗忘）

### 硬件/数据相关

- [[ManiSkill]]: 对比的仿真引擎之一
- [[RoboTwin2]]: 对比的仿真引擎之一

---

## 速查卡片

> [!summary] LabVLA: Grounding VLAs in Scientific Laboratories
> - **核心**: 专为实验室自动化设计的 VLA，RoboGenesis 合成数据 + 两阶段训练
> - **方法**: Qwen3-VL-4B + DiT；FAST 预训练 + Flow Matching 后训练 + Knowledge Insulation
> - **结果**: LabUtopia 71.1%/70.0% ID/OOD，超 π₀ 约 7.8/6.8 pp；真实 Franka 最高 92%
> - **代码**: [GitHub](https://github.com/zjunlp/LabVLA)

---

*笔记创建时间: 2026-06-12*
