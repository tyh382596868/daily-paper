---
title: "RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models"
method_name: "RoVLA"
authors: [Jingzhou Luo, Yifan Wen, Yongjie Bai, Xinshuai Song, Yang Liu, Liang Lin]
year: 2026
venue: arXiv
tags: [vla, robustness, multi-consistency, flow-matching, adversarial-perturbation, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.19678v1
created: 2026-05-20
---

# 论文笔记：RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 中山大学 HCP Lab（HCPLab-SYSU） |
| 日期 | May 2026 |
| 项目主页 | https://github.com/HCPLab-SYSU/RoVLA |
| 对比基线 | [[π₀]] / [[π₀.₅]] / [[OpenVLA]] / [[GR00T N1.7]] / InternVL3.5+DiT |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19678) / [HTML](https://arxiv.org/html/2605.19678v1) / [Code](https://github.com/HCPLab-SYSU/RoVLA) |

---

## 一句话总结

> RoVLA 把"鲁棒性"当成显式训练目标，在指令语义、轨迹演化、观测扰动三个维度施加[[多一致性约束]]，让 [[VLA]] 在 [[LIBERO]]-Plus 上从 77.1% 提到 82.0%，并在语言改写场景上获得 +27 点的巨大增益。

---

## 核心贡献

1. **把鲁棒性变成显式目标**: 指出现有 [[VLA]] 的鲁棒性多是数据扩展的"副产品"，而非端到端学习中被显式优化的对象；RoVLA 在三种最常诱发失败的变换（指令语义变换、轨迹演化变换、观测扰动变换）下显式约束策略。
2. **[[Instructional Consistency|指令一致性 (IC)]]**: 用 Qwen3-8B 生成语义等价的指令改写集合，训练中均匀采样，隐式正则化策略把多样语言表达映射到一致的任务语义。
3. **[[Evolutionary Consistency|演化一致性 (EC)]]**: 约束 [[Flow Matching|流匹配]] 去噪过程不同时间步预测的速度场彼此一致，保证动作意图在整个生成过程中连贯。
4. **[[Observational Consistency|观测一致性 (OC)]]**: 对语义特征与机器人状态施加[[对抗扰动]]，要求扰动分支的预测与干净分支（stop-gradient）对齐，提升对视觉/本体感知扰动的鲁棒性。
5. **强结果**: [[LIBERO]]-Plus 上 disturbance-exposed 82.0%、zero-shot 74.3%，[[RoboTwin]] 2.0 randomized 50.0%，真机 5 任务平均 60% 成功率，全面超越强基线。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在分布偏移下非常脆弱。论文把失败归纳为三类"伪能力"：

- **伪视觉理解 (pseudo visual understanding)**: 策略对视角、背景、光照高度敏感。
- **伪语言遵循 (pseudo language following)**: 语义等价但措辞不同的指令会产生不一致的动作。
- **伪组合泛化 (pseudo compositional generalization)**: 多个轻微扰动叠加会引发严重性能崩塌。

### 现有方法的局限

论文将既有鲁棒性工作分为三类，并指出它们都把鲁棒性当作副产品：

1. **数据扩展**: [[π₀.₅]]、异构机器人数据，靠规模换鲁棒性。
2. **预测建模**: [[UniVLA]]、[[WorldVLA]]、RynnVLA，引入未来预测增强表征。
3. **后训练适配**: RIPT-VLA、VLA-RL、RobustVLA，通过额外阶段微调。

诊断 benchmark（[[LIBERO]]-Plus 揭示 7 个扰动维度的脆弱性、VLATest 揭示场景模糊/光照/相机位姿失败）只能"发现问题"，没有把鲁棒性变成训练目标。

### 本文的动机

作者认为鲁棒性应当是端到端策略学习中的**显式优化对象**。关键观察是：诱发 [[VLA]] 失败的扰动可归结为三类变换——指令语义变换、轨迹演化变换、观测扰动变换。如果在训练中对策略在这三类变换下施加一致性约束，就能直接逼模型学到稳定不变的任务表征，而不依赖海量数据的隐式覆盖。

---

## 方法详解

### 模型架构

<!-- RoVLA 的双系统主干 + 三一致性约束 -->

RoVLA 采用 **[[双系统架构]]（高层语义 + 低层动作生成）** 作为骨干，并在其上叠加三个一致性约束：

- **输入**: 语言指令 $T$ + 多视角图像观测 $\mathbf{I}_t$ + 机器人本体状态 $\mathbf{q}_t$
- **高层 — 语义编码器**: [[InternVL]]3.5-2B，作为 [[VLM]] 抽取语言 token $l_t \in \mathbb{R}^{N_l \times d}$ 与视觉 token $v_t \in \mathbb{R}^{N_v \times d}$；取第 16 层 decoder 的中层语义特征
- **低层 — 动作生成器**: 32 层的 [[DiT]]（[[扩散变换器]]），通过 [[AdaLN]] 注入去噪时间步 $\tau$，把状态 $\mathbf{q}_t$ 与含噪动作 $\mathbf{A}_t^\tau$ 投影到共享潜空间
- **动作生成范式**: [[Flow Matching|条件流匹配]]，沿线性概率路径预测速度场
- **输出**: [[Action Chunking|动作块]] $\mathbf{A}_t \in \mathbb{R}^{L \times d_a}$（chunk 长度 $L=16$）

三个一致性约束 [[Instructional Consistency|IC]]、[[Evolutionary Consistency|EC]]、[[Observational Consistency|OC]] 只作用于训练阶段，推理时回到标准双系统前向 + $K$ 步去噪，不增加部署成本。

### 核心模块

#### 模块 1: [[Instructional Consistency|指令一致性 (IC)]]

**设计动机**: 让策略在面对语义等价但措辞不同的指令时，把它们都映射到一致的任务语义，缓解"伪语言遵循"。

**具体实现**:
- 用 Qwen3-8B（关闭 thinking 模式）为每条原始指令生成语义等价改写集合 $\mathcal{D}_T$
- 设计 **7 个 prompt 模板**：用户意图式、功能目标式、礼貌请求式、简洁命令式、教学式、抽象目标式、功能指代式
- 每模板最多采样 5 条，解码参数 temperature $=0.7$、top_p $=0.9$、max_new_tokens $=512$
- 后处理：把 "The user wants" 规范化为 "I want"，过滤空输出/少于 8 字符/拒答（"sorry"、"unable"）的候选，并去重
- 训练时每次迭代从 $\mathcal{D}_T$ 中**均匀采样**一条指令，不引入额外损失项 —— IC 是通过数据层面的一致性隐式正则化

#### 模块 2: [[Evolutionary Consistency|演化一致性 (EC)]]

**设计动机**: [[Flow Matching|流匹配]] 通过多步去噪生成动作，作者希望不同去噪阶段预测的速度场指向一致的任务意图，避免生成轨迹"中途变心"。

**具体实现**:
- 同一样本采样两个时间步 $\tau_1, \tau_2 \in [0,1]$，分别得到含噪动作 $\mathbf{A}_t^{\tau_1}$、$\mathbf{A}_t^{\tau_2}$
- 用 [[DiT]] 计算两份干净（无扰动）速度场预测 $\hat{\mathbf{v}}_{\text{clean}}^{\tau_1}$、$\hat{\mathbf{v}}_{\text{clean}}^{\tau_2}$
- 用 L2 距离约束两者一致（公式 10）
- 时间步采样不用均匀分布，而用偏向大 $\tau$ 的 [[Beta 分布时间步采样|Beta 分布]]（公式 11），让约束更多落在接近干净动作的去噪后期

#### 模块 3: [[Observational Consistency|观测一致性 (OC)]]

**设计动机**: 缓解"伪视觉理解"——视觉特征或本体状态发生扰动时，策略输出应保持稳定。

**具体实现**:
- 对视觉 token $v_t$ 与机器人状态 $\mathbf{q}_t$ 施加 **[[对抗扰动]]**：沿 $\mathcal{L}_{\text{EC}}$ 梯度方向归一化后加扰（公式 12-13），步长用 $\min(\alpha, \epsilon_{\text{adv}}\|\cdot\|_2)$ 自适应裁剪
- 把扰动后的特征送入 [[DiT]]，得到扰动分支预测 $\hat{\mathbf{v}}_{\text{pert}}^{\tau_i}$
- 用 [[Observational Consistency|OC 损失]] 要求扰动分支对齐干净分支，并对干净分支施加 **stop-gradient** $\operatorname{sg}(\cdot)$，防止梯度回流污染未扰动分支（公式 14）
- 超参数 $\alpha = 0.01$、$\epsilon_{\text{adv}} = 0.03$

#### 模块 4: [[自适应损失加权]]训练目标

**设计动机**: 一致性正则项与监督损失量纲不同，固定权重难以平衡；用 EMA 跟踪监督损失自适应调整正则权重。

**具体实现**: 监督损失对干净分支和扰动分支都回归 ground-truth 速度（公式 15-17），一致性正则把 EC、OC 合并（公式 18），总损失用随训练自适应衰减的权重 $\lambda_j$（公式 19-21）。

---

## 关键公式

### 公式 1: [[Action Chunking|动作块]]定义

$$
\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+L-1}] \in \mathbb{R}^{L \times d_a}
$$

**含义**: 策略一次预测长度为 $L$ 的动作块，而非单步动作。

**符号说明**:
- $L$: 动作块长度（实验取 16）
- $d_a$: 单步动作维度
- $\mathbf{a}_{t+i}$: 第 $t+i$ 步的动作向量

### 公式 2: 线性概率路径

$$
\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t^{\text{gt}} + (1-\tau)\,\epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \tau \in [0,1]
$$

**含义**: [[Flow Matching|流匹配]] 在 ground-truth 动作与高斯噪声之间构造线性插值路径。

**符号说明**:
- $\tau$: 流匹配时间步（自由变量），$\tau=0$ 为纯噪声、$\tau=1$ 为干净动作
- $\mathbf{A}_t^{\text{gt}}$: ground-truth 动作块
- $\epsilon$: 标准高斯噪声

### 公式 3: Ground-truth 速度场

$$
\mathbf{v}^{\text{gt}} = \mathbf{A}_t^{\text{gt}} - \epsilon
$$

**含义**: 线性路径下从噪声指向干净动作的目标速度，与 $\tau$ 无关。

### 公式 4: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau}\Big[\, \big\| \mathbf{V}_\theta(T, \mathbf{I}_t, \mathbf{A}_t^{\tau}, \mathbf{q}_t, \tau) - \mathbf{v}^{\text{gt}} \big\|_2^2 \,\Big]
$$

**含义**: 标准条件流匹配训练目标，让网络预测的速度场回归 ground-truth 速度。

**符号说明**:
- $\mathbf{V}_\theta$: 整个 RoVLA 策略网络（VLM + DiT）
- $T$: 语言指令；$\mathbf{I}_t$: 图像观测；$\mathbf{q}_t$: 机器人状态
- $\mathbb{E}_{\tau}$: 对时间步 $\tau$ 求期望

### 公式 5: [[VLM]] 语义编码

$$
[\,l_t,\; v_t\,] = \operatorname{VLM}_{\theta_1}(T, \mathbf{I}_t)
$$

**含义**: 高层语义编码器（[[InternVL]]3.5-2B）把指令与图像编码成语言 token $l_t$ 和视觉 token $v_t$。

**符号说明**:
- $\theta_1$: VLM 参数（仅最后 4 层解冻）
- $l_t \in \mathbb{R}^{N_l \times d}$、$v_t \in \mathbb{R}^{N_v \times d}$: 语言/视觉 token

### 公式 6: [[DiT]] 动作生成器

$$
\hat{\mathbf{v}}^{\tau} = \operatorname{DiT}_{\theta_2}(l_t, v_t, \mathbf{A}_t^{\tau}, \mathbf{q}_t, \tau)
$$

**含义**: 低层动作生成器以语义 token、含噪动作、状态、时间步为条件预测速度场。

**符号说明**:
- $\theta_2$: 32 层 DiT 参数（随机初始化）
- $\hat{\mathbf{v}}^{\tau}$: 时间步 $\tau$ 处预测的速度场

### 公式 7: 语义等价指令集合

$$
\mathcal{D}_T = \{\,T^{(1)}, \ldots, T^{(N_{\text{lang}})}\,\}
$$

**含义**: [[Instructional Consistency|IC]] 用 Qwen3-8B 为每条指令构造的语义等价改写集合。

**符号说明**:
- $N_{\text{lang}}$: 改写数量（LIBERO/真机约 15 条，RoboTwin 2.0 用 100 条原生多样指令）

### 公式 8-9: 双时间步干净速度预测

$$
\hat{\mathbf{v}}_{\text{clean}}^{\tau_1} = \operatorname{DiT}_{\theta_2}(l_t, v_t, \mathbf{A}_t^{\tau_1}, \mathbf{q}_t, \tau_1)
$$

$$
\hat{\mathbf{v}}_{\text{clean}}^{\tau_2} = \operatorname{DiT}_{\theta_2}(l_t, v_t, \mathbf{A}_t^{\tau_2}, \mathbf{q}_t, \tau_2)
$$

**含义**: 在两个独立采样的去噪时间步 $\tau_1, \tau_2$ 处分别得到干净（无对抗扰动）的速度场预测。

### 公式 10: [[Evolutionary Consistency|演化一致性损失]]

$$
\mathcal{L}_{\text{EC}} = \big\| \hat{\mathbf{v}}_{\text{clean}}^{\tau_1} - \hat{\mathbf{v}}_{\text{clean}}^{\tau_2} \big\|_2^2
$$

**含义**: 约束不同去噪阶段预测的速度场一致，使动作意图在整个生成过程中连贯。

**符号说明**:
- $\tau_1, \tau_2$: 两个独立采样的时间步

### 公式 11: [[Beta 分布时间步采样]]

$$
p(\tau) = \operatorname{Beta}\!\left(\frac{s-\tau}{s};\; 1.5,\; 1\right), \quad s = 0.999
$$

**含义**: 时间步不采用均匀分布，而用偏向大 $\tau$（接近干净动作）的 Beta 分布，使一致性约束更聚焦于去噪后期阶段。

**符号说明**:
- $s = 0.999$: 缩放常数
- $\operatorname{Beta}(\cdot; 1.5, 1)$: 形状参数 $(1.5, 1)$ 的 Beta 分布

### 公式 12: 视觉特征对抗扰动

$$
\tilde{v}_t = v_t + \eta_v \cdot \frac{\nabla_{v_t}\mathcal{L}_{\text{EC}}}{\big\| \nabla_{v_t}\mathcal{L}_{\text{EC}} \big\|_2}, \quad \eta_v = \min\!\big(\alpha,\; \epsilon_{\text{adv}}\|v_t\|_2\big)
$$

**含义**: 沿 $\mathcal{L}_{\text{EC}}$ 的归一化梯度方向对视觉 token 施加对抗扰动，步长 $\eta_v$ 用绝对上界 $\alpha$ 与相对上界 $\epsilon_{\text{adv}}\|v_t\|_2$ 取小值自适应裁剪。

**符号说明**:
- $\nabla_{v_t}\mathcal{L}_{\text{EC}}$: EC 损失对视觉 token 的梯度
- $\alpha = 0.01$: 扰动幅度绝对上界
- $\epsilon_{\text{adv}} = 0.03$: 相对扰动比例

### 公式 13: 机器人状态对抗扰动

$$
\tilde{\mathbf{q}}_t = \mathbf{q}_t + \eta_q \cdot \frac{\nabla_{\mathbf{q}_t}\mathcal{L}_{\text{EC}}}{\big\| \nabla_{\mathbf{q}_t}\mathcal{L}_{\text{EC}} \big\|_2}, \quad \eta_q = \min\!\big(\alpha,\; \epsilon_{\text{adv}}\|\mathbf{q}_t\|_2\big)
$$

**含义**: 同公式 12，对机器人本体状态 $\mathbf{q}_t$ 施加对抗扰动，模拟本体感知噪声。

### 公式 14: [[Observational Consistency|观测一致性损失]]

$$
\mathcal{L}_{\text{OC}} = \frac{1}{2}\sum_{i=1}^{2} \big\| \hat{\mathbf{v}}_{\text{pert}}^{\tau_i} - \operatorname{sg}\!\big(\hat{\mathbf{v}}_{\text{clean}}^{\tau_i}\big) \big\|_2^2
$$

**含义**: 要求扰动分支预测 $\hat{\mathbf{v}}_{\text{pert}}^{\tau_i}$ 对齐干净分支预测；干净分支用 stop-gradient 包裹，作为不动的目标，梯度只更新扰动分支。

**符号说明**:
- $\operatorname{sg}(\cdot)$: stop-gradient 算子，阻断梯度回流
- $\hat{\mathbf{v}}_{\text{pert}}^{\tau_i}$: 扰动特征经 DiT 得到的速度场预测
- 求和遍历两个时间步 $i \in \{1,2\}$

### 公式 15-16: 干净分支与扰动分支监督损失

$$
\mathcal{L}^{\text{clean}} = \frac{1}{2}\sum_{i=1}^{2} \big\| \hat{\mathbf{v}}_{\text{clean}}^{\tau_i} - \mathbf{v}^{\text{gt}} \big\|_2^2
$$

$$
\mathcal{L}^{\text{pert}} = \frac{1}{2}\sum_{i=1}^{2} \big\| \hat{\mathbf{v}}_{\text{pert}}^{\tau_i} - \mathbf{v}^{\text{gt}} \big\|_2^2
$$

**含义**: 在两个时间步上，分别让干净分支和扰动分支的速度预测回归 ground-truth 速度 $\mathbf{v}^{\text{gt}}$。

### 公式 17: [[自适应损失加权|SFT 损失]]

$$
\mathcal{L}_{\text{SFT}} = \frac{1}{2}\big(\mathcal{L}^{\text{clean}} + \mathcal{L}^{\text{pert}}\big)
$$

**含义**: 监督微调损失是干净分支与扰动分支监督损失的平均。

### 公式 18: 一致性正则项

$$
\mathcal{L}_{\text{C}} = \frac{1}{2}\big(\mathcal{L}_{\text{EC}} + \mathcal{L}_{\text{OC}}\big)
$$

**含义**: 把演化一致性与观测一致性损失合并为统一的一致性正则项。

### 公式 19: 总训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda_j \cdot \mathcal{L}_{\text{C}}
$$

**含义**: 总损失 = 监督损失 + 自适应加权的一致性正则项。

**符号说明**:
- $\lambda_j$: 第 $j$ 步的自适应权重（公式 20-21）
- $j$: 训练迭代步索引

### 公式 20-21: [[自适应损失加权|自适应权重]]与其 [[EMA]]

$$
\lambda_j = \frac{1}{1 + \mathcal{L}_j^{\text{ema}}}
$$

$$
\mathcal{L}_j^{\text{ema}} = \gamma \cdot \mathcal{L}_{j-1}^{\text{ema}} + (1-\gamma)\cdot \mathcal{L}^{\text{clean}}, \quad \gamma = 0.95, \quad \mathcal{L}_0^{\text{ema}} = 100
$$

**含义**: 用 [[EMA]] 跟踪干净分支监督损失 $\mathcal{L}^{\text{clean}}$；监督损失大（训练早期）时 $\lambda_j$ 小，正则约束弱；监督损失下降后 $\lambda_j$ 增大，一致性约束逐渐主导。这实现了"先学好基本策略、再强化鲁棒性"的课程式调度。

**符号说明**:
- $\gamma = 0.95$: EMA 衰减系数
- $\mathcal{L}_0^{\text{ema}} = 100$: EMA 初值（故意取大，使训练初期 $\lambda_j$ 接近 0）

### 公式 22-23: 推理 —— 高斯初始化 + 前向 Euler 积分

$$
\mathbf{A}_t^{0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

$$
\mathbf{A}_t^{\tau + \Delta\tau} = \mathbf{A}_t^{\tau} + \Delta\tau \cdot \operatorname{DiT}_{\theta_2}(l_t, v_t, \mathbf{A}_t^{\tau}, \mathbf{q}_t, \tau), \quad \Delta\tau = \frac{1}{K}
$$

**含义**: 推理时从高斯噪声出发，用前向 Euler 法以 $K$ 步积分 ODE 还原动作块；实验取 $K=8$。

**符号说明**:
- $K = 8$: 去噪步数
- $\Delta\tau$: 单步步长

---

## 关键图表

### Figure 1: VLA 鲁棒性失败示例

![Figure 1](https://arxiv.org/html/2605.19678v1/x1.png)

**说明**: 展示现有 [[VLA]] 在视觉观测变化和"语义等价但措辞不同"的指令改写下缺乏鲁棒性——同一任务换个说法或换个光照/视角，策略就会失败。这是 RoVLA 提出"伪视觉理解 / 伪语言遵循 / 伪组合泛化"三类失败模式的动机起点。

### Figure 2: RoVLA 总体架构

![Figure 2](https://arxiv.org/html/2605.19678v1/x2.png)

**说明**: (a) 训练阶段：RoVLA 采用[[双系统架构]]——高层语义抽取（[[InternVL]]3.5-2B）+ 低层动作生成（[[DiT]]），并叠加三个一致性约束 [[Instructional Consistency|IC]]、[[Evolutionary Consistency|EC]]、[[Observational Consistency|OC]]。(b) 推理阶段：策略回到标准双系统前向，执行 $K$ 步去噪生成动作，三个约束不引入额外部署成本。

### Figure 3: 真机评估任务

![Figure 3](https://arxiv.org/html/2605.19678v1/x3.png)

**说明**: 在 Franka Research 3 机器人（配腕部相机）上构建 5 个桌面操作任务，评估真实环境下的感知、指令理解与操作能力。

### Figure 4: LIBERO-Plus 定性结果

![Figure 4](https://arxiv.org/html/2605.19678v1/x4.png)

**说明**: 在多种扰动条件下——(a) 物体数量增多、(b) 语言指令表达变化、(c) 光照变化、(d) 注入相机模糊、(e) 视角变化——RoVLA 仍能维持相对稳定的动作决策。

### Figure 5: RoboTwin 2.0 定性结果

![Figure 5](https://arxiv.org/html/2605.19678v1/x5.png)

**说明**: 主要展示 Randomized 环境下的代表性 rollout 快照。尽管物体位姿、场景布局、视觉外观发生变化，RoVLA 仍保持相对稳定的目标 grounding 与执行轨迹。

### Figure 6: 真机任务 rollout（腕部相机视角）

![Figure 6](https://arxiv.org/html/2605.19678v1/x6.png)

**说明**: 从腕部相机视角可视化 5 个真机桌面操作任务的代表性 rollout。即使各次试验的初始物体配置不同，RoVLA 仍维持相对稳定的执行轨迹。论文特别提到在拉抽屉任务上观察到未经显式训练的失败-恢复行为。

### Table 1: LIBERO-Plus Zero-Shot 扰动泛化（训练于 LIBERO）

| 扰动维度 | RoVLA | InternVL3.5+DiT | 最强基线 | RoVLA 相对最强基线 |
|---|---|---|---|---|
| Camera | 58.4 | 58.3 | π₀-Fast: 65.1 | -6.7 |
| Robot | 36.3 | 37.2 | RIPT-VLA: 31.2 | +5.1 |
| Language | 92.9 | 76.3 | OpenVLA-OFT: 79.5 | +13.4 |
| Light | 95.6 | 95.1 | OpenVLA-OFT: 88.7 | +7.1 |
| Background | 95.0 | 94.8 | OpenVLA-OFT: 93.3 | +1.7 |
| Noise | 80.9 | 79.1 | RIPT-VLA: 73.5 | +7.4 |
| Layout | 73.0 | 74.0 | π₀-Fast: 68.8 | +4.2 |
| **Total** | **74.3** | 71.6 | OpenVLA-OFT: 69.6 | **+2.7** |

**说明**: 仅在干净 LIBERO（1,693 demo）上训练、直接迁移到 7 个扰动维度。最大增益来自 Language（92.9%），比 InternVL3.5+DiT 高 16.6 点、比最强基线高 13.4 点。Camera 维度反而落后 π₀-Fast，是后续局限性讨论的重点。

### Table 2: LIBERO-Plus Disturbance-Exposed 泛化（训练于 LIBERO-Plus）

| 扰动维度 | RoVLA | InternVL3.5+DiT | GR00T-N1.6 | OpenVLA-OFT |
|---|---|---|---|---|
| Camera | **96.6** | 94.2 | 92.6 | 92.8 |
| Robot | 32.0 | 32.4 | 33.5 | 30.3 |
| Language | **91.5** | 64.5 | 80.1 | 85.8 |
| Light | **95.9** | 94.7 | 93.6 | 94.9 |
| Background | **96.1** | 93.0 | 95.4 | 93.9 |
| Noise | **95.1** | 94.7 | 93.6 | 89.3 |
| Layout | 74.1 | 74.8 | 75.0 | 77.6 |
| **Total** | **82.0** | 77.1 | 79.4 | 79.5 |

**说明**: 在含扰动的 LIBERO-Plus（15,874 demo）上训练。RoVLA 总分 82.0%，超 InternVL3.5+DiT 4.9 点。Language 维度 91.5% 比 InternVL3.5+DiT 高 27.0 点。Robot 和 Layout 仍是短板（甚至略低于个别基线）。

### Table 3: RoboTwin 2.0 总体结果

| 方法 | Clean | Randomized |
|---|---|---|
| GO-1 | 37.8 | — |
| [[π₀.₅]] | 43.0 | 43.8 |
| InternVL3.5+DiT | 44.9 | 45.4 |
| **RoVLA** | **48.2** | **50.0** |

**说明**: RoboTwin 2.0 含 50 个双臂任务。RoVLA 在 Clean 环境 48.2%（超 InternVL3.5+DiT +3.3、超 π₀.₅ +5.2），Randomized 环境 50.0%（超 InternVL3.5+DiT +4.6、超 π₀.₅ +6.2）。值得注意的是 RoVLA 在 Randomized 下反而比 Clean 略高，说明对随机化扰动有正向适应。

### Table 4: RoboTwin 2.0 代表性任务（节选）

| 任务 | Clean | Randomized |
|---|---|---|
| Click Bell | 97 | 94 |
| Click Alarmclock | 90 | 94 |
| Place Container Plate | 92 | 89 |
| Dump Bin Bigbin | 82 | 76 |
| Open Laptop | 75 | 60 |
| Place Dual Shoes | 32 | 28 |
| Handover Mic | 34 | 34 |
| Stack Blocks Three | 1 | 4 |

**说明**: 简单 grounding 类任务（敲钟、敲闹钟）接近饱和；涉及精细接触动力学与双臂协调的任务（递麦克风、三块堆叠、双鞋摆放）依然很难，三块堆叠几乎完全失败，是论文承认的局限。

### Table 5: 真机评估结果（Franka Research 3，5 任务）

| 任务 | GR00T-N1.6 | InternVL3.5+DiT | RoVLA |
|---|---|---|---|
| Pick Up Banana | 80 | 60 | **80** |
| Pick Up Apple | 50 | 60 | **70** |
| Put Banana in Bowl | 70 | 50 | **80** |
| Put Apple in Bowl | 40 | 20 | **50** |
| Put Apple in Drawer | 10 | 0 | **20** |
| **Average** | 50 | 38 | **60** |

**说明**: 每任务 25 条示教（共 125 条）。RoVLA 平均 60%，比 GR00T-N1.6 高 10 点、比 InternVL3.5+DiT 高 22 点。最难的 Put Apple in Drawer（含抽屉交互）所有方法都很低，RoVLA 20% 仍为最佳。

### Table 6: 消融实验（LIBERO-Plus，Disturbance-Exposed）

| IC | EC | OC | Camera | Robot | Language | Light | Bg | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| × | × | × | 94.2 | 32.4 | 64.5 | 94.7 | 93.0 | 94.7 | 74.8 | 77.1 |
| × | ✓ | × | 94.6 | 32.5 | 68.4 | 94.0 | 94.5 | 94.2 | 77.0 | 78.2 |
| ✓ | ✓ | × | 93.7 | 30.6 | 89.5 | 94.7 | 93.5 | 94.6 | 73.8 | 80.5 |
| × | ✓ | ✓ | 95.3 | 35.6 | 69.6 | 94.5 | 95.4 | 95.6 | 74.5 | 79.0 |
| ✓ | ✓ | ✓ | 96.6 | 32.0 | 91.5 | 95.9 | 96.1 | 95.1 | 74.1 | **82.0** |

**关键发现**:
- **EC 单独**：总分 +1.1（77.1→78.2），在 Layout 上增益最大（+2.2）。
- **IC+EC**：总分 +3.4（→80.5），Language 从基线 64.5 飙到 89.5（+25.0），说明 IC 的语言改写是语言鲁棒性的主要来源。
- **EC+OC**：总分 +1.9（→79.0），在 Robot（+3.2）和 Noise 上受益，对抗扰动主要提升观测鲁棒性。
- **IC+EC+OC（完整）**：82.0，三模块协同 —— OC 在 IC+EC 基础上把 Camera/Language/Background 进一步推高，三者互补而非冗余。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 1,693 demo（4 suite 合并） | 干净环境 | Zero-shot 实验训练 |
| [[LIBERO]]-Plus | 15,874 成功 demo，7 个扰动维度 | layout/camera/robot/language/light/background/noise | Disturbance-exposed 训练 + 测试 |
| [[RoboTwin]] 2.0 | 2,500 clean + 25,000 randomized demo | 50 个双臂任务 | 训练 + 测试 |
| 真机 (Franka R3) | 5 任务 × 25 demo（共 125） | 桌面操作，腕部相机 | 训练 + 测试 |

### 实现细节

- **语义编码器**: [[InternVL]]3.5-2B，取第 16 层 decoder 中层特征，仅最后 4 层解冻
- **动作生成器**: 随机初始化的 32 层 [[DiT]]，[[AdaLN]] 注入时间步
- **动作生成**: [[Flow Matching|条件流匹配]]，chunk 长度 $L=16$，推理 $K=8$ 步去噪
- **优化器**: AdamW，学习率 $1 \times 10^{-4}$（5% 线性 warmup + cosine 衰减）
- **训练步数**: 60k（LIBERO-Plus / 真机），120k（RoboTwin 2.0）
- **指令改写**: LIBERO / LIBERO-Plus / 真机用 Qwen3-8B 生成约 15 条改写；RoboTwin 2.0 用每任务 100 条原生多样指令
- **对抗扰动超参**: $\alpha = 0.01$、$\epsilon_{\text{adv}} = 0.03$；EMA 系数 $\gamma = 0.95$

### 可视化结果

定性 rollout（Fig 4-6）显示：在物体增多、语言改写、光照变化、相机模糊、视角变化等扰动下，RoVLA 维持稳定动作决策；在 RoboTwin 2.0 Randomized 环境下保持目标 grounding 稳定；真机上即使初始配置变化也能稳定执行，并出现未经训练的失败-恢复行为（拉抽屉时）。

---

## 批判性思考

### 优点

1. **问题框架清晰**: "伪视觉理解 / 伪语言遵循 / 伪组合泛化"三类失败 → 三种变换 → 三个一致性约束，定义与方法一一对应，逻辑非常工整。
2. **训练时约束、推理零开销**: IC/EC/OC 全部只作用于训练，推理回到标准双系统前向，部署成本与普通 VLA 一致，工程友好。
3. **IC 对语言鲁棒性效果显著**: 消融显示 Language 维度从 64.5 提到 91.5（+27 点），证明"语义等价改写 + 均匀采样"是一个简单但极有效的语言正则化手段。
4. **自适应权重设计巧妙**: 用 EMA 跟踪监督损失自动调度 $\lambda_j$，实现"先学策略再强鲁棒"的隐式课程，避免早期正则项压制基础学习。
5. **三模块协同验证充分**: 消融覆盖 EC、IC+EC、EC+OC、完整四种配置，证明三者互补而非冗余。

### 局限性

1. **Camera 维度反而退步**: Zero-shot 设置下 Camera 58.4 落后 π₀-Fast 的 65.1（-6.7），说明对抗扰动对视角偏移这种"几何性"扰动帮助有限，OC 的特征空间扰动并不能覆盖相机外参变化。
2. **Robot / Layout 始终是短板**: 两个基准上 Robot 维度都在 32-36 之间，Layout 也在 74 附近徘徊，甚至略低于个别基线；论文承认对精细接触动力学和复杂双臂协调改进有限。
3. **EC 的合理性存疑**: EC 强制不同去噪时间步的速度场一致，但流匹配里不同 $\tau$ 处的速度场本应不同（除非路径完全线性且模型完美）；这个约束更像"平滑正则"，作者没有给出理论依据说明它为何不损害生成质量。
4. **对抗扰动只在特征空间**: 扰动加在 VLM 输出的视觉 token 和状态向量上，而非原始像素；这能否真实对应部署时的视觉扰动（光照、模糊、遮挡）缺乏分析。
5. **改写依赖外部 LLM**: IC 的指令多样性完全来自 Qwen3-8B，改写质量、覆盖度、是否引入语义漂移没有定量评估；7 个模板的设计也偏经验性。
6. **三块堆叠近乎全失败**: RoboTwin 2.0 Stack Blocks Three 仅 1-4%，说明 RoVLA 对长程精细操作基本无能为力，鲁棒性提升不等于能力提升。

### 潜在改进方向

1. **几何感知扰动**: 针对 Camera 维度的退步，把 OC 扩展到显式的相机位姿/视角扰动，或引入 3D-aware 表征（如 [[VGGT]]）。
2. **接触动力学约束**: 按论文未来工作所述，引入更强的几何与动力学约束，提升精细接触和双臂协调。
3. **EC 的理论化**: 给出 EC 与流匹配速度场关系的理论分析，或改为约束去噪轨迹的"终点"而非"中途速度"。
4. **改写质量评估**: 对 Qwen3-8B 生成的指令做语义保真度过滤（如用嵌入相似度筛选），避免语义漂移污染训练。

### 可复现性评估

- [x] 代码开源（https://github.com/HCPLab-SYSU/RoVLA，论文标注 "Codes will be available"）
- [ ] 预训练模型（待确认）
- [x] 训练细节完整（lr、步数、超参 $\alpha/\epsilon_{\text{adv}}/\gamma$、解冻层都给了）
- [x] 数据集可获取（[[LIBERO]] / [[RoboTwin]] 2.0 公开，真机数据为自采）

---

## 关联笔记

### 基于

- [[InternVL]]: 高层语义编码器 InternVL3.5-2B
- [[DiT]]: 32 层动作生成器骨架
- [[Flow Matching]]: 动作生成的训练范式
- [[双系统架构]]: 高层语义 + 低层动作的骨干结构
- [[AdaLN]]: DiT 中注入去噪时间步的归一化

### 对比

- [[π₀.₅]] / [[π₀]]: 数据扩展路线的代表 VLA 基线
- [[OpenVLA]]: OpenVLA-OFT 是 LIBERO-Plus 多维度的强基线
- [[GR00T N1.7]]: GR00T-N1.6 在真机与 LIBERO-Plus 上作对比
- [[UniVLA]] / [[WorldVLA]]: 预测建模路线的鲁棒性方法

### 方法相关

- [[多一致性约束]]: 论文核心框架
- [[Instructional Consistency]]: 指令语义一致性
- [[Evolutionary Consistency]]: 轨迹演化一致性
- [[Observational Consistency]]: 观测扰动一致性
- [[对抗扰动]]: OC 的扰动生成机制
- [[Beta 分布时间步采样]]: EC 的时间步采样策略
- [[自适应损失加权]]: EMA 调度的正则权重
- [[Action Chunking|动作块]]: 输出形式

### 基准/数据相关

- [[LIBERO]]: LIBERO 与 LIBERO-Plus 评估基准
- [[RoboTwin]]: RoboTwin 2.0 双臂操作基准
- [[sim-to-real]]: 仿真训练 + 真机验证的范式

---

## 速查卡片

> [!summary] RoVLA
> - **核心**: 把鲁棒性当显式训练目标，在指令语义、轨迹演化、观测扰动三个维度施加[[多一致性约束]]
> - **方法**: 双系统骨干（InternVL3.5-2B + 32 层 DiT 流匹配）+ IC（指令改写一致性）+ EC（去噪时间步速度场一致）+ OC（对抗扰动一致性），EMA 自适应加权
> - **结果**: LIBERO-Plus 82.0%（disturbance-exposed）/ 74.3%（zero-shot），RoboTwin 2.0 randomized 50.0%，真机 5 任务平均 60%
> - **亮点**: Language 维度 +27 点；推理零额外开销
> - **代码**: https://github.com/HCPLab-SYSU/RoVLA

---

*笔记创建时间: 2026-05-20*
