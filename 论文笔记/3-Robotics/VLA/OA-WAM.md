---
title: "OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation"
method_name: "OA-WAM"
authors: [Yushan Liu, Peibo Sun, Shoujie Li, Yifan Xie, Lingfeng Zhang, Xintao Chao, Shiyuan Dong, Fang Chen, Xiao-Ping Zhang, Wenbo Ding]
year: 2026
venue: arXiv
tags: [vla, object-centric, world-model, robot-manipulation, robustness, flow-matching, slot-attention]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.06481
created: 2026-05-09
---

# 论文笔记：OA-WAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学 / 上海交通大学 / 南洋理工大学 |
| 日期 | May 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[Pi05\|π₀.₅]], [[Cosmos-Policy]], [[WorldVLA]], [[VLA-JEPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.06481) |

---

## 一句话总结

> OA-WAM 通过将每帧分解为 N+1 个"可寻址对象槽"，在 transformer 每层强制用固定身份地址向量路由注意力，从而赋予机器人操作策略真正的对象级鲁棒性。

---

## 核心贡献

1. **对象可寻址性诊断**: 揭示了现有 [[World Action Model|WAM]] 和 [[VLA]] 鲁棒性差的根本原因——全局 token 将对象身份与上下文纠缠，导致在场景扰动下策略崩溃
2. **Per-slot 地址/内容分解架构**: 提出冻结身份地址向量 $\text{addr}_k$ + 时变内容向量 $\text{cnt}_k^t$ 的槽设计，配合 addr-only key projection 和逐层地址重置钩子，实现架构级对象可寻址保证
3. **Swap-Binding 因果验证**: 设计对象地址交换实验，OA-WAM 获得 0.87 的 swap-binding 余弦相似度（基线 ≤ 0.09），直接验证策略决策确实由地址空间因果路由

---

## 问题背景

### 要解决的问题

机器人操作策略在标准 benchmark 上表现优异，但在轻度场景扰动（相机视角、目标位置、光照、背景、传感器噪声）下显著失效。

### 现有方法的局限

现有 [[World Action Model|WAM]] 变体（DreamerV3、TD-MPC2、UWM、WorldVLA）和 VLA 方法（[[Pi05|π₀.₅]]、OpenVLA）均使用全局 token，将对象的身份、外观、姿态和上下文混合在单一表示中。即使使用 Slot Attention 类方法，也没有**架构级保证**确保槽与对象的稳定绑定。

### 本文的动机

作者认为核心缺陷是"缺乏对象可寻址性"（lack of object addressability）：策略无法通过稳定的对象标识符查询特定对象的状态。若 cross-slot attention 的 key 仅读取固定的身份地址子向量，attention routing 就自然被"锁定"到该对象，不受上下文漂移影响。

---

## 方法详解

### 模型架构

OA-WAM 采用 **多模态 Transformer** 架构，以 [[Chameleon]] 7B 为骨干网络：

- **输入**: RGB 帧 $\mathbf{I}_t$，本体感知 $\mathbf{q}_t \in \mathbb{R}^7$，语言指令 $\ell$，历史动作 $\mathbf{a}_{<t}$
- **Backbone**: Chameleon-style 32 层 Transformer（7B 参数，冻结）
- **核心模块**: [[Object-Addressable Attention|可寻址对象注意力]] + [[Slot Tokenization|槽 Token 化]] + 逐层地址重置
- **输出**: [[Action Chunking|动作块]] $\mathbf{A}_t \in \mathbb{R}^{16 \times 7}$，以及辅助世界状态预测 $\hat{\mathcal{S}}_{t+1}$
- **可训练参数**: ~127M（80M LoRA + 47M 新增头部）

**六路并行 token 流**:
1. BPE 文本 token
2. [[Qwen3-VL]] 名词短语（仅作为 SAM 3 提示，不进主干）
3. [[Chameleon]] VQ-GAN 图像码
4. [[SAM 3]] + [[DINOv3]] + 姿态估计 → 对象槽 token（核心创新）
5. 256-bin 离散化本体感知
6. 256-bin 离散化历史动作

### 核心模块

#### 模块 1：对象槽 Token 化（Object-Slot Tokenization）

**设计动机**: 用 [[SAM 3]] 做实例分割，[[DINOv3]] 提取语义特征，得到每个对象的时变观测；再通过 [[Qwen3-VL]] 提取名词短语作为语言标签，构建冻结的身份地址。

**槽向量构成**（每槽 320 维）：

$$
\mathbf{s}_k^t = [\underbrace{\text{addr}_k}_{32} \| \underbrace{\text{cnt}_k^t}_{256} \| \underbrace{\pi^t}_{16} \| \underbrace{\rho_k}_{16}] \in \mathbb{R}^{320}
$$

各分量：
- $\text{addr}_k$：由语言标签 $\ell_k$ 和首帧视觉特征 $f_k^{(0)}$ 计算，**每轮任务冻结**，作为身份标识符
- $\text{cnt}_k^t$：每帧从 SAM 3 + DINOv3 掩码重新计算，携带时变状态
- $\pi^t$：正弦帧索引编码
- $\rho_k$：可学习角色 embedding（机器人槽、对象槽、padding 槽）

槽数量上限 $N_{\max} = 16$（含 1 个机器人槽 + 最多 15 个对象槽），不足时用掩码 padding。

#### 模块 2：对象可寻址注意力（Object-Addressable Attention）

**设计动机**: 标准 self-attention 的 key 读取完整 token，导致 content 变化影响路由。通过限制 key 只读 addr 子空间，使 routing 仅由稳定身份决定。

**具体实现**:

在槽 token 位置，第 $l$ 层注意力修改为：

$$
\mathbf{K}_k^{(l)} = W_K^{(l)} \cdot \text{mask}_{\leq 32}(\mathbf{x}_k^{(l)})
$$

其中 $\text{mask}_{\leq 32}$ 将第 33 维起全部清零，迫使 key 仅由 addr 子向量决定。Query 和 Value 仍读取完整 token：

$$
\mathbf{Q}_k^{(l)} = W_Q^{(l)} \mathbf{x}_k^{(l)}, \quad \mathbf{V}_k^{(l)} = W_V^{(l)} \mathbf{x}_k^{(l)}
$$

**逐层地址重置钩子**：每个 transformer block 后执行：

$$
\mathbf{x}_k^{(l+1)}[1:32] \leftarrow \text{addr}_k
$$

防止残差流对 addr 子空间的累积污染，同时允许 content/pose/语言信息通过 value projection 和残差传播。

注意力掩码为：跨帧 block-causal（时间因果），帧内槽之间双向（允许对象交互）。

#### 模块 3：预测头部

**世界头部（World Head）** $h_\psi$：通过并行 MLP 对每个对象槽预测下一帧状态：
- 内容分支：$4096 \to 1024 \to 256$（预测 $\hat{c}_k^{t+1}$）
- 姿态分支：$4096 \to 256 \to 9$（预测 $\hat{p}_k^{t+1}$，3×3 旋转矩阵）

**动作头部（Action Head）** $h_\xi$：[[Flow Matching]] MLP，在单次前向传播中用 4 步 Euler 积分完成 16 步动作块的去噪。

**辅助 VQ 头部**：复用 lm_head，加权交叉熵损失 $\mathcal{L}_{\text{vq}}$，辅助图像重建。

---

## 关键公式

### 公式 1：[[VLA|策略输出]] 定义

$$
(\mathbf{A}_t, \hat{\mathcal{S}}_{t+1}) \sim \pi_\theta(\mathbf{I}_{\leq t}, \mathbf{q}_{\leq t}, \ell, \mathbf{a}_{<t})
$$

**含义**: 策略同时输出动作块 $\mathbf{A}_t \in \mathbb{R}^{16 \times 7}$ 和下一帧每对象状态预测 $\hat{\mathcal{S}}_{t+1} = \{(\hat{c}_k^{t+1}, \hat{p}_k^{t+1})\}_{k=1}^N$

**符号说明**:
- $\mathbf{I}_{\leq t}$：到当前帧的所有 RGB 观测
- $\mathbf{q}_{\leq t}$：历史本体感知（关节角等）
- $\ell$：语言指令
- $\mathbf{a}_{<t}$：历史动作

### 公式 2：[[Slot Tokenization|槽向量]] 构成

$$
\mathbf{s}_k^t = [\text{addr}_k(32) \| \text{cnt}_k^t(256) \| \pi^t(16) \| \rho_k(16)] \in \mathbb{R}^{320}
$$

**含义**: 每个对象槽由冻结身份地址、时变内容、时间编码和角色编码拼接构成

**符号说明**:
- $\text{addr}_k = f_{\text{addr}}([\ell_k \| f_k^{(0)}])$：由语言 + 首帧特征计算，每轮冻结
- $\text{cnt}_k^t = f_{\text{cnt}}(\text{raw}_k^t)$：每帧从实例掩码重新计算
- $\pi^t$：正弦位置编码（帧索引）
- $\rho_k$：可学习角色 embedding

### 公式 3：[[Object-Addressable Attention|可寻址注意力]] Key 投影

$$
\mathbf{K}_k^{(l)} = W_K^{(l)} \cdot \text{mask}_{\leq 32}(\mathbf{x}_k^{(l)})
$$

**含义**: 通过截断掩码，强制 attention key 仅由前 32 维（addr 子空间）决定，实现 routing 与内容解耦

**符号说明**:
- $\text{mask}_{\leq 32}(\cdot)$：将 33 维起全部置零的截断操作
- $W_K^{(l)}$：第 $l$ 层 key 投影矩阵
- $\mathbf{x}_k^{(l)}$：第 $l$ 层槽 $k$ 的隐状态

### 公式 4：[[Object-Addressable Attention|逐层地址重置]]

$$
\mathbf{x}_k^{(l+1)}[1:32] \leftarrow \text{addr}_k
$$

**含义**: 每个 transformer block 输出后，强制将前 32 维恢复为原始 addr，防止残差流污染身份子空间

**符号说明**:
- $\mathbf{x}_k^{(l+1)}$：第 $l+1$ 层输入前的槽隐状态
- $\text{addr}_k$：该槽的冻结身份地址向量

### 公式 5：[[World Model|世界损失]]

$$
\mathcal{L}_{\text{world}} = \frac{1}{N} \sum_{k=1}^{N} m_k^{\text{obj}} \left( \|\hat{c}_k^{t+1} - c_k^{t+1}\|_2^2 + \lambda_p \|\hat{p}_k^{t+1} - p_k^{t+1}\|_2^2 \right)
$$

**含义**: 对每个对象槽预测下一帧的内容特征和姿态（机器人槽不参与损失计算）

**符号说明**:
- $m_k^{\text{obj}}$：对象掩码（仅对非机器人槽为 1）
- $\hat{c}_k^{t+1}$：预测的下一帧内容向量（256 维 DINOv3 特征）
- $c_k^{t+1}$：真实的下一帧内容向量
- $\hat{p}_k^{t+1}$：预测的下一帧姿态（9 维旋转矩阵）
- $\lambda_p$：姿态损失权重

### 公式 6：[[Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{\tau,\epsilon} \left\| \mathbf{v}_\xi(\mathbf{A}_t^\tau, \tau, \mathbf{H}_{\text{act\_q}}) - (\mathbf{A}_t - \epsilon) \right\|_2^2
$$

**含义**: 用流匹配训练动作解码器，学习从噪声空间到动作分布的速度场

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$：时间步（插值系数）
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau)\epsilon$：在噪声和真实动作之间插值
- $\mathbf{v}_\xi$：速度场 MLP
- $\mathbf{H}_{\text{act\_q}}$：动作查询 token 的隐状态

### 公式 7：[[Multi-Task Learning|总训练目标]]

$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{act}} + \lambda_w \mathcal{L}_{\text{world}} + \lambda_v \mathcal{L}_{\text{vq}} + \lambda_c \mathcal{L}_{\text{compose}} + \lambda_r \mathcal{L}_{\text{role}}
$$

**含义**: 多任务联合训练，动作预测为主任务，世界预测/图像重建/结构和角色辅助损失共同优化

**符号说明**:
- $\{\lambda_w, \lambda_v, \lambda_c, \lambda_r\} = \{0.5, 0.04, 0.1, 0.05\}$：各损失权重
- $\mathcal{L}_{\text{compose}}$：结构一致性损失（前 30% 训练步从 0 预热）
- $\mathcal{L}_{\text{role}}$：角色分类损失（训练后半段退火至 0）

---

## 关键图表

### Figure 1：系统概览

![Figure 1](https://arxiv.org/2605.06481v1/x1.png)

**说明**: 左侧展示六类典型场景扰动轴（相机、机器人初始状态、布局、光照、背景、传感器噪声）。上右：全局 token 的整体式 WAM 在扰动下 token 漂移导致策略崩溃。下右：OA-WAM 的可寻址槽通过固定 addr routing 保持鲁棒操作。

### Figure 2：OA-WAM 架构

![Figure 2](https://arxiv.org/2605.06481v1/x2.png)

**说明**: 多模态输入编码为六路 token 流，对象槽 token 经 [[SAM 3]] + [[DINOv3]] 提取后由可学习槽适配器投影。所有 token 组装成 block-causal 序列，末尾添加可学习动作查询 [ACT-Q]，经槽感知主干处理。世界头从槽隐状态预测下一帧 $(\hat{c}, \hat{p})$，动作头解码 16 步动作块。**仅槽适配器和新增头部引入可学习参数**，冻结 embed_tokens。

### Figure 3：OA 注意力掩码

![Figure 3](https://arxiv.org/2605.06481v1/x3.png)

**说明**: 注意力掩码设计。跨帧为 block-causal（时间因果），帧内槽之间双向（红色对角线）。$W_K$ 仅读取 $\text{addr}_k$ 的前 32 维，实现 routing 与 content 解耦。

### Figure 4：主实验结果

![Figure 4](https://arxiv.org/2605.06481v1/x4.png)

**说明**: 左：LIBERO-Plus 七轴扰动雷达图（Tab. 2），OA-WAM 在几何扰动轴（相机、机器人初始、布局）显著领先；右：SimplerEnv WidowX (Bridge) 各任务成功率，OA-WAM 达到 79.3% 均值。

### Figure 5：机制诊断

![Figure 5](https://arxiv.org/2605.06481v1/x5.png)

**说明**: (a) LP-camera 成功率 vs 相机旋转角 $\Delta\theta$：V0（完整 OA-WAM）和 V1（关闭 key mask）在分布内重合，分布外随 $\Delta\theta$ 增大显著分叉；(b) 角色查询注意力（r1-4 对应目标/参考/工具/干扰物）在 300 个 LIBERO-Spatial episode 上的均值；(c) 地址交换实验下末端执行器轨迹：OA-WAM 正确偏向交换后的目标，整体基线则不受影响。

### Table 1：LIBERO 和 SimplerEnv WidowX 主实验

| Method | Spatial | Object | Goal | Long | LIBERO Avg | SimplerEnv Avg |
|--------|---------|--------|------|------|-----------|-----------------|
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | — |
| InternVLA-M1 | 98.0 | 99.0 | 93.8 | 92.6 | 95.9 | 71.7 |
| CoWVLA | 97.2 | 97.8 | 94.6 | 92.8 | 95.6 | 76.0 |
| **OA-WAM** | **98.9** | **99.0** | **97.4** | **95.9** | **97.8** | **79.3** |

**关键发现**: OA-WAM 在所有 LIBERO 子任务和 SimplerEnv WidowX 均取得 SOTA。Long 任务（最难）成功率 95.9%，高出 π₀.₅ 3.5%。

### Table 2：LIBERO-Plus 鲁棒性（零样本）

| Method | Camera | Robot Init | Layout | Geo Avg | Light | BG | Language | Noise | Overall Avg |
|--------|--------|-------|--------|---------|-------|----|---------|----|------------|
| π₀.₅ | 75.4 | 77.5 | 85.7 | 79.5 | 96.9 | 94.6 | 85.6 | 89.7 | **85.7** |
| Cosmos-Policy | 75.8 | 63.3 | 82.2 | 73.8 | 96.5 | 88.9 | 81.7 | 92.7 | 82.2 |
| **OA-WAM** | **80.5** | **89.6** | 82.8 | **84.3** | 96.5 | **95.9** | 85.3 | 75.6 | 83.9 |

**关键发现**: OA-WAM 在几何扰动轴（Camera、Robot Init、Layout）的几何均值 84.3%，超越 π₀.₅ 的 79.5%（+4.8%）。传感器噪声轴弱势（75.6% vs 89.7%），原因是冻结分词器在噪声图像下实例分割失败。

### Table 3：OA 约束消融实验

| 变体 | K mask | Reset Hook | LIBERO Avg | LP Camera | LP Robot Init | LP Avg | SimplerEnv | Swap Binding |
|---------|--------|-----------|--------|-----------|----------|--------|-----------|--------------|
| V2（无 OA） | off | off | 95.4 | 60.5 | 64.8 | 76.2 | 56.7 | 0.06 |
| V1（仅 Reset） | off | on | 96.3 | 67.2 | 71.4 | 80.8 | 64.0 | 0.19 |
| **V0（完整 OA）** | **on** | **on** | **97.8** | **80.5** | **89.6** | **83.9** | **79.3** | **0.87** |

**关键发现**: 去掉 K mask（V0→V1）使 LP Camera 下降 13.3%，而 LIBERO 仅下降 1.5%，体现"分布外特异性"——在分布内差异小，OOD 下差异显著。

### Table 4：Swap-Binding 因果测试

| Method | Swap Binding ↑ |
|--------|-----------------|
| OpenVLA | 0.04 |
| π₀ | 0.05 |
| π₀.₅ | 0.05 |
| WorldVLA | 0.09 |
| VLA-JEPA | 0.07 |
| Cosmos-Policy | 0.06 |
| OA-WAM（无 mask） | 0.19 |
| OA-WAM（均值池化） | 0.18 |
| **OA-WAM（完整）** | **0.87** |

**关键发现**: 完整 OA-WAM 的 swap-binding 余弦相似度 0.87，比所有基线高 10 倍以上（最高基线 0.09）。验证策略决策确实通过地址空间路由到目标对象，而非记忆训练时布局。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 子集（Spatial/Object/Goal/Long） | 标准操作 benchmark | 训练 + 测试 |
| LIBERO-Plus | 7 扰动轴 | 零样本鲁棒性评测 | 纯测试（OOD） |
| SimplerEnv WidowX | Bridge 任务集 | 真实数据分布 | 跨分布泛化测试 |

### 实现细节

- **Backbone**: Chameleon 7B（32 层 Transformer，冻结主干）
- **可训练参数**: ~127M（80M LoRA on {q,k,v,o,gate,up,down}_proj + 47M 新增头部和适配器）
- **槽适配器 + 预测头**: 仅在 LIBERO 上训练；LIBERO-Plus 全程 hold-out
- **动作解码**: Flow Matching，推理时 4 步 Euler 积分，单次前向传播完成
- **感知延迟**: ~95 ms/帧（主干 + 头部 ~5.6 ms，感知管线为瓶颈）
- **评估**: 三随机种子均值，Appendix H 提供置信区间

### 可视化结果

- Figure 5(c) 展示：地址交换后 OA-WAM 末端执行器轨迹明显偏向新目标位置，整体基线轨迹保持不变，直觉上验证对象可寻址性的物理意义。
- Figure 5(b) 展示：角色查询（target/reference/tool/distractor）的注意力分布清晰分离，证明槽角色编码有效区分对象功能。

---

## 批判性思考

### 优点

1. **架构级保证**: 不同于依赖训练启发式的 slot attention 方法，key mask + reset hook 从架构上强制实现对象可寻址性，理论基础清晰
2. **因果可验证**: Swap-Binding 测试是方法论创新，提供对象绑定质量的量化因果指标，可推广到其他对象中心方法的评估
3. **轻量适配**: 冻结 7B 主干，仅 127M 可训练参数，效率高且继承预训练语言 + 视觉知识

### 局限性

1. **仿真验证局限**: 所有实验在模拟器中完成，现实机器人部署的鲁棒性未验证
2. **感知瓶颈**: 感知管线（SAM 3 + DINOv3）约 95 ms/帧，比主干 5.6 ms 慢 17 倍；传感器噪声下实例分割失败直接导致 -17.1% 性能下降
3. **冻结分词器的天花板**: 对小目标、反射/透明物体、遮挡场景的分割失败无法通过微调策略头部改善
4. **干扰物一致性假设**: 辅助损失假设目标和干扰物弱耦合，可能在强交互场景失效

### 潜在改进方向

1. 端到端可学习的分割模块，替代冻结 SAM 3，以提升传感器噪声下的鲁棒性
2. 将 addr/content 分解扩展到多帧历史聚合（目前每帧独立提取 content）
3. 真实机器人验证和 sim-to-real 迁移分析

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（损失权重、LoRA 配置、步数均在附录）
- [x] 数据集可获取（LIBERO、SimplerEnv、LIBERO-Plus 均公开）

---

## 关联笔记

### 基于

- [[Chameleon]]: 7B 多模态骨干网络
- [[SAM 3]]: 对象实例分割（用于提取槽）
- [[DINOv3]]: 视觉特征提取（用于 content 向量）
- [[Qwen3-VL]]: 语言名词短语提取（用于 addr 构建）
- [[Flow Matching]]: 动作解码器训练范式

### 对比

- [[Pi05|π₀.₅]]: 主要比较基线，OA-WAM 在几何鲁棒性上超越
- [[Cosmos-Policy]]: WAM 基线对比
- [[WorldVLA]]: 对象中心 WAM 对比，swap-binding 仅 0.09
- [[VLA-JEPA]]: 潜在空间预测 VLA 对比

### 方法相关

- [[Object-Addressable Attention]]: 本文核心技术
- [[Slot Tokenization]]: 对象槽 token 化设计
- [[Action Chunking]]: 16 步动作块输出
- [[World Action Model]]: 本文改进的范式
- [[VLA]]: 视觉-语言-动作模型范畴

### 数据集相关

- [[LIBERO]]: 训练 + 标准测试 benchmark
- [[SimplerEnv]]: 跨分布泛化测试

---

## 速查卡片

> [!summary] OA-WAM (2026)
> - **核心**: 对象可寻址性 —— 冻结 addr + addr-only key projection + 逐层 reset，使 attention routing 由稳定身份标识符决定
> - **方法**: Chameleon 7B 骨干 + SAM3/DINOv3 槽提取 + 可寻址注意力 + Flow Matching 动作头
> - **结果**: LIBERO 97.8%，SimplerEnv 79.3%，LIBERO-Plus 几何轴 84.3%（+4.8% vs π₀.₅），swap-binding 0.87（基线 ≤0.09）
> - **代码**: 未公开

---

*笔记创建时间: 2026-05-09*
