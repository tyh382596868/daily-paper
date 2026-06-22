---
type: concept
aliases: [Flow Matching, 流匹配, Rectified Flow, Continuous Normalizing Flow]
---

# Flow Matching

## 定义
Flow Matching 是一类基于连续归一化流（CNF）的生成模型训练方法，通过回归向量场来学习从噪声分布到数据分布的概率路径，无需模拟 ODE 即可高效训练。

## 数学形式
设 $p_0$ 为噪声分布，$p_1$ 为数据分布，条件概率路径 $p_t(x|x_1)$ 定义为从 $x_0 \sim p_0$ 到 $x_1 \sim p_1$ 的插值：

$$x_t = (1-t)x_0 + t x_1, \quad t \in [0,1]$$

训练目标为回归条件向量场：

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_1, x_0} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2$$

推理时沿 $\dot{x}_t = v_\theta(x_t, t)$ 求解 ODE。

## 核心要点
1. 比 DDPM 更直接的训练目标，向量场可直接从数据对 $(x_0, x_1)$ 构造
2. 线性插值路径（Rectified Flow）产生接近直线的 ODE 轨迹，步数更少
3. 可扩展到一步生成（consistency distillation、DMD）
4. 在机器人动作生成中广泛应用（Diffusion Policy 的 flow-matching 变体）

## 代表工作
- [[Diffusion Policy]]: 将 flow matching 用于连续机器人动作生成
- [[EquiVLA]]: EquiActor 基于 flow-matching action head
- [[ManiFlow]]: INN adapter 加速一步 flow matching

## 相关概念
- [[Diffusion Model]]
- [[Consistency Models]]
- [[DMD]]
- [[DDPM]]
