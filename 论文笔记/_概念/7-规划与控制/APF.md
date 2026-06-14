---
type: concept
aliases: [Artificial Potential Field, 人工势场法, 势场规划]
---

# APF（人工势场法）

## 定义
APF（Artificial Potential Field）是一种经典的实时运动规划方法，将机器人所处环境建模为势场：目标点产生吸引力，障碍物产生排斥力，机器人沿负梯度方向运动。

## 数学形式
总势场：
$$U(\mathbf{q}) = U_{\text{att}}(\mathbf{q}) + U_{\text{rep}}(\mathbf{q})$$

吸引势（抛物线型）：
$$U_{\text{att}}(\mathbf{q}) = \frac{1}{2} k_{\text{att}} \|\mathbf{q} - \mathbf{q}_{\text{goal}}\|^2$$

排斥势（距离倒数型）：
$$U_{\text{rep}}(\mathbf{q}) = \begin{cases} \frac{1}{2} k_{\text{rep}}\left(\frac{1}{\rho} - \frac{1}{\rho_0}\right)^2 & \rho \leq \rho_0 \\ 0 & \rho > \rho_0 \end{cases}$$

生成力：$\mathbf{F}(\mathbf{q}) = -\nabla U(\mathbf{q})$

## 核心要点
1. 计算代价低，适合实时规划（可在 GPU 上并行化）
2. 致命缺陷：局部极小值问题——势场极小值不一定是全局目标
3. 在凹形障碍物附近容易陷入振荡
4. G-MAPP 等工作用 CUDA 并行化 APF 实现多臂实时碰撞避免

## 代表工作
- Khatib (1986): 原始 APF 论文
- [[G-MAPP]]: GPU 并行 APF for 多机械臂

## 相关概念
- [[MPC]]
- [[RRT-Connect]]
- [[MPPI]]
