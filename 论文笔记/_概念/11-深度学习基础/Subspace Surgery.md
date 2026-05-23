---
type: concept
aliases: [Subspace Surgery, 子空间手术, Rowspace Surgery]
---

# Subspace Surgery

## 定义

[[TRM]] 论文用的一种机制实验工具：用线性 probe 的行空间矩阵 $W$ 构造投影算子 $P_W$，把 latent 投到「任务相关子空间」或其残差，然后**只在该子空间内计算 MSE**，验证任务信号是否集中在小子空间里。

## 数学形式

投影算子（[[Moore-Penrose 伪逆]]）:
$$
P_W = W^\top (W W^\top)^\dagger W
$$

子空间 MSE:
$$
c_{\text{sub}}(z) = \| P_W (\hat z_{t+H} - z_g) \|_2^2
$$

残差 MSE:
$$
c_{\text{res}}(z) = \| (I - P_W) (\hat z_{t+H} - z_g) \|_2^2
$$

## 核心要点

1. **诊断问题**: 任务信号是「不在 latent 里」还是「在 latent 里但被欧氏距离淹没」？
2. **[[TRM]] TwoRoom 结果 (Table 8)**:
   - Raw latent MSE: 1.7% 成功率
   - **XY-rowspace latent MSE: 90.8%** （只用 <1% 的子空间）
   - Residual-only latent MSE: 1.7% （占 MSE 99% 的部分几乎没信息）
3. **结论**: 信号在 latent 里，但被欧氏距离的均匀加权埋掉了——这是「学习型代价」存在的合法性证据
4. **应用边界**: 需要先训一个线性 probe 找到任务相关子空间；当任务变量不能线性解码时此技术失效

## 代表工作

- [[TRM]]: 用 XY 位置 probe 在 LeWM/PLDM 上做子空间手术

## 相关概念

- [[Terminal Proximity Cost]]
- [[Latent MPC]]
- [[TRM]]
- [[线性探测]]
