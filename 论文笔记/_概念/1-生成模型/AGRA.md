---
type: concept
aliases: [Action-Grounded Representation Alignment]
---

# AGRA (Action-Grounded Representation Alignment)

## 定义
将 WAM 中的表示对齐损失重用为动作引导信号，使预见到的 future state 表示直接 inform action head。

## 数学形式
$$a_t = f_\text{policy}(o_t, z_\text{align}), \quad z_\text{align} = \text{SigLIP-align}(o_t, \hat{o}_{t+1})$$

## 核心要点
1. 不重新设计架构，重用 alignment loss 副产品
2. 用 SigLIP 做表示对齐，DiT 生成 future frame
3. 来自 HKU + XPENG Robotics

## 代表工作
- [[AGRA]]: arXiv 2606.12217

## 相关概念
- [[WAM]]
- [[SigLIP]]
- [[DiT]]
