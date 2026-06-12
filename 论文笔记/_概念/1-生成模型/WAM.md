---
type: concept
aliases: [World Action Model, 世界动作模型]
---

# WAM (World Action Model)

## 定义
将视频生成能力与机器人动作预测结合的模型范式，通过预测未来视觉帧来引导动作决策。

## 数学形式
$$a_t = f_\text{policy}(z_t), \quad z_t = f_\text{WM}(o_{\leq t}, a_{<t})$$

## 核心要点
1. 用视频生成模型（DiT/扩散模型）预测 future frame
2. 从预测帧中提取隐表示引导动作头
3. 可分为"重建导向"（视频质量优先）和"表示导向"（动作对齐优先）两类
4. 代表工作：Genie、UniSim、IRASim、RepWAM

## 代表工作
- [[RepWAM]]: 表示中心 WAM，用 RepViTok 联合优化视觉保真度和表示一致性
- [[AGRA]]: 把 WAM 的对齐损失重用来引导动作
- [[World-Pilot]]: 轻量 WAP（世界动作先验）引导 VLA

## 相关概念
- [[WEAVER]]
- [[Diffusion Policy]]
- [[IDM]]
