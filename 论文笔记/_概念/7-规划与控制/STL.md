---
type: concept
aliases: [Signal Temporal Logic, 信号时序逻辑]
---

# STL（Signal Temporal Logic）

## 定义
一种用于描述连续时间信号时序属性的形式化逻辑语言，可表达机器人行为的时间约束（如"先到达 A 区域，然后在 5 秒内避开障碍物"）。

## 数学形式
$$\phi ::= \mu \mid \neg\phi \mid \phi_1 \wedge \phi_2 \mid \phi_1 \mathcal{U}_{[a,b]} \phi_2 \mid \mathbf{F}_{[a,b]}\phi \mid \mathbf{G}_{[a,b]}\phi$$

其中 $\mathbf{F}_{[a,b]}$（最终）、$\mathbf{G}_{[a,b]}$（始终）、$\mathcal{U}_{[a,b]}$（直到）为时序算子，$[a,b]$ 为时间区间。

## 核心要点
1. 支持量化语义（robustness degree），不只是布尔满足与否，还能衡量"满足程度"
2. 可组合表达复杂时序约束，适合安全关键的 human-in-the-loop 场景
3. 与世界模型结合可用于推理时轨迹制导：采样动作序列→世界模型预测状态→检查 STL 满足度→重采样
4. 规范定义繁琐，实际部署中需要结合领域知识编写

## 代表工作
- Maler & Nickovic (2004)：STL 原始定义
- [[STL-WM]]：将 STL 与扩散策略 + 世界模型结合做推理时制导

## 相关概念
- [[MPPI]]
- [[Diffusion Policy]]
- [[World Model]]
