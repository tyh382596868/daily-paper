---
type: concept
aliases: [Prediction Regret, 预测 regret, 隐空间 regret, Latent Regret]
---

# Prediction Regret

## 定义
Prediction Regret（预测 regret）是衡量一个（世界）模型在某条轨迹上预测误差大小的标量度量，常用预测与真实之间的误差范数定义，作为课程学习中"轨迹难度"或"学习潜力"的代理指标。

## 数学形式
隐空间预测 regret（RMS 形式）：
$$
\ell_{\mathrm{regret}}(\tau)=\sqrt{\frac{1}{HCN_{\mathrm{lat}}}\sum_{t=S}^{S+H-1}\left\|z_{t}^{\mathrm{pred}}-z_{t}^{\mathrm{real}}\right\|_{2}^{2}}
$$

## 核心要点
1. 直接对比模型预测与真实 rollout（ground-truth），区别于值函数估计的 regret。
2. 高 regret 轨迹被认为更难、信息量更大，优先纳入训练课程。
3. 其跨周期变化量 $\Delta\ell_{\mathrm{regret}}$ 可作为学习进度信号——下降说明已被学会。

## 代表工作
- [[PROWL]]: 用隐空间预测 regret 作为对抗策略奖励和 PAT buffer 排序的核心度量

## 相关概念
- [[PAT Buffer]]
- [[PLR]]
- [[World Model]]
- [[Action-Follow Score]]
