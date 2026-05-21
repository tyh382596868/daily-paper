---
type: concept
aliases: [PAT Buffer, Prioritized Adversarial Trajectory Buffer, 优先级对抗轨迹缓冲]
---

# PAT Buffer

## 定义
PAT Buffer（Prioritized Adversarial Trajectory buffer，优先级对抗轨迹缓冲）是一种存放对抗发现的失败轨迹的回放缓冲，按预测误差、动作保真度和学习进度对轨迹重排序，把发现的失败转化为渐进变难的训练课程。

## 数学形式
轨迹优先级（综合得分）：
$$
\mathrm{score}(\tau)=z_{\mathcal{B}}(\ell_{\mathrm{regret}})+\lambda_{\mathrm{AFS}}\,z_{\mathcal{B}}(\ell_{\mathrm{AFS}})+\Delta\ell_{\mathrm{regret}}
$$
其中 $z_{\mathcal{B}}(\cdot)$ 为缓冲内 z-score 归一化，$\Delta\ell_{\mathrm{regret}}$ 为学习进度项。

## 核心要点
1. 容量有限（如 $K=256$），超出时淘汰最低优先级条目。
2. 每次学习器（世界模型）更新后对缓冲内全部轨迹重新打分（rescore）——已解决的轨迹失去优先级，停滞/回退的保持高优先级。
3. 课程始终聚焦在尚未解决的失败模式上，与被动数据按比例 $r$ 混合用于训练。

## 代表工作
- [[PROWL]]: 提出 PAT buffer，把对抗策略发现的世界模型失败轨迹转化为优先级课程

## 相关概念
- [[PLR]]
- [[KL 约束对抗课程]]
- [[Prediction Regret]]
- [[Action-Follow Score]]
