---
type: concept
aliases: [情境算子学习, In-Context Operator Learning, ICON]
---

# In-Context Operator Learning（情境算子学习）

## 定义

情境算子学习（In-Context Operator Learning, ICON）是一种将函数映射关系的学习建模为情境学习（in-context learning）问题的框架：通过在推理时提供少量输入-输出对作为"情境提示"，网络无需参数更新即可适应新的映射函数。

## 数学形式

$$
\hat{y} = f_\theta\!\left(x \mid \{(x_i, y_i)\}_{i=1}^K\right)
$$

其中 $\{(x_i, y_i)\}$ 为检索到的 $K$ 个情境样本，$f_\theta$ 的参数在推理时固定不变。

## 核心要点

1. **函数学习视角**: 将每次推理视为对一个新算子（函数）的学习，情境样本定义该算子
2. **无参数更新**: 推理时完全依赖情境提示，不执行梯度更新
3. **跨任务泛化**: 只要新任务与训练时见过的算子族属于同一类，即可通过情境样本适应
4. **原始应用**: 最早用于偏微分方程求解器学习（ICON, VICON），后扩展到机器人执行

## 代表工作

- [[ICON]]: 首次提出用情境学习框架学习算子（用于 PDE 求解）
- [[VICON]]: 将 ICON 扩展到视觉观测的多物理流体动力学预测
- [[V2T-ICON]] / [[VICX]]: 将情境算子学习应用于机器人视频-状态映射

## 相关概念

- [[In-Context Learning]]
- [[In-Context Retrieval]]
- [[Transformer]]
