---
type: concept
aliases: [Pearson Correlation Coefficient, 皮尔逊相关系数, r值]
---

# Pearson 相关系数

## 定义

Pearson 相关系数（Pearson Correlation Coefficient）衡量两个连续变量之间线性相关程度，取值范围 $[-1, 1]$，在机器人学习中常用于评估世界模型（虚拟评估器）预测与真实世界表现之间的一致性。

## 数学形式

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

## 核心要点

1. **取值解读**: $r \to 1$ 表示强正相关，$r \to 0$ 表示无线性相关，$r \to -1$ 表示强负相关
2. **世界模型评估**: 在策略评估场景中，用 $r$ 衡量"世界模型中策略成功率"与"真实世界策略成功率"的一致性，$r$ 越高说明世界模型越可作为真实环境的代理
3. **统计显著性**: 通常同时报告 p 值，$p < 0.05$ 表示相关性统计显著
4. **局限性**: 只捕捉线性关系，对非线性相关和离群点敏感

## 代表工作

- [[MemWorld]]: 用 Pearson 相关系数评估世界模型策略评估器质量，Mem-World 达到 r=0.97（vs Ctrl-World r=0.85）

## 相关概念

- [[世界模型]]
- [[11-深度学习基础]]
