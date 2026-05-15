---
type: concept
aliases: [历史证据 Token, history-evidence token, History-Evidence Token]
---

# 历史证据 Token

## 定义

一个由历史编码器（如 [[VGGT]]）输出 pool 得到的**单 token 全局摘要**，作为额外条件 token 拼到主干上下文末尾，提供"显式的历史承诺"信号。

## 数学形式

$$
\bar{e}_t = \text{Pool}(U_t), \quad C_t = [F_t';\; e_t^{\text{tok}}]
$$

其中 $U_t$ 为投影后的历史 token 序列，$F_t'$ 为经[[门控交叉注意力]]融合过的当前 token。

## 核心要点

1. **粒度**: 与序列级注入互补——序列级让 "每个当前 token" 都能查历史，而 evidence token 提供 "一个全局视角"
2. **轻量**: 只增加 1 个 token 的上下文，几乎无推理开销
3. **可解释性较弱**: 但消融显示去掉它会掉点（[[IntentVLA]] 在 SimplerEnv 上 -3.4）
4. **与 [[CLS Token]] 类似**: 充当"汇总位"

## 代表工作

- [[IntentVLA]]: 完整方法 vs "history fusion 无 intent token" 消融：72.9% vs 69.5%

## 相关概念

- [[CLS Token]]
- [[短期意图表征]]
- [[门控交叉注意力]]
