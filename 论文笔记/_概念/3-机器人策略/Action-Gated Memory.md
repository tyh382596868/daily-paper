---
type: concept
aliases: [动作门控记忆, AURA 记忆]
---

# Action-Gated Memory

## 定义
以动作变化量为门控信号，选择性更新机器人策略的观测记忆，实现 O(1) 恒定 VRAM 占用。

## 数学形式
$$g_t = \mathbf{1}[\|a_t - a_{t-1}\| > \epsilon], \quad m_t = g_t \cdot \text{Update}(m_{t-1}, o_t) + (1-g_t) \cdot m_{t-1}$$

## 核心要点
1. 动作未发生显著变化时不更新记忆状态
2. 记忆大小恒定，不随 horizon 增长
3. 与标准 KV cache 相比显著降低长任务 VRAM 占用

## 代表工作
- [[AURA]]: 提出 Action-Gated Memory 机制

## 相关概念
