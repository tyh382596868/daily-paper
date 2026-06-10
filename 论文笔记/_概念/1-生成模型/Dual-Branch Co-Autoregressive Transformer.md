---
type: concept
aliases: [DCT, 双分支协同自回归变换器]
---

# Dual-Branch Co-Autoregressive Transformer (DCT)

## 定义
RoboScape 提出的双分支协同自回归 Transformer 架构，两个分支分别处理 RGB 视频和深度图，通过跨分支特征注入实现深度物理先验对 RGB 生成的增强。

## 数学形式

$$
h_{RGB}^{l+1} = \text{STBlock}_{RGB}(h_{RGB}^{l} + \phi(h_{depth}^{l}))
$$

$$
h_{depth}^{l+1} = \text{STBlock}_{depth}(h_{depth}^{l})
$$

## 核心要点
1. **双分支结构**: RGB 分支和深度分支各自由堆叠的 ST-Transformer 块组成，独立处理各自模态
2. **跨分支注入**: 深度分支特征经线性投影后注入 RGB 分支，使 RGB 生成具备 3D 空间意识
3. **时间因果注意力**: 时间维度使用因果自注意力，保证生成的时序因果性（不看未来帧）
4. **空间双向注意力**: 空间维度使用双向注意力，实现帧内全局上下文建模

## 代表工作
- [[RoboScape]]: DCT 是 RoboScape 的核心生成架构

## 相关概念
- [[Spatial-Temporal Transformer]]
- [[因果自注意力]]
- [[双向注意力]]
- [[MAGVIT-2]]
