---
type: concept
aliases: [Video Diffusion, Video Diffusion Model, VDM]
---

# Video Diffusion

## 定义
Video Diffusion Model（VDM）是将扩散模型（Diffusion Model）扩展到视频序列生成的方法，通过在时序帧上联合建模空间与时间一致性，生成连贯的动态视觉内容。

## 核心要点
1. **时序建模**：在标准图像扩散基础上增加时间维度注意力（temporal attention）或 3D 卷积，保证帧间一致性
2. **两大范式**：扩散模型（逐步去噪）和自回归模型（逐帧/逐 token 预测）是视频生成的主流路径
3. **具身 AI 应用**：在机器人世界模型中作为视频预测器，预测动作执行后的未来观测

## 数学形式
$$x_0 \sim q(x_0),\quad x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

## 代表工作
- [[IRASim]]: 具身 AI 视频扩散模型
- [[RoboScape]]: 物理感知具身世界模型
- [[Vid2World]]: 视频扩散模型 → 交互式世界模型

## 相关概念
- [[世界模型]]
- [[Diffusion Policy]]
- [[IRASim]]
