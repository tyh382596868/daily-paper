---
type: concept
aliases: [MolmoAct2, MolmoAct]
---

# MolmoAct2

## 定义

基于 [[Molmo2]] 视觉-语言主干扩展的 [[VLA]] 模型族，把 Molmo 的高质量空间定位与指代能力迁移到机器人动作生成，常作为 VLA benchmark 的强基线。

## 核心要点

1. **主干**：以 [[Molmo2]] 多模态模型为视觉-语言编码器
2. **动作头**：采用 chunk-policy + 流匹配 / 自回归动作 token 解码
3. **典型用法**：作为 [[LIBERO]] / [[RoboCasa]] / 自建 VLA benchmark 的 baseline，体现"无 history 的强 frame-only 策略"性能上限
4. **变体**：[[MolmoAct2-Think]] 引入显式 CoT 中间推理

## 代表工作

- [[MolmoAct2-Think]]：思考链增强版本
- 被 [[IntentVLA]] 等历史条件化方法用作 frame-only 对比基线

## 相关概念

- [[Molmo2]]
- [[VLA]]
- [[Action Chunking]]
- [[MolmoAct2-Think]]
