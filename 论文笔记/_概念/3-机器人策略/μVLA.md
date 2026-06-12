---
type: concept
aliases: [muVLA, micro VLA, Recurrent Memory VLA]
---

# μVLA (Recurrent Memory VLA)

## 定义
在 VLA 中引入循环记忆（recurrent state）处理部分可观测（POMDP）操作场景，基于 OpenVLA + TBPTT 训练。

## 数学形式
$$h_t = f_\text{RNN}(h_{t-1}, o_t, a_{t-1}), \quad a_t = f_\text{policy}(h_t)$$

## 核心要点
1. 基于 OpenVLA 扩展，加入 TBPTT 训练循环状态
2. EMA 做状态平滑
3. 针对遮挡/历史依赖的 POMDP 操作场景
4. 仅仿真验证（has_real_world=False）

## 代表工作
- [[μVLA]]: arXiv 2606.12497

## 相关概念
- [[OpenVLA]]
- [[VLA]]
