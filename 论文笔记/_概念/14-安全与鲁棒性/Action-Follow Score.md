---
type: concept
aliases: [Action-Follow Score, AFS, AFS-EPE, 动作跟随得分]
---

# Action-Follow Score

## 定义
Action-Follow Score（AFS，动作跟随得分）是评估视频世界模型"是否正确响应动作"的度量：在解码后的像素帧上比较预测运动与真实运动（通常用[[光流]]）的偏差，专门捕捉外观指标（LPIPS/SSIM）看不见的运动失败。

## 数学形式
$$
\ell_{\mathrm{AFS}}(\tau)=\frac{1}{N\cdot H_{\mathrm{raw}}}\sum_{t=0}^{N-1}\sum_{i=1}^{H_{\mathrm{raw}}}\left\|F_{t,i}^{\mathrm{pred}}-F_{t,i}^{\mathrm{real}}\right\|_{2}
$$
其中 $F^{\mathrm{pred}}, F^{\mathrm{real}}$ 为预测/真实光流向量（EPE，端点误差）。

## 核心要点
1. 基于光流的端点误差（EPE），衡量动作条件动力学的保真度。
2. 弥补 LPIPS/SSIM 等外观指标的盲区——这些指标对短时程运动失败"视而不见"。
3. 在动作密集帧上能显示 2–3 倍于外观指标的方法差距。

## 代表工作
- [[PROWL]]: 提出 AFS-EPE 作为评估指标并纳入 PAT buffer 优先级得分，揭示外观指标的运动盲区

## 相关概念
- [[光流]]
- [[SEA-RAFT]]
- [[Prediction Regret]]
- [[World Model]]
