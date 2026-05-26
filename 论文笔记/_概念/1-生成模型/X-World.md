---
type: concept
aliases: [X-World, XPeng World Model]
---

# X-World

## 定义

X-World 是 XPeng（小鹏）PWM Team 提出的视频世界模型，结合 [[DiT]] backbone + [[3D Causal VAE]] latent space + [[Rectified Flow]] 训练目标，能够生成多相机环视驾驶视频。被 [[X-Foresight]] 作为 [[Vision Renderer]] 的 backbone 使用。

## 核心要点

1. **DiT backbone**: Transformer-based 扩散，处理时空 latent token
2. **3D Causal VAE latent**: 在 8×8 空间 + 4× 时间下采样后的 latent 上做扩散
3. **Rectified Flow 目标**: velocity matching，沿直线插值
4. **多相机环视**: 支持 7 路相机（front fisheye, front narrow, side ×4, rear）联合生成
5. **作为 backbone 复用**: X-Foresight 直接把它接到 LDM 输出端做照片级渲染

## 代表工作

- X-World（XPeng PWM Team）
- [[X-Foresight]]: 直接复用作 Vision Renderer backbone

## 相关概念

- [[DiT]]
- [[3D Causal VAE]]
- [[Rectified Flow]]
- [[Vision Renderer]]
- [[视频扩散模型]]
