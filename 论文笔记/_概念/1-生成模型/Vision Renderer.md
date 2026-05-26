---
type: concept
aliases: [Vision Renderer, 视觉渲染器]
---

# Vision Renderer

## 定义

Vision Renderer 是 [[X-Foresight]] 的第二阶段模块：基于 [[X-World]]（[[DiT]] + [[3D Causal VAE]]）+ [[Rectified Flow]]，把 LDM 预测的低维 camera token 解码为 7 路环视的**照片级未来视频**。设计上与 [[Large Drive Model|LDM]] 解耦，仅 condition 于 camera token、不接受动作输入（避免渲染器通过动作做 shortcut）。

## 核心要点

1. **解耦输入**: 仅看 camera token，**不**看动作——强迫 LDM 通过 token 真的承载视觉语义
2. **[[Cross-View Attention|跨视角注意力]]**: 沿时间轴与相机轴交替施加，保证 7 相机几何 / 物体 ID / 运动一致
3. **Drift 缓解**:
   - **Latent Sink**: 锚定稳定参考上下文 across rollout 步
   - **Latent Augmentation**: 训练时对当前-步 latent 加扰动，使分布贴近推理期
4. **三阶段训练**:
   - Stage II 先在 ground-truth 动作上预训练
   - Stage III 再在 LDM 预测的 token 上微调（LDM 冻结），适应不完美 token
5. **质量飞跃**: 在 1 s 视野下 FID 1.51 / FVD 11.28（vs Camera Latent Decoder baseline 10.97 / 135.56，~85% / 92% 相对改进）

## 代表工作

- [[X-Foresight]]: 首次提出，作为 LDM 的下游照片级渲染模块
- [[X-World]]: 作为 backbone

## 相关概念

- [[X-World]]
- [[Rectified Flow]]
- [[DiT]]
- [[3D Causal VAE]]
- [[Cross-View Attention]]
- [[Large Drive Model]]
