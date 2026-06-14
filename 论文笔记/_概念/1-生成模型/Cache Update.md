---
type: concept
aliases: [Cache Update, 缓存更新, 记忆更新, Memory Update]
---

# Cache Update

## 定义

Cache Update（缓存更新）是视频世界模型空间记忆流水线中的第三阶段：在生成新一段视频帧后，将新观测到的场景信息反投影并并入持久化 3D 缓存，从而扩展已知场景的覆盖范围。

## 数学形式

以 [[Mirage]] 的 [[Latent Spatial Memory]] 为例，Cache Update 公式为：

$$
\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{p}_{uv}, \mathbf{F}_{uv})\}_{(u,v) \in \Lambda^t}
$$

**符号说明**:
- $\mathcal{M}$: 全局 3D 空间记忆（点特征集合）
- $\Lambda^t$: 第 $t$ 帧中满足过滤条件的有效 latent 格点集合
- $\mathbf{p}_{uv}$: 由深度估计反投影得到的 3D 世界坐标
- $\mathbf{F}_{uv}$: 重新编码生成帧后提取的 clean latent 特征

## 核心要点

1. **三阶段流水线的最终步骤**：初始化 → 读出（Readout）→ **Cache Update**，完成一次完整的生成-记忆循环
2. **动态物体过滤**：更新前需过滤掉运动物体（人、车等）和天空区域，避免非静态几何污染 cache
   - 工具：开放词汇实体提取器 + 视频分割模型（如 SAM 3）
3. **clean latent 重编码**：更新时使用重新编码的"干净" latent（非扩散噪声中间状态），保证特征质量
4. **渐进式 cache 增长**：每次更新添加新视角下的未知区域点，已有点不覆盖（保留历史观测）
5. **RGB 版本的对应操作**：RGB 点云的 cache update 原理相同，但存储 RGB 颜色而非 latent 特征

## 工作流

```
生成第 t 帧视频
    ↓
重新编码生成帧 → clean VAE latents
    ↓
动态物体分割 → 过滤掉运动实体 + 天空
    ↓
深度估计 → 有效 latent 格点反投影到 3D
    ↓
并入全局 cache: M ← M ∪ {新点}
```

## 代表工作

- [[Mirage]]: 在 latent 空间实现 Cache Update，避免 RGB 往返编码开销
- [[Voyager]]: RGB 点云版本的 cache update
- [[Spatia]]: 类似的空间记忆更新机制

## 相关概念

- [[Latent Spatial Memory]]: Mirage 中 cache 存储的数据结构
- [[深度引导反投影]]: 将 2D 像素/latent cell 提升到 3D 的技术
- [[RGB 点云]]: cache 存储 RGB 颜色的替代方案
- [[z-buffer]]: cache 读出（Readout）阶段的深度测试机制
