---
type: concept
aliases: [SAM3, Segment Anything Model 3, SAM v3]
---

# SAM 3

## 定义

SAM 3（Segment Anything Model 3）是 Meta 发布的第三代通用图像分割模型，支持文本、点、框等多种提示模式，具备更强的实例分割和视频跟踪能力，适用于机器人感知中的对象实例提取。

## 核心要点

1. **多模态提示**: 支持文本描述、坐标点、边界框等多种提示形式进行实例分割
2. **零样本泛化**: 无需针对新物体类别重新训练，具备强泛化能力
3. **视频一致性**: 支持跨帧实例追踪，为机器人连续操作提供稳定的对象掩码序列
4. **在 OA-WAM 中的角色**: 用文本名词短语（来自 Qwen3-VL）作为提示，分割每帧中的对象实例，提供 DINOv3 特征提取的掩码区域

## 代表工作

- [[OA-WAM]]: SAM 3 负责对象实例分割，输出掩码后由 DINOv3 提取 content 特征，构建槽 token

## 相关概念

- [[DINOv3]]
- [[Qwen3-VL]]
- [[Slot Tokenization]]
- [[VLA]]
