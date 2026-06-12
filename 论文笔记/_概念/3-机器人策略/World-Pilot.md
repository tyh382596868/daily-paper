---
type: concept
aliases: [World Pilot, WAP, World-Action Prior]
---

# World-Pilot (世界动作先验引导 VLA)

## 定义
中科院自动化所提出的轻量世界动作先验（WAP）模块，在 VLA action decoder 前插入 future state 语义预见，引导动作生成而不做完整视频生成。

## 数学形式
$$a_t = f_\text{VLA}(o_t, l, z_\text{WAP}), \quad z_\text{WAP} = f_\text{compact}(o_t)$$

## 核心要点
1. 不做完整视频生成，只提取未来状态的语义压缩表示
2. 计算 overhead 比完整 WAM 小
3. LIBERO + RoboCasa OOD 泛化验证

## 代表工作
- [[World-Pilot]]: arXiv 2606.12403

## 相关概念
- [[WAM]]
- [[VLA]]
- [[LIBERO]]
