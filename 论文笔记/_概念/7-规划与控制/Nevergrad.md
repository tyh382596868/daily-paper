---
type: concept
aliases: [NG, Nevergrad optimizer]
---

# Nevergrad

## 定义
Meta 开源的**梯度无关（gradient-free）**优化库，集成 CMA、DE、Bayesian 等多种黑盒优化算法，并内置自适应选择策略，**几乎不需要超参调节**。

## 核心要点
1. **零超参优势**：自适应选择内部算法，对新任务"开箱即用"
2. **梯度无关**：适合不可微目标函数（如离散动作、规划代价、神经架构搜索）
3. **典型用法**:
   ```python
   import nevergrad as ng
   optimizer = ng.optimizers.NGOpt(parametrization=dim, budget=100)
   recommendation = optimizer.minimize(loss_fn)
   ```
4. **在世界模型规划中的地位**:
   - [[JEPA-WM]] 论文测试发现 NG 在真实数据上与 [[CEM]] 持平
   - 优点是免调参，缺点是收敛速度比 [[CEM]] 慢、在 2D 导航上探索过多

## 代表工作
- [[JEPA-WM]]: 在 [[DROID]]、[[RoboCasa]] 上验证 NG 与 [[CEM]] 相当

## 相关概念
- [[CEM]]
- [[MPC]]
