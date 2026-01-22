# ResTacVLA + VQVAE 快速参考

## 核心修复（一句话总结）

**从 `action = noisy_actions` 改为 `action = state_t - state_t-1`**

这使得 Unit-Align VQVAE 获得真实的历史动作上下文，而非模型要预测的目标动作。

---

## 关键改动速查

### 1. 数据流
```
数据源: LeRobot (state_t, state_t-1, 图像, 触觉)
   ↓
处理: action_prev = state_t - state_t-1
   ↓
模型: encode_tactile(tactile, visual, action_prev)  ← 正确！
   ↓
输出: (vq_features, logvar)
```

### 2. 文件改动总结

```python
# model.py
Observation.action_prev: Optional[Array]  # 新增字段

# data_loader.py
if has_vqvae:
    delta_timestamps_dict["state_prev"] = [-1 / fps]  # 加载历史状态

# restac_policy.py
action_prev = state - state_prev  # 计算前一动作
inputs["action_prev"] = action_prev  # 传递给模型

# pi0_restac.py
action_prev = obs.action_prev  # 从 observation 获取
tactile_features, gate_value, logvar = encode_tactile(
    tactile_image,
    visual_image,
    action_prev  # ← 使用前一动作
)
```

### 3. 配置

```python
# 启用 VQVAE
config = Pi0_ResTacConfig(
    residual_vqvae_checkpoint="/path/to/vqvae.ckpt"  # 关键!
)

# 自动会:
# 1. 加载 VQVAE
# 2. 数据加载器请求 state_prev
# 3. 计算 action_prev
# 4. 使用 logvar 加权 loss
```

---

## 验证清单

运行以下检查确保集成正确：

```bash
# 1. 语法检查
python -m py_compile src/openpi/models/pi0_restac.py
python -m py_compile src/openpi/models/model.py
python -m py_compile src/openpi/training/data_loader.py
python -m py_compile src/openpi/policies/restac_policy.py

# 2. 导入检查
python -c "from openpi.models.model import Observation; print(Observation.__annotations__)"
# 应该包含 'action_prev'

# 3. 配置创建
python -c "from openpi.training.config import Pi0_ResTacConfig; config = Pi0_ResTacConfig(residual_vqvae_checkpoint=None); print('OK')"
```

---

## Logvar 含义速记

| Logvar 值 | 含义 | 操作 |
|---------|------|------|
| 高 (e.g., 1.0) | 低确定性 | sparse loss 权重大 |
| 低 (e.g., -3.0) | 高确定性 | sparse loss 权重小 |

**公式**: `loss = flow_loss + weight * sparse_loss * exp(logvar)`

---

## 常见问题速答

| Q | A |
|---|---|
| action_prev 是什么? | state_t - state_t-1，前一时刻执行的动作 |
| 为什么不用 noisy_actions? | 那些是模型要预测的输出，不是输入上下文 |
| state_prev 从哪来? | LeRobot 通过 delta_timestamps 提供 |
| action_prev 如果没有? | 自动默认为零向量 |
| logvar 怎么用? | 加权 sparse loss: loss * exp(logvar) |
| 影响现有代码吗? | 不影响，完全向后兼容 |

---

## 测试 action_prev 计算

```python
import numpy as np

# 测试 action_prev 计算
state_t = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])  # 当前状态
state_prev = np.array([0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])  # 前一状态

action_prev = state_t - state_prev
print(f"action_prev: {action_prev}")  # [0.1, 0, 0, 0, 0, 0, 0]
# ✓ 机器人沿 X 轴移动了 0.1 单位

# 如果没有前一状态
action_prev_zero = np.zeros_like(state_t)
print(f"action_prev (no history): {action_prev_zero}")  # [0, 0, ...]
# ✓ 表示静止状态
```

---

## 调试技巧

### 检查 action_prev 是否被正确加载

```python
# 在数据加载中添加调试
from openpi.training.data_loader import create_data_loader

data_loader = create_data_loader(config)

for obs, actions in data_loader:
    print(f"Has action_prev: {obs.action_prev is not None}")
    if obs.action_prev is not None:
        print(f"action_prev shape: {obs.action_prev.shape}")
        print(f"action_prev values: {obs.action_prev[0]}")
    break
```

### 检查 logvar 值

```python
# 在训练中监控 logvar
logvar = outputs['logvar']
print(f"logvar mean: {logvar.mean()}")  # 应该在 -2 到 2 之间
print(f"logvar std: {logvar.std()}")
print(f"exp(logvar) mean: {jnp.exp(logvar).mean()}")  # 通常 0.1 - 10
```

---

## 关键概念图

```
Unit-Align VQVAE 工作流
========================

输入:
├─ tactile_image: 当前触觉观测
├─ visual_image: 当前视觉观测
└─ action_prev: 前一时刻执行的动作 ← 关键!

Prophet (预期触觉预测):
├─ 输入: (visual_image, action_prev)
├─ 输出: (mu, logvar)
└─ logvar = 不确定性

Observation (真实触觉):
├─ 输入: tactile_image
├─ 输出: z_real

残差 VQ 编码:
├─ 输入: (z_real - z_exp)
├─ 输出: q_event (VQ codes)
└─ q_event = 离散事件表示

输出给 ResTacVLA:
├─ vq_features (or q_event)
├─ logvar (不确定性)
└─ → 两阶段融合 → 动作修正
```

---

## 文档导航

| 文档 | 用途 |
|-----|------|
| **本文** | 快速参考（这里！） |
| CHANGES_SUMMARY.md | 完整改动列表 |
| ACTION_PREV_INTEGRATION.md | action_prev 细节 |
| README_RESTAC_CHANGES.md | 修改总结、使用指南 |
| VQVAE_INTEGRATION_GUIDE.md | VQVAE forward 实现 |
| RESTAC_IMPLEMENTATION_SUMMARY.md | 架构总结 |

---

## 一行总结各文件改动

```
model.py          → 添加 Observation.action_prev 字段
data_loader.py    → 条件加载 state_prev (仅 VQVAE 模式)
restac_policy.py  → 计算并传递 action_prev
pi0_restac.py     → 使用 action_prev 调用 encode_tactile
```

---

**最后更新**: 2024-01-21 | **状态**: 完成 ✅ | **需要**: 实际数据测试
