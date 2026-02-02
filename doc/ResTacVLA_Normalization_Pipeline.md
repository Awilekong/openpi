# ResTacVLA 输入数据归一化文档

## 1. 概述

ResTacVLA 相比基础 Pi0/Pi05 模型，新增了两个输入：
- **触觉图像 (tactile_image)**: GelSight 传感器图像
- **action_prev**: 前一时刻执行的动作 (即上一时刻的 action)

本文档说明这两个新增输入的归一化处理逻辑。

---

## 2. 触觉图像 (tactile_image) 归一化

### 2.1 归一化方式

触觉图像采用与**视觉图像相同的归一化方式**，归一化到 **[-1, 1]** 范围。

### 2.2 归一化公式

```
x_norm = x / 255.0 * 2.0 - 1.0
```

### 2.3 代码实现

**位置**: `src/openpi/policies/restac_policy.py`

```python
def _parse_tactile_image(image) -> np.ndarray:
    """Parse tactile image to float32 (H, W, C) format, normalized to [-1, 1]."""
    image = np.asarray(image)

    # Handle CHW to HWC conversion
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    # Convert to float32 and normalize to [-1, 1] (same as visual images)
    if np.issubdtype(image.dtype, np.integer):
        # uint8 [0, 255] -> float32 [-1, 1]
        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0

    return image
```

---

## 3. action_prev 归一化

### 3.1 归一化方式

action_prev 使用与 **actions 相同的归一化统计量**（mean/std 或 q01/q99）。

### 3.2 归一化公式

**PI05 (分位数归一化)**:
```
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0  →  范围 [-1, 1]
```

**PI0 (Z-score 归一化)**:
```
x_norm = (x - mean) / (std + 1e-6)
```

### 3.3 代码实现

**位置**: `src/openpi/policies/restac_policy.py`

```python
@dataclasses.dataclass(frozen=True)
class ResTacNormalizeActionPrev(transforms.DataTransformFn):
    """Normalize action_prev using the same statistics as actions."""
    norm_stats: dict
    use_quantiles: bool = True  # True for PI05, False for PI0

    def __call__(self, data: dict) -> dict:
        action_prev = data["action_prev"]
        stats = self.norm_stats["actions"]

        if self.use_quantiles:
            q01 = stats.q01[..., :action_prev.shape[-1]]
            q99 = stats.q99[..., :action_prev.shape[-1]]
            data["action_prev"] = (action_prev - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean = stats.mean[..., :action_prev.shape[-1]]
            std = stats.std[..., :action_prev.shape[-1]]
            data["action_prev"] = (action_prev - mean) / (std + 1e-6)

        return data
```

---

## 4. 归一化总结

| 输入数据 | 归一化方式 | 输出范围 |
|----------|------------|----------|
| **视觉图像** | `x / 255 * 2 - 1` | [-1, 1] |
| **触觉图像** | `x / 255 * 2 - 1` | [-1, 1] |
| **state** | 分位数归一化 (PI05) | [-1, 1] |
| **actions** | 分位数归一化 (PI05) | [-1, 1] |
| **action_prev** | 分位数归一化 (使用 actions 统计量) | [-1, 1] |

---

## 5. 关键代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| 触觉图像归一化 | `src/openpi/policies/restac_policy.py` | `_parse_tactile_image()` |
| action_prev 归一化 | `src/openpi/policies/restac_policy.py` | `ResTacNormalizeActionPrev` |
| action_prev 反归一化 | `src/openpi/policies/restac_policy.py` | `ResTacUnnormalizeActionPrev` |
| 数据配置 | `src/openpi/training/config.py` | `LeRobotResTacDataConfig` |
