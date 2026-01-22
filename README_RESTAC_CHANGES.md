# ResTacVLA - VQ Code 和 Logvar 集成

## 概览

本次实现在 openpi 的 ResTacVLA 模型中集成了 Unit-Align 的 Residual VQVAE，提供：

1. **VQ Codes（触觉事件编码）**: 离散的语义触觉事件表示
2. **Logvar（触觉不确定性）**: Prophet 网络对视觉预测的置信度

## 核心修改

### 文件: `openpi/src/openpi/models/pi0_restac.py`

#### 1. 新增类：ResidualVQVAEWrapper (第 48-138 行)

```python
class ResidualVQVAEWrapper:
    """Unit-Align 的 Residual VQVAE 包装器"""

    - __init__(): 加载 checkpoint 和初始化模型
    - forward(): 提取 VQ codes 和 logvar
    - _load_checkpoint(): 从文件加载权重
    - _construct_config_from_hparams(): 从 hparams 构造配置
```

**关键功能:**
- 自动检测 PyTorch 可用性
- 支持 CPU/GPU 自动选择
- 包含错误处理和 fallback 机制

#### 2. 更新类：TactileEncoderPlaceholder (第 380-401 行)

**原始功能:**
```python
# 旧: 返回 (features, sigma)
features, sigma = self.tactile_encoder(tactile_image)
```

**新功能:**
```python
# 新: 返回 (features, logvar)
# 支持 VQVAE 或 placeholder 模式
features, logvar = self.tactile_encoder(
    tactile_image,
    visual_image=visual_image,      # 可选，用于 VQVAE
    action=action                    # 可选，用于 VQVAE
)
```

**两种模式:**
- **Placeholder 模式**: 无 checkpoint，使用简单 MLP
- **VQVAE 模式**: 有 checkpoint，调用 ResidualVQVAEWrapper

#### 3. 更新方法：encode_tactile() (第 628-670 行)

**返回值变化:**
```python
# 旧: 返回 (tactile_features, gate_value)
# 新: 返回 (tactile_features, gate_value, logvar)

tactile_features, gate_value, logvar = self.encode_tactile(
    tactile_image,
    visual_image=visual_image,
    action=action
)
```

**处理逻辑:**
- 调用 TactileEncoderPlaceholder 的 `__call__` 方法
- 提取 logvar 用于后续 loss 计算
- 传递 visual_image 和 action 给 VQVAE

#### 4. 更新方法：embed_suffix() (第 775-838 行)

**返回值扩展:**
```python
# 旧: 返回 6 个值
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, tactile_features, gate_value

# 新: 返回 7 个值，加入 logvar
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, tactile_features, gate_value, logvar
```

**实现细节:**
- 从 observation 提取 visual_image（base_0_rgb）
- 从 noisy_actions 提取 action（第一时间步）
- 调用 encode_tactile 获得 logvar
- 缺少 tactile_image 时返回零向量

#### 5. 更新方法：compute_loss() (第 906-926 行)

**Logvar 在损失中的使用:**
```python
# 基础 sparse loss
sparse_loss = jnp.mean(gate_value)  # scalar

# 用 logvar 加权
logvar_mean = jnp.mean(logvar)
sparse_loss_weighted = sparse_loss * jnp.exp(logvar_mean)

# 最终损失
total_loss = flow_loss + weight * sparse_loss_weighted
```

**含义:**
- 高 logvar（低置信度）→ 更强的稀疏性约束
- 低 logvar（高置信度）→ 更弱的稀疏性约束

#### 6. 更新方法：sample_actions() (第 950-977 行)

**提取 logvar 一次:**
```python
# 在主循环外提取（不变化）
tactile_features, gate_value, logvar = self.encode_tactile(
    tactile_image,
    visual_image=visual_image,
    action=jnp.zeros((batch_size, self.action_dim))  # 初始动作为零
)

# 在 step 函数中使用 embed_suffix 返回新的 logvar（虽然被丢弃了）
suffix_tokens, ..., logvar = self.embed_suffix(observation, x_t, time)
```

#### 7. 更新配置：Pi0_ResTacConfig (第 275-285 行)

**新参数:**
```python
residual_vqvae_checkpoint: str | None = None
```

**用途:**
```python
# 启用 VQVAE
config = Pi0_ResTacConfig(
    residual_vqvae_checkpoint="/path/to/checkpoint.ckpt"
)

# 使用 placeholder
config = Pi0_ResTacConfig(
    residual_vqvae_checkpoint=None  # 默认值
)
```

#### 8. 更新初始化：Pi0_ResTac.__init__() (第 580-597 行)

**自动加载 VQVAE:**
```python
# 检查并加载 checkpoint
if config.residual_vqvae_checkpoint:
    try:
        vqvae_wrapper = ResidualVQVAEWrapper(
            checkpoint_path=config.residual_vqvae_checkpoint,
            frozen=True
        )
    except Exception as e:
        logger.warning(f"Failed to load VQVAE: {e}")
        vqvae_wrapper = None

# 传递给 TactileEncoderPlaceholder
self.tactile_encoder = TactileEncoderPlaceholder(
    output_dim=config.tactile_encoder_dim,
    vqvae_wrapper=vqvae_wrapper,
    rngs=rngs
)
```

## 数据流图

### 训练流程

```
observation (including tactile_image, visual_image)
    ↓
embed_prefix()  →  prefix tokens
    ↓
embed_suffix() {
    - extract tactile_image, visual_image, action
    - encode_tactile() {
        - TactileEncoderPlaceholder.__call__() {
            - ResidualVQVAEWrapper.forward() [if VQVAE available]
            - 或 placeholder MLP
        }
        - FactorizedGate() to compute gate_value
    }
    → tactile_features, gate_value, logvar
}
    ↓
Transformer forward (PaliGemma + LLM)
    ↓
two_stage_tactile_fusion() → tactile_correction
    ↓
compute_loss() {
    - flow_loss = ||v_t - u_t||²
    - sparse_loss = mean(gate_value)
    - sparse_loss_weighted = sparse_loss * exp(logvar)
    - total_loss = flow_loss + weight * sparse_loss_weighted
}
```

### 推理流程

```
observation (including tactile_image)
    ↓
embed_prefix() [cached]
    ↓
Loop (diffusion denoising):
    embed_suffix() → action tokens + adarms_cond + logvar
    ↓
    Transformer forward
    ↓
    two_stage_tactile_fusion() → correction
    ↓
    v_t = action_out_proj(suffix_out + correction)
    ↓
    x_t = x_t - v_t * dt
```

## 配置使用示例

### 最小化配置
```python
from openpi.models.pi0_restac import Pi0_ResTacConfig

config = Pi0_ResTacConfig()
model = config.create(rng=jax.random.PRNGKey(0))
```

### 使用 Unit-Align VQVAE
```python
config = Pi0_ResTacConfig(
    residual_vqvae_checkpoint="/path/to/unit_align_checkpoint.ckpt",
    tactile_encoder_dim=64,    # 与 VQVAE 的 event embedding dim 匹配
    fusion_dim=512,
    sparse_loss_weight=0.01,
)

model = config.create(rng=jax.random.PRNGKey(0))
```

## 重要备注

### 向后兼容性
✓ 完全向后兼容
- 不提供 checkpoint 时自动使用 placeholder
- 现有代码无需修改
- 权重加载器已支持新参数

### 数据格式
- **Tactile image:** [B, 128, 160, 3] 或 [B, 3, 128, 160]
- **Visual image:** [B, 3, 224, 224]
- **Action:** [B, 7]
- **Logvar:** [B, 1]

### 性能注意
- VQVAE 完全冻结（frozen=True）
- 推理时只增加一个 forward pass 的开销
- 可通过 checkpoint 预加载和缓存优化

## 权重文件格式

### 现有支持（openpi/training/weight_loaders.py）

`Pi0ResTacWeightLoader` 已正确处理新参数：

```python
class Pi0ResTacWeightLoader(WeightLoader):
    """
    加载 ResTac 模型权重，保留新的触觉相关参数：
    - .*lora.* (LoRA 适配)
    - .*tactile.* (触觉编码器)
    - .*gate.* (门控网络)
    - .*cross_attn.* (交叉注意力)
    - .*time_mlp.* (时间编码 for Pi05)
    """
```

## 完整性检查清单

- [x] ResidualVQVAEWrapper 框架完成
- [x] TactileEncoderPlaceholder 更新完成
- [x] encode_tactile 集成完成
- [x] embed_suffix 集成完成
- [x] Loss 计算更新完成
- [x] sample_actions 更新完成
- [x] 配置参数添加完成
- [x] 模型初始化逻辑完成
- [x] Imports 和错误处理完成
- [ ] ResidualVQVAEWrapper.forward() 需要完整实现（见 VQVAE_INTEGRATION_GUIDE.md）
- [ ] 端到端测试
- [ ] 实际数据验证

## 相关文档

- `RESTAC_IMPLEMENTATION_SUMMARY.md` - 实现总结
- `VQVAE_INTEGRATION_GUIDE.md` - 完整的实现指南（forward 方法）
- `openpi/src/openpi/policies/restac_policy.py` - 数据 transform（无需修改）

## 联系方式

遇到问题？参考：
1. `VQVAE_INTEGRATION_GUIDE.md` 中的"常见问题"
2. Unit-Align 的 CLAUDE.md
3. openpi 的官方文档
