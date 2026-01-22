# Pi0_ResTac 架构验证：Pi0 vs Pi05

## 核心发现

**✅ pi0_restac.py 完全基于 Pi05 架构实现**

不是 Pi0，而是 **Pi05**。

## 证据清单

### 1. 类文档声明（Line 630-632）
```python
class Pi0_ResTac(_model.BaseModel):
    """
    ResTacVLA: Two-Stage Cross-Attention Tactile Fusion Model (Pi05-based).

    Uses Pi05 backend with adaRMS for timestep injection.
    ...
    """
```

### 2. AdaRMS 配置（Line 658）
```python
llm = nnx_bridge.ToNNX(
    _gemma.Module(
        configs=[paligemma_config, action_expert_config],
        embed_dtype=config.dtype,
        adarms=True,  # Pi05 uses adaRMS  ← 关键标记
    )
)
llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])  # ← Pi05初始化
```

### 3. Max Token Length（Line 511）
```python
max_token_len: int = 200  # Pi05 uses 200 tokens
```

**对比**：
- **Pi0**：max_token_len = 48（如果 pi05=False）
- **Pi05**：max_token_len = 200（如果 pi05=True）
- **pi0_restac**：硬编码为 200 ✅

### 4. Time MLP（用于 AdaRMS）（Line 680-681）
```python
# ============ Time MLP for adaRMS (Pi05) ============
self.time_mlp_in = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)
self.time_mlp_out = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)
```

**说明**：这是 Pi05 特有的时间步条件化机制。Pi0 没有这个。

### 5. embed_suffix 中的 AdaRMS 处理（Line 909-955）

文档说明：
```python
"""
Uses time_mlp for adaRMS conditioning.  # ← 明确说明 Pi05 特性

Returns:
    ...
    adarms_cond: adaRMS conditioning [B, action_dim_internal]  # ← Pi05 特有输出
"""
```

实现：
```python
# 3. Time embedding + MLP for adaRMS conditioning
time_emb = posemb_sincos(
    timestep,
    self.action_in_proj.out_features,
    min_period=4e-3,
    max_period=4.0
)
time_emb = self.time_mlp_in(time_emb)
time_emb = nnx.swish(time_emb)
time_emb = self.time_mlp_out(time_emb)
adarms_cond = nnx.swish(time_emb)  # ← AdaRMS 条件向量

return action_tokens, input_mask, ar_mask, adarms_cond, tactile_features, gate_value, logvar
```

### 6. Transformer 前向传播中使用 AdaRMS（Line 1013）
```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens],
    mask=full_attn_mask,
    positions=positions,
    adarms_cond=[None, adarms_cond]  # Pi05: pass adaRMS conditioning  ← 关键
)
```

**Pi0** 没有 `adarms_cond` 参数。

### 7. 推理循环中也使用 AdaRMS（Line 1064）
```python
(_, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,
    adarms_cond=[None, adarms_cond]  # Pi05: pass adaRMS conditioning  ← 一致
)
```

## Pi0 vs Pi05 的关键区别

| 特性 | Pi0 | Pi05 | pi0_restac |
|------|-----|------|-----------|
| **AdaRMS Conditioning** | ❌ | ✅ | ✅ |
| **Time MLP** | ❌ | ✅ | ✅ |
| **Max Token Length** | 48 | 200 | 200 ✅ |
| **Timestep Injection** | 无 | AdaRMS | AdaRMS ✅ |
| **Discrete State Input** | No | Yes | Yes ✅ |
| **adarms_cond 传递** | 无 | 有 | 有 ✅ |

## 导入和依赖

### 不直接导入 Pi0 类
```python
# pi0_restac.py 中没有：
# from openpi.models.pi0 import Pi0
# class Pi0_ResTac(Pi0): ...
```

### 只使用工具函数
```python
# 仅引用工具函数（来自 pi0.py）：
# - make_attn_mask()
# - posemb_sincos()
```

## 结论

✅ **pi0_restac 完全基于 Pi05 架构实现**

- 明确的文档声明
- AdaRMS 全面集成
- Time MLP 用于时间步条件化
- 正确的 token 长度（200）
- 正确的条件化方式

**您的目的**"在pi05基础上改"**已经完全符合**！

## 建议

1. ✅ 继续在 Pi05 基础上进行改进
2. ✅ 所有 ResTac 特定的修改（tactile fusion、VQVAE 集成）都与 Pi05 架构兼容
3. ✅ AdaRMS 条件化和 Time MLP 正确工作
4. ✅ 数据流正确：timestep → time_mlp → adarms_cond → transformer

---

**验证时间**：2026-01-21
**验证状态**：✅ 完成
