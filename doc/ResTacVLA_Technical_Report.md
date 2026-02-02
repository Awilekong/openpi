# ResTacVLA 详细技术分析报告

本报告详细分析 ResTac（Residual Tactile VLA）模型的设计、数据流、网络架构、损失函数和训练流程。

**更新**: 当前版本提供两种融合架构：
1. **Pi0_ResTac (ForceVLA-style)**: 使用 Self-Attention + Gate 的残差融合
2. **Pi0_ResTac_TokenInAE (NEW)**: 将触觉 token 直接注入 Action Expert 输入

---

## 目录
1. [整体架构概述](#1-整体架构概述)
2. [数据加载与处理](#2-数据加载与处理)
3. [网络架构](#3-网络架构)
4. [网络数据流](#4-网络数据流)
5. [特征维度变换](#5-特征维度变换)
6. [损失函数](#6-损失函数)
7. [训练流程](#7-训练流程)
8. [推理流程](#8-推理流程)
9. [代码索引](#9-代码索引)
10. [Token-in-AE 架构详解](#10-token-in-ae-架构详解)

---

## 1. 整体架构概述

### 1.1 设计理念

ResTacVLA 是基于 Pi05 架构的触觉增强视觉-语言-行动模型。提供两种融合策略：

#### 策略 A: ForceVLA-style (Pi0_ResTac)
```
最终动作 = 基础VLA输出 + 门控触觉修正
a_final = a_base + g × tactile_correction
```

- **残差结构**: 触觉作为可选修正，不破坏基础VLA性能
- **自门控**: 当触觉无效/不确定时，g→0，自动禁用修正
- **ForceVLA风格融合**: concat → Self-Attention → 取最后50个 → 门控

#### 策略 B: Token-in-AE (Pi0_ResTac_TokenInAE) 【新增】
```
surprise_token = gate × tactile_token + (1-gate) × default_embedding
Action Expert 输入 = [surprise_token, action_0, ..., action_49]
```

- **直接注入**: 触觉信息作为 token 直接注入 Action Expert
- **门控融合**: gate=1 时使用真实触觉，gate=0 时使用可学习默认 embedding
- **无残差**: Action Expert 直接输出最终动作

### 1.2 类继承关系

**Pi0_ResTac (ForceVLA-style)**:
```
Pi0_ResTac
    ↓ 继承
BaseModel (flax.nnx)
    ↓ 组合
├── PaliGemma (VLM + Image Encoder)
├── TactileEncoder (VQVAE触觉编码，必须提供checkpoint)
├── NecessityGate (基于不确定性的门控网络)
├── tactile_to_vlm_proj (触觉投影到VLM维度)
└── fusion_self_attn (ForceVLA风格Self-Attention融合)
```

**Pi0_ResTac_TokenInAE (Token-in-AE)**:
```
Pi0_ResTac_TokenInAE
    ↓ 继承
BaseModel (flax.nnx)
    ↓ 组合
├── PaliGemma (VLM + Image Encoder)
├── vqvae_wrapper (VQVAE 直接使用，不经过 TactileEncoder)
├── NecessityGate (基于不确定性的门控网络)
├── tactile_token_proj (q_event [64] → token [1024])
└── default_tactile_embedding (可学习的默认 embedding)
```

### 1.3 核心维度常量
| 常量 | 值 | 说明 |
|------|-----|------|
| `vlm_dim` | 2048 | PaliGemma Gemma-2B 隐藏维度 |
| `action_dim_internal` | 1024 | Action Expert Gemma-300M 维度 |
| `fusion_dim` | 512 | 触觉融合空间维度 |
| `action_dim` | 32 | 外部动作维度（7D填充到32D） |
| `action_horizon` | 50 | 预测动作步数 |
| `use_gate` | True | 是否使用门控机制（消融实验配置） |

---

## 2. 数据加载与处理

### 2.1 数据集格式
**数据路径**: `/home/dataset-local/data/megvii_post/tac/plug_charger/`

**元数据** (`meta/info.json`):
- 总剧集: 105
- 总帧数: 9,355
- FPS: ~15
- 机器人: Franka

**特征定义**:
```
observation.state [7D]:
  - ee_x, ee_y, ee_z        # 末端执行器位置
  - ee_rot_x, ee_rot_y, ee_rot_z  # RPY旋转
  - gripper                  # 夹爪状态

action [7D]:
  - 同state格式，表示目标位姿

action_prev [7D]:
  - state_t - state_t-1  # 预计算的前一动作

observation.images:
  - main_realsense_rgb     [224, 224, 3] uint8
  - side_realsense_rgb     [224, 224, 3] uint8
  - handeye_realsense_rgb  [224, 224, 3] uint8
  - gelsight_left_rgb      [128, 160, 3] uint8  # 触觉传感器
```

### 2.2 数据转换流程

```
┌─────────────────────────────────────────────────────────────┐
│ LeRobot Dataset (原始格式)                                   │
│ observation.state, observation.images.*, action, action_prev │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: ResTacInputs (restac_policy.py:77-164)              │
│ ├─ state [7D] → pad → [32D]                                 │
│ ├─ action_prev [7D] → pad → [32D]                           │
│ ├─ 图像格式转换: CHW→HWC, float→uint8                        │
│ └─ 图像映射: main→base_0, handeye→left_wrist, side→right_wrist │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: DeltaActions (transforms.py:204-244)                │
│ mask = (T,T,T,T,T,T,F)  # 前6维delta，第7维(夹爪)绝对值     │
│ actions[:6] = actions[:6] - state[:6]                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Normalize (transforms.py:115-146)                   │
│ 使用 quantile normalization (Pi05默认):                     │
│ x_norm = (x - q01) / (q99 - q01) * 2 - 1  # 范围 [-1, 1]   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: ResizeImages (transforms.py:185-191)                │
│ 所有图像统一到 224×224（含触觉 128×160→224×224）            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: TokenizePrompt                                      │
│ 文本提示 → token序列 [max_token_len=200]                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 模型输入格式                                                 │
│ state: [B, 32]                                              │
│ action_prev: [B, 32]                                        │
│ images: {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb,   │
│          tactile_0} 各 [B, 224, 224, 3]                     │
│ actions: [B, 50, 32]                                        │
│ tokenized_prompt: [B, 200]                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 输入处理关键代码

**ResTacInputs** (`restac_policy.py:77-164`):
```python
# 状态填充
state = transforms.pad_to_dim(state, action_dim=32)  # [7] → [32]

# 图像映射（PI05模式）
names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "tactile_0")
images = (main_image, handeye_image, side_image, tactile_image)
```

### 2.4 数据归一化详解

**重要**: 不同类型的数据有不同的归一化方式，理解这些差异对于正确使用模型至关重要。

#### 2.4.1 归一化统计量计算

归一化统计量通过 `scripts/compute_norm_stats.py` 预计算：

```python
# compute_norm_stats.py:102-103
keys = ["state", "actions"]  # 只有这两个key被归一化
stats = {key: normalize.RunningStats() for key in keys}
```

**注意**: `action_prev` 和 图像 不在归一化的keys列表中！

#### 2.4.2 各数据类型归一化方式

| 数据类型 | 归一化方式 | 归一化位置 | 范围 |
|---------|-----------|-----------|------|
| **state** | Quantile (Pi05) | transforms.Normalize | [-1, 1] |
| **actions** | Quantile (Pi05) | transforms.Normalize | [-1, 1] |
| **action_prev** | **不归一化** | - | 原始值 |
| **视觉图像** | uint8→float32 | Observation.from_dict | [-1, 1] |
| **触觉图像** | uint8→float32 | Observation.from_dict | [-1, 1] |

#### 2.4.3 state 和 actions 归一化

**归一化公式** (Quantile, Pi05默认):
```python
# transforms.py:141-145
def _normalize_quantile(self, x, stats: NormStats):
    q01, q99 = stats.q01, stats.q99
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

# 输出范围: [-1, 1]
```

**反归一化公式** (推理输出处理):
```python
# transforms.py:175-181
def _unnormalize_quantile(self, x, stats: NormStats):
    q01, q99 = stats.q01, stats.q99
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

#### 2.4.4 action_prev 处理 (重要!)

**action_prev 不进行归一化处理**，直接使用原始数值。

```
action_prev = state_t - state_t-1  # 原始值，单位与state相同
```

**原因**: action_prev 被送入 Unit-Align VQVAE 的 Prophet 网络，该网络期望原始物理单位的输入。

**数据流**:
```
数据集 action_prev [7D]
    │
    ├─→ ResTacInputs: pad到32D
    │   action_prev [7D] → [32D] (后25维填0)
    │
    ├─→ 不经过 Normalize 转换
    │
    ├─→ Observation.from_dict: 直接存储
    │
    └─→ 模型使用: encode_tactile() 传给 VQVAE
        action_prev[:7] 被提取并送入 Prophet 网络
```

#### 2.4.5 图像归一化 (包括触觉图像)

**所有图像（视觉 + 触觉）统一处理**：

```python
# model.py:121-126 (Observation.from_dict)
# If images are uint8, convert them to [-1, 1] float32.
for key in data["image"]:
    if data["image"][key].dtype == np.uint8:
        data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
```

**图像归一化公式**:
```
float32_image = uint8_image / 255.0 * 2.0 - 1.0
# 输入: [0, 255] uint8
# 输出: [-1, 1] float32
```

**触觉图像处理流程**:
```
GelSight 原始图像 [128, 160, 3] uint8
    │
    ├─→ ResTacInputs: 解析为 tactile_0
    │
    ├─→ ResizeImages: 调整为 [224, 224, 3] uint8
    │
    ├─→ Observation.from_dict: 转换为 float32
    │   uint8 / 255.0 * 2.0 - 1.0 → [-1, 1] float32
    │
    └─→ 模型使用: encode_tactile() → VQVAE
```

#### 2.4.6 训练 vs 推理归一化处理

**训练时**:
```
数据加载 → ResTacInputs → DeltaActions → Normalize(state, actions)
         → ResizeImages → TokenizePrompt → Observation.from_dict(图像归一化)
         → 模型训练
```

**推理时**:
```
输入观测 → ResTacInputs → DeltaActions → Normalize(state, actions)
         → ResizeImages → TokenizePrompt → Observation.from_dict(图像归一化)
         → 模型推理 → Unnormalize(actions) → AbsoluteActions → ResTacOutputs
```

**关键区别**:
- 训练：Normalize 处理 state 和 actions
- 推理：Normalize 处理 state；Unnormalize 处理输出 actions
- action_prev：训练和推理都不归一化

#### 2.4.7 归一化相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 归一化统计量计算 | scripts/compute_norm_stats.py | 102-113 |
| Normalize转换 | transforms.py | 115-146 |
| Unnormalize转换 | transforms.py | 148-181 |
| 图像uint8→float32 | model.py | 121-126 |
| ResizeImages | transforms.py | 185-191 |
| 训练数据转换链 | data_loader.py | 190-197 |
| 推理数据转换链 | policy_config.py | 75-94 |

---

## 3. 网络架构

### 3.1 模块组成图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pi0_ResTac                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ PaliGemma (VLM Backbone)                                     │   │
│  │ ├─ img: SigLIP So400m/14 (图像编码)                          │   │
│  │ │   输入: [B, 224, 224, 3] → 输出: [B, ~576, 1152]           │   │
│  │ └─ llm: Gemma-2B + Gemma-300M (双专家LLM)                    │   │
│  │       PaliGemma: 前缀处理，dim=2048                          │   │
│  │       ActionExpert: 后缀处理，dim=1024，带adaRMS             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TactileEncoder (VQVAE触觉编码, 348-405行)                    │   │
│  │ 必须提供有效的VQVAE checkpoint，否则报错                     │   │
│  │ ├─ ResidualVQVAEWrapper: 加载预训练VQVAE模型                │   │
│  │ │   q_event [B, 64] from Unit-Align VQ codebook             │   │
│  │ │   logvar [B, 1] from Prophet网络 (不确定性估计)           │   │
│  │ └─ project_vq: [B, 64] → [B, fusion_dim=512]                │   │
│  │ 输出: features [B, 1, 512], logvar [B, 1]                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ NecessityGate (门控网络, 411-448行)                          │   │
│  │ g = necessity(σ) = sigmoid((σ - threshold) / temperature)  │   │
│  │ ├─ σ (logvar): 不确定性，来自VQVAE Prophet网络              │   │
│  │ ├─ threshold: 可学习参数，初始化为0                         │   │
│  │ └─ temperature: 可学习参数，初始化为1                       │   │
│  │ 语义：不确定性高(logvar大) → 门开(g→1) → 启用触觉修正       │   │
│  │ 输入: logvar [B, 1] → 输出: g [B, 1] ∈ [0,1]                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ForceVLA-style Tactile Fusion (触觉融合模块)                 │   │
│  │                                                               │   │
│  │ tactile_to_vlm_proj: Linear(512, 2048)                       │   │
│  │   投影触觉特征到VLM维度                                       │   │
│  │                                                               │   │
│  │ fusion_self_attn: MultiHeadAttention(in=2048, out=1024)      │   │
│  │   Self-Attention 处理 concat([prefix_out, tactile])          │   │
│  │   (替代 ForceVLA 的 LIMoE)                                   │   │
│  │                                                               │   │
│  │ 融合流程:                                                     │   │
│  │   concat([prefix_out, tactile]) → Self-Attn → 取最后50个     │   │
│  │   → × gate → gated_correction [B, 50, 1024]                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Action Projections                                          │   │
│  │ ├─ action_in_proj: [32] → [1024] (输入投影)                 │   │
│  │ └─ action_out_proj: [1024] → [32] (输出投影)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Time MLP (adaRMS条件化, Pi05特性, 679-681行)                 │   │
│  │ time_emb [B, 1024] → time_mlp_in → swish → time_mlp_out    │   │
│  │ 输出: adarms_cond [B, 1024] 用于调制Action Expert           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 ForceVLA-style Fusion 结构
```
位置: pi0_restac.py (tactile_fusion_forcevla_style)

ForceVLA vs ResTac 对比:
┌──────────┬──────────────────────────┬──────────────────────────┐
│ 组件     │ ForceVLA                 │ ResTac                   │
├──────────┼──────────────────────────┼──────────────────────────┤
│ 输入     │ prefix_out + force       │ prefix_out + tactile     │
│ 处理     │ LIMoE (Self-Attn + MOE)  │ Self-Attn + Gate         │
│ 取值     │ 最后50个                 │ 最后50个 (相同)          │
│ 调制     │ MOE 专家选择             │ Gate 开关                │
│ 输出     │ + suffix_out             │ + suffix_out (相同)      │
└──────────┴──────────────────────────┴──────────────────────────┘

融合模块组成:
  ├─ tactile_to_vlm_proj: Linear(512, 2048)
  │   将触觉特征从 fusion_dim 投影到 VLM 维度
  │
  └─ fusion_self_attn: MultiHeadAttention
      in_features: 2048 (VLM dimension)
      out_features: 1024 (Action dimension)
      num_heads: 8

计算流程:
  tactile_vlm = tactile_to_vlm_proj(tactile_features)  # [B, 1, 2048]
  fused = concat([prefix_out, tactile_vlm], axis=1)    # [B, S+1, 2048]
  attn_out = fusion_self_attn(fused)                   # [B, S+1, 1024]
  correction = attn_out[:, -50:]                       # [B, 50, 1024]
  gated_correction = gate_value × correction           # [B, 50, 1024]
```

---

## 4. 网络数据流

### 4.1 训练时完整数据流

```
输入:
  observation: {images, state, action_prev, prompt}
  actions: [B, 50, 32]

═══════════════════════════════════════════════════════════════════════
Step 1: 流匹配目标生成 (compute_loss:987-994)
═══════════════════════════════════════════════════════════════════════
  noise ~ N(0, 1)                    # [B, 50, 32]
  t ~ Beta(1.5, 1) × 0.999 + 0.001   # [B]
  x_t = t × noise + (1-t) × actions  # 插值噪声动作 [B, 50, 32]
  u_t = noise - actions              # 流匹配目标速度 [B, 50, 32]

═══════════════════════════════════════════════════════════════════════
Step 2: 编码前缀 (embed_prefix:850-884)
═══════════════════════════════════════════════════════════════════════
  3个视觉图像 [B, 3, 224, 224, 3]
      │
      ├─→ SigLIP编码 → image_tokens [B, 3, ~576, 1152]
      │
      ├─→ Gemma投影 → [B, 3, ~576, 2048]
      │
  tokenized_prompt [B, 200]
      │
      ├─→ Embedding → [B, 200, 2048]
      │
      └─→ 拼接 → prefix_tokens [B, S_prefix, 2048]
                 prefix_mask, prefix_ar_mask

═══════════════════════════════════════════════════════════════════════
Step 3: 编码后缀 + 提取触觉 (embed_suffix:891-961)
═══════════════════════════════════════════════════════════════════════
  state [B, 32]
      ├─→ action_in_proj → [B, 1, 1024]

  x_t [B, 50, 32]
      ├─→ action_in_proj → [B, 50, 1024]

  time [B]
      ├─→ posemb_sincos(1024) → [B, 1024]
      ├─→ time_mlp_in → swish → time_mlp_out → swish
      └─→ adarms_cond [B, 1024]

  拼接 → suffix_tokens [B, 51, 1024]
         suffix_mask, suffix_ar_mask

  tactile_image [B, 224, 224, 3]
      ├─→ TactileEncoder
      │     ├─→ VQVAE或MLP编码
      │     └─→ raw_features [B, 1, 512], logvar [B, 1]
      │
      ├─→ tactile_proj → tactile_features [B, 1, 512]
      │
      └─→ FactorizedGate(features, logvar) → gate_value [B, 1]

═══════════════════════════════════════════════════════════════════════
Step 4: Transformer前向 (compute_loss:1003-1014)
═══════════════════════════════════════════════════════════════════════
  合并mask:
    input_mask = concat(prefix_mask, suffix_mask)
    ar_mask = concat(prefix_ar_mask, suffix_ar_mask)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = cumsum(input_mask) - 1

  (prefix_out, suffix_out) = PaliGemma.llm(
      [prefix_tokens, suffix_tokens],
      mask=attn_mask,
      positions=positions,
      adarms_cond=[None, adarms_cond]  # 只对action expert使用adaRMS
  )

  prefix_out: [B, S_prefix, 2048]  (VLM输出)
  suffix_out: [B, 51, 1024]        (Action Expert输出)

═══════════════════════════════════════════════════════════════════════
Step 5: ForceVLA风格触觉融合 (tactile_fusion_forcevla_style)
═══════════════════════════════════════════════════════════════════════
  1. 投影触觉到VLM维度:
    tactile_vlm = tactile_to_vlm_proj(tactile_features)  # [B, 1, 2048]

  2. 拼接 prefix_out 和 tactile:
    fused = concat([prefix_out, tactile_vlm], axis=1)  # [B, S+1, 2048]

  3. Self-Attention (替代ForceVLA的LIMoE):
    attn_out = fusion_self_attn(fused)  # [B, S+1, 1024]

  4. 取最后50个 (ForceVLA风格):
    correction = attn_out[:, -50:]  # [B, 50, 1024]

  5. 门控 (替代MOE):
    gate_expanded = gate_value[:, :, None]  # [B, 1, 1]
    gated_correction = gate_expanded × correction  # [B, 50, 1024]

═══════════════════════════════════════════════════════════════════════
Step 6: 残差输出 + 损失计算 (compute_loss:1025-1046)
═══════════════════════════════════════════════════════════════════════
  action_out = suffix_out[:, -50:] + gated_correction  # [B, 50, 1024]
  v_t = action_out_proj(action_out)                     # [B, 50, 32]

  # 流匹配损失
  flow_loss = (v_t - u_t)²                             # [B, 50, 32]

  # 稀疏损失
  sparse_loss = mean(gate_value) × exp(mean(logvar))

  # 总损失
  total_loss = mean(flow_loss, axis=-1) + 0.01 × sparse_loss  # [B, 50]
```

### 4.2 推理时数据流 (sample_actions:1053-1140)

```
输入: observation (无actions)

═══════════════════════════════════════════════════════════════════════
Step 1: 缓存前缀 (一次性计算)
═══════════════════════════════════════════════════════════════════════
  prefix_tokens, prefix_mask, _ = embed_prefix(observation)
  (prefix_out_cached, _), kv_cache = PaliGemma.llm([prefix_tokens, None])

═══════════════════════════════════════════════════════════════════════
Step 2: 缓存触觉特征 (一次性计算)
═══════════════════════════════════════════════════════════════════════
  tactile_features, gate_value, logvar = encode_tactile(
      tactile_image, visual_3views, action_prev
  )

═══════════════════════════════════════════════════════════════════════
Step 3: 扩散反向循环 (10-50步)
═══════════════════════════════════════════════════════════════════════
  x_t ← N(0, 1)  # [B, 50, 32]
  dt = -1/num_steps

  for t in [1.0 → 0.0]:
      # 编码后缀 (使用当前x_t)
      suffix_tokens, ..., adarms_cond = embed_suffix(obs, x_t, t)

      # Transformer前向 (使用KV缓存)
      (_, suffix_out), _ = PaliGemma.llm(
          [None, suffix_tokens],
          kv_cache=kv_cache,
          adarms_cond=[None, adarms_cond]
      )

      # ForceVLA风格融合 (重用缓存的prefix_out和tactile_features)
      gated_correction = tactile_fusion_forcevla_style(
          tactile_features, prefix_out_cached, suffix_out, gate_value
      )

      # 计算速度并更新
      action_out = suffix_out[:, -50:] + gated_correction
      v_t = action_out_proj(action_out)

      # Euler更新
      x_t = x_t + dt × v_t
      t = t + dt

  return x_0  # [B, 50, 32]
```

---

## 5. 特征维度变换

### 5.1 完整维度流转表

| 阶段 | 输入 | 操作 | 输出 | 代码位置 |
|------|------|------|------|----------|
| **前缀路径** |
| RGB图像 | [B, 224, 224, 3] | SigLIP | [B, ~576, 1152] | embed_prefix |
| 图像token | [B, ~576, 1152] | Gemma投影 | [B, ~576, 2048] | embed_prefix |
| 文本prompt | [B, 200] | Embedding | [B, 200, 2048] | embed_prefix |
| 前缀tokens | [B, S, 2048] | Gemma-2B | [B, S, 2048] | compute_loss:1009 |
| **后缀路径** |
| state | [B, 32] | action_in_proj | [B, 1, 1024] | embed_suffix |
| actions | [B, 50, 32] | action_in_proj | [B, 50, 1024] | embed_suffix |
| time | [B] | sincos+MLP | [B, 1024] | embed_suffix |
| 后缀tokens | [B, 51, 1024] | Gemma-300M | [B, 51, 1024] | compute_loss:1009 |
| **触觉路径** |
| 触觉图像 | [B, 224, 224, 3] | VQVAE | [B, 64] q_event | encode_tactile |
| VQ码 | [B, 64] | project_vq | [B, 1, 512] | TactileEncoder |
| logvar | [B, 1] | NecessityGate | [B, 1] gate | encode_tactile |
| **融合路径 (ForceVLA风格)** |
| 触觉投影 | [B, 1, 512] | tactile_to_vlm_proj | [B, 1, 2048] | fusion |
| 拼接 | [B,S,2048]+[B,1,2048] | concat | [B, S+1, 2048] | fusion |
| Self-Attn | [B, S+1, 2048] | fusion_self_attn | [B, S+1, 1024] | fusion |
| 取最后50 | [B, S+1, 1024] | slice | [B, 50, 1024] | fusion |
| 门控修正 | [B,1,1]×[B,50,1024] | 逐元素乘 | [B, 50, 1024] | fusion |
| **输出路径** |
| action_out | [B, 50, 1024] | action_out_proj | [B, 50, 32] | compute_loss:1028 |
| 后处理 | [B, 50, 32] | 截取前7维 | [B, 50, 7] | ResTacOutputs |

### 5.2 维度变换图示

```
                    ┌──────────────────────────────────────────┐
                    │           输入 Observation                │
                    └──────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┬──────────────────┐
          │                    │                    │                  │
          ▼                    ▼                    ▼                  ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ 3×视觉   │        │ 文本     │        │ 状态     │        │ 触觉     │
    │[B,3,224, │        │[B,200]   │        │[B,32]    │        │[B,224,   │
    │ 224,3]   │        │          │        │          │        │ 224,3]   │
    └────┬─────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘
         │                   │                   │                   │
         │ SigLIP            │ Embed             │ Proj              │ VQVAE/MLP
         ▼                   ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
    │[B,~1728, │        │[B,200,   │        │[B,1,     │        │[B,1,512] │
    │ 1152]    │        │ 2048]    │        │ 1024]    │        │+logvar   │
    └────┬─────┘        └────┬─────┘        └────┬─────┘        │[B,1]     │
         │                   │                   │              └────┬─────┘
         │ 投影到2048        │                   │                   │
         ▼                   ▼                   │              Gate │
    ┌──────────────────────────┐                 │                   ▼
    │ prefix_tokens            │                 │              ┌──────────┐
    │ [B, S_prefix, 2048]      │                 │              │gate [B,1]│
    └────────────┬─────────────┘                 │              └────┬─────┘
                 │                               │                   │
                 │                    ┌──────────┴───────┐           │
                 │                    │ +actions+time    │           │
                 │                    ▼                  │           │
                 │           ┌──────────────────┐       │           │
                 │           │suffix_tokens     │       │           │
                 │           │[B, 51, 1024]     │       │           │
                 │           └────────┬─────────┘       │           │
                 │                    │                 │           │
    ═════════════╪════════════════════╪═════════════════╪═══════════╪═════════
                 │    Transformer     │                 │           │
                 ▼                    ▼                 │           │
          ┌──────────┐        ┌──────────┐             │           │
          │prefix_out│        │suffix_out│             │           │
          │[B,S,2048]│        │[B,51,1024│             │           │
          └────┬─────┘        └────┬─────┘             │           │
               │                   │                   │           │
    ═══════════╪═══════════════════╪═══════════════════╪═══════════╪═════════
               │   ForceVLA-style  │                   │           │
               │   Fusion          ▼                   │           │
               │           ┌──────────────┐            │           │
               │           │提取最后50步  │            │           │
               │           │[B,50,1024]   │            │           │
               │           └──────┬───────┘            │           │
               │                  │                    │           │
               ▼                  │                    ▼           │
          ┌───────────────────────┼──────────────────────────────────▼─────┐
          │ ForceVLA-style Fusion:                                        │
          │ tactile_vlm = tactile_to_vlm_proj(tactile_feat)              │
          │ [B,1,512] → [B,1,2048]                                       │
          │                                                               │
          │ fused = concat([prefix_out, tactile_vlm]) → [B,S+1,2048]     │
          │ attn_out = fusion_self_attn(fused) → [B,S+1,1024]           │
          │ correction = attn_out[:,-50:] → [B,50,1024]                  │
          │                                                               │
          │ gated_correction = gate × correction → [B,50,1024]           │
          └──────────────────────────────┬────────────────────────────────┘
                                         │
    ═════════════════════════════════════╪════════════════════════════════════
                     残差融合            │
                         ┌───────────────┘
                         ▼
                  ┌──────────────┐
                  │suffix_out[-50│
                  │:] + gated_   │
                  │correction    │
                  │[B,50,1024]   │
                  └──────┬───────┘
                         │
                         │ action_out_proj
                         ▼
                  ┌──────────────┐
                  │ v_t          │
                  │ [B,50,32]    │
                  └──────────────┘
```

---

## 6. 损失函数

### 6.1 损失组成

**位置**: `pi0_restac.py:980-991`

```python
# 1. 流匹配损失 (主损失，始终启用)
flow_loss = (v_t - u_t)²  # [B, 50, 32]
# v_t: 模型预测的速度场
# u_t = noise - actions: 目标速度场
total_loss = mean(flow_loss, axis=-1)  # [B, 50]

# 2. 稀疏损失 (可选，默认禁用)
# 通过 config.use_sparse_loss 控制是否启用
if config.use_sparse_loss:
    # gate_value: [B, 1]，取batch均值得到标量
    sparse_loss = mean(gate_value)
    total_loss = total_loss + sparse_loss_weight × sparse_loss
```

### 6.2 损失设计理念

| 损失项 | 公式 | 目的 | 是否默认启用 |
|--------|------|------|-------------|
| 流匹配 | ‖v_t - u_t‖² | 学习从噪声到动作的速度场 | 是 |
| 稀疏 | mean(g) | 鼓励门控保持关闭 | 否 (可选) |

### 6.3 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_sparse_loss` | bool | False | 是否启用稀疏损失 |
| `sparse_loss_weight` | float | 0.01 | 稀疏损失权重 (仅当use_sparse_loss=True时有效) |
| `use_gate` | bool | True | 是否使用门控调制（消融实验：False时直接加correction） |

**关键特性**:
- **残差安全**: 当门控g→0时，模型退化为基础VLA
- **不确定性驱动**: NecessityGate基于logvar自动控制门开关
- **简化设计**: 去除了modulation模块和exp(logvar)惩罚项

---

## 7. 训练流程

### 7.1 训练配置

**位置**: `config.py:1084-1121`

```python
TrainConfig(
    name="pi05_restac",
    model=Pi0_ResTacConfig(
        fusion_dim=512,
        num_cross_attn_heads=8,
        use_sparse_loss=False,  # 稀疏损失默认禁用
        residual_vqvae_checkpoint="path/to/vqvae/checkpoint",  # 必须提供！
    ),
    data=LeRobotResTacDataConfig(repo_id="your_tactile_dataset"),
    weight_loader=Pi0ResTacWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=50_000,
    batch_size=4,
)
```

**重要**: `residual_vqvae_checkpoint` 是必须参数，必须提供有效的VQVAE checkpoint路径。

### 7.2 权重加载策略

**位置**: `weight_loaders.py:77-107`

```python
class Pi0ResTacWeightLoader:
    """加载Pi05基础权重，保留新增触觉模块"""

    def load(self, params):
        loaded = restore_params(pi05_checkpoint_path)
        return _merge_params(
            loaded, params,
            missing_regex=".*tactile.*|.*gate.*|.*stage.*cross_attn.*|.*time_mlp.*"
        )
```

**保留的新参数**:
- `.*tactile.*`: 触觉编码器 (TactileEncoder, project_vq)
- `.*gate.*`: NecessityGate (sigma_threshold, sigma_temperature)
- `.*stage.*cross_attn.*`: 两阶段交叉注意力
- `.*time_mlp.*`: adaRMS时间MLP

### 7.3 优化器配置

```python
学习率调度 (CosineDecay):
  warmup_steps: 1,000
  peak_lr: 2.5e-5
  decay_steps: 30,000
  decay_lr: 2.5e-6

优化器 (AdamW):
  β1: 0.9
  β2: 0.95
  weight_decay: 1e-10
  gradient_clip: 1.0

EMA:
  decay: 0.99
```

### 7.4 数据批次组织

```python
batch_size: 4 (全局)
local_batch_size: 4 // num_devices

batch结构:
{
    "state": [4, 32],
    "action_prev": [4, 32],
    "images": {
        "base_0_rgb": [4, 224, 224, 3],
        "left_wrist_0_rgb": [4, 224, 224, 3],
        "right_wrist_0_rgb": [4, 224, 224, 3],
        "tactile_0": [4, 224, 224, 3]
    },
    "actions": [4, 50, 32],
    "tokenized_prompt": [4, 200],
    "tokenized_prompt_mask": [4, 200]
}
```

---

## 8. 推理流程

### 8.1 推理优化策略

1. **前缀缓存**: 视觉+语言只编码一次，缓存KV
2. **触觉缓存**: tactile_features和gate_value只计算一次
3. **扩散步数**: 默认10步，可调整为50步提高质量

### 8.2 输出后处理

```
模型输出: [B, 50, 32]
    │
    ├─→ Denormalize: 反归一化
    │   x = (x_norm + 1) / 2 × (q99 - q01) + q01
    │
    ├─→ AbsoluteActions: 恢复绝对值
    │   action[:6] = action[:6] + state[:6]
    │
    └─→ ResTacOutputs: 截取
        output = action[:, :7]  # [B, 50, 7]
```

---

## 9. 代码索引

### 9.1 核心文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 模型定义 | `src/openpi/models/pi0_restac.py` | ResTac网络架构 |
| 策略处理 | `src/openpi/policies/restac_policy.py` | 输入输出转换 |
| 训练配置 | `src/openpi/training/config.py` | 训练参数配置 |
| 数据加载 | `src/openpi/training/data_loader.py` | LeRobot数据加载 |
| 转换函数 | `src/openpi/transforms.py` | 数据预处理转换 |
| 权重加载 | `src/openpi/training/weight_loaders.py` | 检查点加载 |

### 9.2 关键行号索引

**pi0_restac.py**:
| 功能 | 行号 |
|------|------|
| Pi0_ResTacConfig | 454-538 |
| ResidualVQVAEWrapper | 50-202 |
| CrossAttentionBlock (保留但不使用) | 246-341 |
| TactileEncoder | 348-405 |
| NecessityGate | 411-448 |
| Pi0_ResTac.__init__ | 592-690 |
| encode_tactile | 700-740 |
| tactile_fusion_forcevla_style | 745-800 |
| embed_prefix | 805-840 |
| embed_suffix | 845-920 |
| compute_loss | 925-1000 |
| sample_actions | 1005-1100 |

**restac_policy.py**:
| 功能 | 行号 |
|------|------|
| _parse_image | 27-39 |
| _compute_action_prev | 42-61 |
| ResTacInputs | 77-164 |
| ResTacOutputs | 167-179 |
| ResTacNormalization | 182-214 |
| ResTacDenormalization | 217-236 |

**config.py**:
| 功能 | 行号 |
|------|------|
| LeRobotResTacDataConfig | 362-416 |
| pi05_restac配置 | 1084-1098 |
| pi05_restac_lora配置 | 1099-1121 |

**transforms.py**:
| 功能 | 行号 |
|------|------|
| Normalize | 115-146 |
| Denormalize | 148-181 |
| ResizeImages | 185-191 |
| DeltaActions | 204-244 |
| AbsoluteActions | 246-270 |
| pad_to_dim | 423-430 |
| make_bool_mask | 433-452 |

### 9.3 修改位置快速查找

| 需求 | 修改文件 | 关键位置 |
|------|---------|----------|
| 调整融合维度 | pi0_restac.py | 467 (fusion_dim) |
| 修改门控机制 | pi0_restac.py | 411-448 (NecessityGate) |
| 启用/禁用稀疏损失 | pi0_restac.py | 469 (use_sparse_loss) |
| 调整稀疏损失权重 | pi0_restac.py | 470 (sparse_loss_weight) |
| 启用/禁用门控消融 | pi0_restac.py | 484 (use_gate) |
| 添加新的图像输入 | restac_policy.py | 113-139 |
| 修改动作维度 | pi0_restac.py | 462 (action_dim) |
| 修改触觉融合 | pi0_restac.py | tactile_fusion_forcevla_style |
| 修改数据增强 | transforms.py | 185-191 |
| 修改归一化方式 | transforms.py | 115-181 |
| 指定VQVAE checkpoint | pi0_restac.py | 473 (residual_vqvae_checkpoint) |

---

## 验证方法

### 单元测试
```bash
# 测试输入转换
python -c "from openpi.policies.restac_policy import make_restac_example, ResTacInputs; ..."

# 测试模型前向
python -c "from openpi.models.pi0_restac import Pi0_ResTacConfig; ..."
```

### 端到端训练
```bash
cd /home/dataset-local/code/openpi
python scripts/train.py --config pi05_restac
```

### 推理测试
```bash
python scripts/serve_policy.py --config pi05_restac --checkpoint <path>
```

---

## 10. 快速启动训练指南

使用数据集 `/home/dataset-local/data/megvii_post/tac/plug_charger` 开启训练，需完成以下步骤：

### Step 1: 准备 VQVAE Checkpoint (必须)

```bash
# 获取 Unit-Align VQVAE checkpoint
# 选项A: 从预训练模型下载
wget <vqvae_checkpoint_url> -O /path/to/vqvae/checkpoint

# 选项B: 自行训练 VQVAE (参考 Unit-Align 论文)
```

**重要**: 没有 VQVAE checkpoint 无法启动训练！

### Step 2: 计算归一化统计量

```bash
cd /home/dataset-local/code/openpi

python scripts/compute_norm_stats.py \
    --data-config LeRobotResTacDataConfig \
    --repo-id /home/dataset-local/data/megvii_post/tac/plug_charger \
    --output-path assets/norm_stats/plug_charger.json
```

### Step 3: 创建任务描述文件 (可选)

如果 `prompt_from_task=True`，需要创建 `meta/tasks.json`:

```bash
# 在数据集目录下创建
cat > /home/dataset-local/data/megvii_post/tac/plug_charger/meta/tasks.json << 'EOF'
{
    "0": "plug the charger into the socket"
}
EOF
```

### Step 4: 修改训练配置

编辑 `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_restac_plug_charger",
    model=pi0_restac.Pi0_ResTacConfig(
        fusion_dim=512,
        num_cross_attn_heads=8,
        use_sparse_loss=False,
        residual_vqvae_checkpoint="/path/to/vqvae/checkpoint",  # ← 修改为实际路径
    ),
    data=LeRobotResTacDataConfig(
        repo_id="/home/dataset-local/data/megvii_post/tac/plug_charger",  # ← 数据集路径
        base_config=DataConfig(
            prompt_from_task=True,
            norm_stats="assets/norm_stats/plug_charger.json",  # ← Step 2 生成的文件
        ),
    ),
    weight_loader=weight_loaders.Pi0ResTacWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"  # Pi05 基础权重
    ),
    num_train_steps=50_000,
    batch_size=4,
)
```

### Step 5: 启动训练

```bash
cd /home/dataset-local/code/openpi

# 全量微调
python scripts/train.py --config pi05_restac_plug_charger

# 或 LoRA 微调 (显存较少时)
python scripts/train.py --config pi05_restac_plug_charger_lora
```

### 检查清单

| 项目 | 状态 | 说明 |
|------|------|------|
| VQVAE Checkpoint | ❓ | 必须提供，无默认值 |
| Pi05 基础权重 | ✅ | 已配置 GCS 路径 |
| 归一化统计量 | ❓ | 需运行 compute_norm_stats.py |
| 数据集格式 | ✅ | 已是 LeRobot 格式 |
| tasks.json | ❓ | 如需 prompt_from_task 则需创建 |

---

## 10. Token-in-AE 架构详解

### 10.1 设计动机

Token-in-AE 架构将触觉信息作为一个"惊喜 token"直接注入 Action Expert 的输入序列，而不是通过残差结构添加修正。

**核心思想**：
- 触觉信息增益由 gate 值决定（来自 Prophet 的不确定性）
- 高不确定性 → gate ≈ 1 → 使用真实触觉 token
- 低不确定性 → gate ≈ 0 → 使用可学习的默认 embedding（相当于"无触觉"）

### 10.2 数据流

```
触觉图像 + 视觉3视角 + action_prev
        ↓
   ResidualVQVAE (不变)
        ↓
q_event [B, 64] + logvar [B, 1]
        ↓                    ↓
tactile_token_proj      NecessityGate
  Linear(64→1024)            ↓
        ↓               gate [B, 1]
tactile_token [B, 1024]      ↓
        ↓                    ↓
        └──────────┬─────────┘
                   ↓
    surprise_token = gate × tactile_token
                   + (1-gate) × default_embedding
                   [B, 1, 1024]
                   ↓
    拼接到 AE 输入最前面:
    suffix_tokens = [surprise_token, action_0, ..., action_49]
                   [B, 1+action_horizon, 1024]
                   ↓
    Action Expert Forward
    (surprise_token 对所有 action token 可见)
                   ↓
    suffix_out [B, 1+action_horizon, 1024]
                   ↓
    取最后 action_horizon 个: suffix_out[:, -50:]
                   ↓
    action_out_proj → v_t [B, 50, 32]
```

### 10.3 Attention Mask 设计

```python
suffix_ar_mask = [False, True, False, False, ..., False]
                    ↑      ↑     ↑
                surprise  第1个  其余
                 token   action action
```

- **surprise_token (False)**: 双向可见，所有 action token 都能看到它
- **第1个 action token (True)**: AR（自回归），只能看到之前的 token
- **其余 action tokens (False)**: 双向可见

### 10.4 与 ForceVLA-style 对比

| 特性 | ForceVLA-style | Token-in-AE |
|------|----------------|-------------|
| 触觉融合位置 | Transformer 之后 | Transformer 输入 |
| 融合方式 | Self-Attention + 残差 | 直接拼接 |
| 残差结构 | 有 | 无 |
| 新增模块 | tactile_to_vlm_proj, fusion_self_attn | tactile_token_proj, default_embedding |
| Gate 作用 | 缩放 correction 幅度 | 选择真实/默认 token |
| 参数量 | 较多 | 较少 |

### 10.5 配置选项

```python
# config.py 中的配置
TrainConfig(
    name="pi05_restac_token_in_ae",
    model=pi0_restac.Pi0_ResTac_TokenInAEConfig(
        use_sparse_loss=False,          # 是否启用稀疏损失
        sparse_loss_weight=0.01,        # 稀疏损失权重
        residual_vqvae_checkpoint="...", # VQVAE checkpoint 路径（必须）
    ),
    ...
)
```

### 10.6 代码索引

**pi0_restac.py**:
| 功能 | 行号 |
|------|------|
| Pi0_ResTac_TokenInAEConfig | ~1110-1200 |
| Pi0_ResTac_TokenInAE.__init__ | ~1210-1310 |
| encode_tactile_raw | ~1320-1360 |
| compute_surprise_token | ~1365-1400 |
| embed_suffix_with_tactile | ~1440-1490 |
| compute_loss | ~1500-1570 |
| sample_actions | ~1580-1660 |

### 10.7 快速启动

```bash
# 使用 Token-in-AE 架构训练
python scripts/train.py --config pi05_restac_token_in_ae

# 使用 LoRA 微调（低显存）
python scripts/train.py --config pi05_restac_token_in_ae_lora
```

---

**报告完成**
