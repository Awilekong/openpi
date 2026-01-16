# $\pi_{0.5}$ 基础模型架构深度解析

本文档详述了 `pi05_base` 视觉-语言-动作 (VLA) 模型的架构设计、数据流及核心接口维度规格。

## 1. 架构全览

$\pi_{0.5}$ 采用基于 **流匹配 (Flow-matching)** 的 VLA 架构。模型在逻辑上分为 **前缀 (Prefix)** 与 **后缀 (Suffix)** 两部分，两者通过共享的注意力空间进行交互。

*   **前缀 (Encoder/VLM):** **PaliGemma (3B)**。负责将多模态输入编码为高维特征序列。
*   **后缀 (Action Expert):** **Gemma 300M**。负责在 VLM 特征的条件约束下，通过流匹配生成动作序列。

## 2. 数据流与维度规格 (Data Flow & Dimensions)

### 2.1 视觉数据流
图像数据经过 SigLIP 编码器处理，并投影到 VLM 的嵌入空间。

1.  **输入 (Input):**
    *   分辨率: $224 \times 224$ RGB 图像。
    *   数量: $N$ 个视角 (如 base, wrist 等)。
2.  **编码 (SigLIP So400m/14):**
    *   **Patching:** 采用 $14 \times 14$ 的 Patch Size。
    *   **Token 数量:** 每个图像生成 $(224/14) \times (224/14) = 16 \times 16 = \mathbf{256}$ 个 Token。
    *   **原始维度:** SigLIP 输出维度为 1152。
3.  **投影 (Projection):**
    *   通过线性层将维度从 1152 映射至 **2048** (匹配 Gemma 2B 的 `width`)。
    *   **最终输出:** 形状为 $[B, N \times 256, 2048]$ 的视觉 Token 序列。

### 2.2 语言与状态数据流
状态被文本化处理，与任务指令共同构成语言输入。这是 $\pi_{0.5}$ 与 $\pi_0$ 的核心架构差异之一。

#### **$\pi_0$ vs $\pi_{0.5}$ 状态处理对比**
| 特性 | $\pi_0$ (旧版) | $\pi_{0.5}$ (本模型) |
| :--- | :--- | :--- |
| **形式** | **连续向量 (Continuous Vector)** | **离散 Token (Discrete Tokens)** |
| **处理位置** | **后缀 (Suffix)** | **前缀 (Prefix)** |
| **输入方式** | 直接投影后与动作 Embedding 拼接，输入到 Action Expert | 归一化 $\to$ 离散化 $\to$ 文本化 $\to$ Tokenize，输入到 VLM |
| **交互深度** | 浅层交互（仅在 Expert 中可见） | 深层交互（与文本/图像进行全向 Self-Attention） |

1.  **状态处理 (State Processing in $\pi_{0.5}$):**
    *   **归一化:** 机器人本体状态 (关节角等) 归一化至 $[-1, 1]$。
    *   **离散化:** 量化为 256 个分箱 (Bins)，映射为整数索引 $0 \sim 255$。
    *   **文本化:** 整数序列转换为字符串 (如 `"12 45 200..."`)，拼接在任务指令之后。
    *   格式: `Task: {prompt}, State: {state_str};\nAction: `
2.  **标记化 (Tokenization):**
    *   使用 SentencePiece Tokenizer (Vocabulary Size: 257,152)。
    *   **Token 数量:** 动态长度，受配置 `max_token_len` 限制 (通常为 48-200)。
3.  **嵌入 (Embedding):**
    *   **维度:** **2048** (Gemma 2B `width`)。
    *   **最终输出:** 形状为 $[B, L_{text}, 2048]$ 的语言 Token 序列。

## 3. VLM 骨干与特征输出 (Prefix Interface)

VLM (Prefix) 实际上并不输出单一的“特征向量”，而是输出供后缀关注的 **Key/Value Cache**。

*   **模型:** Gemma 2B 架构。
*   **输入序列:** [图像 Token 序列, 文本+状态 Token 序列]。
*   **模型维度 ($D_{model}$):** 2048。
*   **注意力头 ($H$):** 8 头。
*   **Head 维度 ($D_{head}$):** 256。
*   **输出接口 (Interface Output):**
    *   VLM 计算其自身的 Key (K) 和 Value (V)。
    *   **K/V 维度:** $[B, S_{prefix}, 8, 256]$。
    *   这些 K/V 对被保留在 Cache 中，作为 Action Expert 关注的上下文 (Context)。

## 4. 动作专家网络深度拆解 (Action Expert Detail)

Action Expert 是动作生成的执行者，以下是其内部运作的详细拆解。

### 4.1 模型规格 (Model Specification)
*   **基础架构:** **Gemma 300M**。
*   **层数 (Layers):** 18 层。
*   **模型宽度 (Width/Embed Dim):** **1024**。
*   **MLP 维度:** 4096。
*   **注意力头:** 8 头 (Head Dim = 256，与 VLM 2B 对齐)。

### 4.2 输入构成 (Inputs)
Action Expert 的输入由三部分组成：
1.  **动作序列 ($x_t$):**
    *   **维度与空间:** 始终在 **32 维动作空间 (Action Space)** 中。无论是训练时的噪声合成，还是推理时的 $x_t$ 初始化（从高斯噪声开始），其原始维度均为 32。
    *   **处理机制:** 在输入 Action Expert 之前，通过 `action_in_proj` (Linear $32 \to 1024$) 投影至 **1024 维嵌入空间 (Embedding Space)**。
    *   **序列长度:** 等于 **动作视野 (Action Horizon)**，例如 10 或 50。这意味着输入 Expert 的动作 Token 序列形状为 $[B, H_{action}, 1024]$。
2.  **VLM KV Cache (Context):**
    *   来源: 前缀 (Prefix) 的输出。
    *   **Token 数量:** $N_{img} \times 256 + L_{text}$。
        *   例如：1 张图 + 48 文本 Token $\approx 304$ Tokens。
        *   例如：2 张图 + 200 文本 Token $\approx 712$ Tokens。
    *   作用: 作为 Cross-Attention 的 Key 和 Value。
3.  **时间步 ($t$):**
    *   来源: 标量时间步 $t \in [0, 1]$。
    *   处理:
        *   正弦位置编码 (Sinusoidal Embedding)。
        *   MLP 处理 (`time_mlp_in` $\to$ Swish $\to$ `time_mlp_out`)。
        *   **注入方式:** 通过 **AdaRMSNorm** 注入每一层的 LayerNorm 中，调节 Scale 和 Shift 参数。**注意：时间步不作为 Token 拼接到序列中，这是 $\pi_{0.5}$ 的特有设计。**

### 4.3 注意力掩码构建 (Attention Mask Construction)
为了保证自回归生成的一致性，Attention Mask 采用了混合模式：

*   **前缀部分 (Prefix - VLM):** **全向可见 (Bidirectional)**。
    *   图像和文本 Token 之间可以互相“看见”。
    *   `ar_mask` 值为 `False` (0)。
*   **后缀部分 (Suffix - Expert):** **因果可见 (Causal)**。
    *   第 $i$ 个动作 Token 只能看见：
        1.  所有的前缀 Token (VLM Context)。
        2.  自己及之前的动作 Token ($0 \sim i$)。
    *   不能看见未来的动作 Token。
    *   `ar_mask` 值为 `True` (1)。

### 4.4 输出与投影 (Output & Projection)
Action Expert 经过 18 层 Transformer 处理后：

1.  **Token 提取:**
    *   从输出序列中提取最后 $H_{action}$ 个 Token (对应输入的动作序列)。
    *   切片操作: `output[:, -action_horizon:, :]`。
2.  **最终投影:**
    *   经过 `action_out_proj` (Linear $1024 \to 32$)。
    *   **输出:** 预测的速度场向量 $v_t$，维度为 $[B, H_{action}, 32]$。

---

## 总结：接口维度一览

| 模块 | 属性 | 维度/数值 | 备注 |
| :--- | :--- | :--- | :--- |
| **图像** | 原始输入 | $224 \times 224 \times 3$ | |
| **SigLIP** | Patch Token 数量 | 256 | $14 \times 14$ Patch |
| **VLM (Prefix)** | Embedding 维度 | **2048** | Gemma 2B 规格 |
| | Head 数量 | 8 | |
| | Head 维度 | **256** | 交互接口的核心维度 |
| **Action Expert** | Embedding 维度 | **1024** | Gemma 300M 规格 |
| | Head 数量 | 8 | |
| | Head 维度 | **256** | 与 VLM 对齐 |
| **交互** | Q (Expert) | $[B, S_{suffix}, 8, 256]$ | |
| | K, V (Prefix) | $[B, S_{prefix}, 8, 256]$ | |