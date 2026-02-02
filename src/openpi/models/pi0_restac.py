"""
ResTacVLA: ForceVLA-style Tactile Fusion for Vision-Language-Action Models

This module implements the ResTacVLA architecture, which uses a ForceVLA-style fusion
mechanism with Self-Attention and Gate to integrate tactile information:

Fusion pattern: concat([prefix_out, tactile]) → Self-Attention → take_last_50 → Gate → + suffix_out

Key features:
- Self-gated sparse activation (g → 0 when no tactile event)
- ForceVLA-style fusion using Self-Attention instead of LIMoE
- Gate mechanism replaces MOE expert selection
"""

import dataclasses
import logging
from typing import Tuple
import os
import numpy as np

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

# Import for Unit-Align integration (optional)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Unit-Align VQVAE Wrapper
# =============================================================================

class ResidualVQVAEWrapper(nnx.Module if TORCH_AVAILABLE else object):
    """
    Wrapper for Unit-Align's Residual VQVAE.

    Loads a pre-trained Residual VQ-VAE checkpoint and provides:
    - VQ codes extraction (discrete semantic event codes)
    - Logvar extraction (uncertainty from Prophet network)
    """

    def __init__(self, checkpoint_path: str, frozen: bool = True):
        """
        Initialize VQVAE wrapper.

        Args:
            checkpoint_path: Path to Unit-Align residual_vqvae checkpoint
            frozen: Whether to freeze all parameters (inference only)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResidualVQVAEWrapper")

        self.checkpoint_path = checkpoint_path
        self.frozen = frozen
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Import here to avoid circular dependency
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(checkpoint_path), '../..'))

        try:
            from UniT.taming.models.residual_vqmodel import ResidualVQModel
            self.model_class = ResidualVQModel
        except ImportError:
            raise ImportError(
                f"Could not import ResidualVQModel from Unit-Align. "
                f"Make sure Unit-Align is in the Python path."
            )

        self._load_checkpoint()
        logger.info(f"✓ Loaded ResidualVQVAE from {checkpoint_path}")

    def _load_checkpoint(self):
        """Load checkpoint and initialize model."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            hparams = checkpoint.get('hyper_parameters', {})
        else:
            state_dict = checkpoint
            hparams = {}

        # For now, we'll store the state_dict and hparams for later use
        # Full model initialization would require more information
        self.state_dict = state_dict
        self.hparams = hparams
        self.model = None  # Model will be initialized on first use

    def forward(
        self,
        tactile_image: jnp.ndarray,  # [B, H, W, C]
        visual_3views: jnp.ndarray | None,   # [B, 3, 3, 224, 224] 三视角
        action_prev: jnp.ndarray     # [B, 7]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract q_event codes and logvar from tactile observations using Unit-Align VQVAE.

        IMPORTANT: Data format must match Unit-Align's expectations:
        - Prophet输入: 三视角拼接视觉 [B, 3, 3, 224, 224] + action_prev [B, 7]
        - Obs Encoder输入: tactile_image [B, 3, 128, 160]

        Args:
            tactile_image: Tactile image [B, 128, 160, 3] (HWC format from JAX)
            visual_3views: 3-view visual [B, 3, 3, 224, 224] (properly stacked from observation)
            action_prev: Previous action [B, 7]

        Returns:
            q_event_pooled: Pooled VQ event codes [B, 64] (from q_event [B,64,H,W])
            logvar: Prophet网络的log-variance [B, 1]
        """
        # Convert JAX arrays to PyTorch
        tactile_pt = torch.from_numpy(np.asarray(tactile_image)).float().to(self.device)
        action_pt = torch.from_numpy(np.asarray(action_prev)).float().to(self.device)

        # Rearrange tactile to CHW format: [B, H, W, C] → [B, C, H, W]
        if tactile_pt.dim() == 4 and tactile_pt.shape[-1] == 3:
            tactile_pt = tactile_pt.permute(0, 3, 1, 2)  # [B, 3, 128, 160]

        # Handle 3-view visual input
        # visual_3views should be [B, 3, 3, 224, 224] (batch, num_views, channels, height, width)
        if visual_3views is None:
            # Fallback if 3-view visual is not available
            logger.warning("visual_3views is None in ResidualVQVAEWrapper.forward(). Using placeholder.")
            B = tactile_pt.shape[0]
            q_event_pooled = torch.ones(B, 64, device=self.device) * 0.5
            logvar = torch.zeros(B, 1, device=self.device)
        else:
            visual_pt_3views = torch.from_numpy(np.asarray(visual_3views)).float().to(self.device)
            # Convert from [-1, 1] (SigLIP) to ImageNet normalization for VQVAE
            # Step 1: [-1, 1] -> [0, 1]
            visual_pt_3views = (visual_pt_3views + 1.0) / 2.0
            # Step 2: Apply ImageNet normalization: (x - mean) / std
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 1, 3, 1, 1)
            visual_pt_3views = (visual_pt_3views - imagenet_mean) / imagenet_std
            # Verify shape: [B, 3, 3, 224, 224]
            if visual_pt_3views.dim() == 5 and visual_pt_3views.shape[1:] == (3, 3, 224, 224):
                # Correct format, ready for Prophet
                pass
            else:
                logger.warning(
                    f"visual_3views has unexpected shape {visual_pt_3views.shape}. "
                    f"Expected [B, 3, 3, 224, 224]. Using placeholder."
                )
                B = tactile_pt.shape[0]
                q_event_pooled = torch.ones(B, 64, device=self.device) * 0.5
                logvar = torch.zeros(B, 1, device=self.device)

        if visual_3views is not None:
            with torch.no_grad():
                # 准备 Unit-Align ResidualVQModel 的 batch 格式
                batch = {
                    'tactile_image': tactile_pt,      # [B, 3, 128, 160] - Obs Encoder 输入
                    'visual_image': visual_pt_3views,  # [B, 3, 3, 224, 224] - Prophet 输入（三视角）
                    'action_prev': action_pt           # [B, 7] - Prophet 输入（前一动作）
                }

                try:
                    if self.model is None:
                        logger.debug("ResidualVQVAE model not initialized. Using placeholder.")
                        B = tactile_pt.shape[0]
                        q_event_pooled = torch.ones(B, 64, device=self.device) * 0.5
                        logvar = torch.zeros(B, 1, device=self.device)
                    else:
                        # Unit-Align ResidualVQModel forward
                        outputs = self.model(batch)

                        # 提取关键输出
                        q_event = outputs['q_event']  # [B, 64, H, W] - 量化事件表示
                        logvar = outputs['logvar']    # [B, 1] - Prophet的不确定性

                        # q_event 池化：[B, 64, H, W] → [B, 64]
                        # 使用平均池化
                        B = q_event.shape[0]
                        q_event_pooled = q_event.view(B, q_event.shape[1], -1).mean(dim=-1)  # [B, 64]

                except Exception as e:
                    logger.warning(f"Error in ResidualVQVAE forward: {e}. Using placeholder.")
                    B = tactile_pt.shape[0]
                    q_event_pooled = torch.ones(B, 64, device=self.device) * 0.5
                    logvar = torch.zeros(B, 1, device=self.device)

        # 转换回 JAX
        q_event_pooled_jax = jnp.asarray(q_event_pooled.cpu().numpy().astype(np.float32))  # [B, 64]
        logvar_jax = jnp.asarray(logvar.cpu().numpy().astype(np.float32))  # [B, 1]

        return q_event_pooled_jax, logvar_jax


# =============================================================================
# Attention Mask Utilities (from pi0.py)
# =============================================================================

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# =============================================================================
# Cross-Attention Block
# =============================================================================

class CrossAttentionBlock(nnx.Module):
    """
    Cross-modal Cross-Attention module.
    Query comes from one modality, Key/Value come from another modality.

    Args:
        query_dim: Input dimension for query
        kv_dim: Input dimension for key/value
        out_dim: Output dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        rngs: Random number generators
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        rngs: nnx.Rngs = None
    ):
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.dropout_rate = dropout

        # Projection layers
        self.q_proj = nnx.Linear(query_dim, out_dim, rngs=rngs)
        self.k_proj = nnx.Linear(kv_dim, out_dim, rngs=rngs)
        self.v_proj = nnx.Linear(kv_dim, out_dim, rngs=rngs)
        self.out_proj = nnx.Linear(out_dim, out_dim, rngs=rngs)

        # Layer normalization
        self.norm_q = nnx.LayerNorm(query_dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(kv_dim, rngs=rngs)
        self.norm_out = nnx.LayerNorm(out_dim, rngs=rngs)

    def __call__(
        self,
        query: jax.Array,        # [B, Q_len, query_dim]
        key_value: jax.Array,    # [B, KV_len, kv_dim]
        deterministic: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Forward pass of cross-attention.

        Args:
            query: Query tensor [B, Q_len, query_dim]
            key_value: Key/Value tensor [B, KV_len, kv_dim]
            deterministic: Whether to apply dropout

        Returns:
            output: Attention output [B, Q_len, out_dim]
            attn_weights: Attention weights [B, num_heads, Q_len, KV_len]
        """
        # Pre-Norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # Linear projections
        q = self.q_proj(q)   # [B, Q_len, out_dim]
        k = self.k_proj(kv)  # [B, KV_len, out_dim]
        v = self.v_proj(kv)  # [B, KV_len, out_dim]

        # Reshape for multi-head attention
        q = einops.rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Store attention weights for visualization
        attn_weights_out = attn_weights

        # Dropout (if training)
        if not deterministic and self.dropout_rate > 0:
            # Note: In nnx, dropout is handled differently
            dropout_mask = jax.random.bernoulli(
                jax.random.PRNGKey(0), 1 - self.dropout_rate, attn_weights.shape
            )
            attn_weights = jnp.where(dropout_mask, attn_weights / (1 - self.dropout_rate), 0)

        # Apply attention to values
        out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        out = einops.rearrange(out, 'b h l d -> b l (h d)')

        # Output projection + Post-Norm
        out = self.out_proj(out)
        out = self.norm_out(out)

        return out, attn_weights_out


# =============================================================================
# Tactile Encoder (VQVAE-based)
# =============================================================================

class TactileEncoder(nnx.Module):
    """
    Tactile encoder using Unit-Align Residual VQVAE.

    Requires a valid VQVAE checkpoint. Will raise an error if VQVAE is not provided.

    Features:
    - q_event [B, 64] from Unit-Align VQ codebook
    - logvar [B, 1] from Prophet network (uncertainty estimation)
    - Project q_event → fusion_dim (via project_vq layer)

    Args:
        fusion_dim: Target dimension for VQ projection
        vqvae_wrapper: ResidualVQVAEWrapper instance (required)
        rngs: Random number generators
    """

    def __init__(self, fusion_dim: int = 512, vqvae_wrapper=None, rngs: nnx.Rngs = None):
        if vqvae_wrapper is None:
            raise ValueError(
                "TactileEncoder requires a valid VQVAE wrapper. "
                "Please provide 'residual_vqvae_checkpoint' in Pi0_ResTacConfig."
            )

        self.fusion_dim = fusion_dim
        self.vqvae_wrapper = vqvae_wrapper

        # Project VQ codes to fusion space: q_event [B, 64] → [B, fusion_dim]
        self.project_vq = nnx.Linear(64, fusion_dim, rngs=rngs)  # 64 = Unit-Align VQ embed dim

    def __call__(
        self,
        tactile_image: jax.Array,
        visual_3views: jax.Array,
        action_prev: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Encode tactile image to features and logvar using VQVAE.

        Args:
            tactile_image: [B, H, W, C] 触觉图像
            visual_3views: [B, 3, 3, 224, 224] 三视角视觉图像
            action_prev: [B, 7] 前一动作

        Returns:
            features: [B, 1, fusion_dim] 触觉特征 (投影后)
            logvar: [B, 1] 不确定性 (来自Prophet网络)
        """
        # Unit-Align VQVAE encoding
        q_event_pooled, logvar = self.vqvae_wrapper(tactile_image, visual_3views, action_prev)
        # q_event_pooled: [B, 64]

        # Project to fusion_dim: [B, 64] → [B, fusion_dim]
        features = self.project_vq(q_event_pooled)  # [B, fusion_dim]

        # Reshape to [B, 1, fusion_dim] for compatibility with gate and fusion
        features = features[:, None, :]

        return features, logvar  # [B, 1, fusion_dim], [B, 1]


# =============================================================================
# Necessity Gate Network
# =============================================================================

class NecessityGate(nnx.Module):
    """
    Necessity-based Gate: g = necessity(σ)

    The gate output g is positively correlated with sigma (logvar) input.
    Higher uncertainty (larger logvar) means tactile correction is more needed.

    Gate function:
        g = sigmoid((σ - threshold) / temperature)

    Where threshold and temperature are learnable parameters.

    Args:
        rngs: Random number generators
    """

    def __init__(self, rngs: nnx.Rngs = None):
        # Learnable parameters for necessity function
        # Initialize threshold to 0, temperature to 1
        self.sigma_threshold = nnx.Param(jnp.array(0.0))
        self.sigma_temperature = nnx.Param(jnp.array(1.0))

    def __call__(
        self,
        tactile_features: jax.Array,  # [B, tactile_dim] (unused, kept for interface compatibility)
        sigma: jax.Array              # [B, 1]
    ) -> jax.Array:
        """
        Compute gate value based on uncertainty (sigma/logvar).

        Args:
            tactile_features: Tactile features [B, tactile_dim] (unused)
            sigma: Log-variance from tactile encoder [B, 1]

        Returns:
            g: Gate value in [0, 1], shape [B, 1]
        """
        # Necessity: positively correlated with sigma (logvar)
        # Higher logvar (more uncertainty) → higher gate value → more tactile correction
        # Use softplus to ensure positive temperature
        temp = jax.nn.softplus(self.sigma_temperature) + 1e-6
        g = jax.nn.sigmoid((sigma - self.sigma_threshold) / temp)  # [B, 1]

        return g


# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass(frozen=True)
class Pi0_ResTacConfig(_model.BaseModelConfig):
    """Configuration for ResTacVLA model (Pi05-based with tactile fusion)."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Model dimensions (Pi05 defaults)
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200  # Pi05 uses 200 tokens

    # Tactile-specific configuration
    fusion_dim: int = 512               # Dimension of fusion space
    num_cross_attn_heads: int = 8       # Number of attention heads
    cross_attn_dropout: float = 0.1     # Dropout rate

    # Sparsity loss configuration
    use_sparse_loss: bool = False       # Whether to add sparse loss (default: disabled)
    sparse_loss_weight: float = 0.01    # Weight for sparse loss (only used if use_sparse_loss=True)

    # Ablation: Gate mechanism
    use_gate: bool = True               # Whether to use gate modulation (ablation study)

    # Unit-Align Residual VQVAE checkpoint (required)
    # Must provide a valid checkpoint path for tactile encoding
    residual_vqvae_checkpoint: str | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_ResTac":
        return Pi0_ResTac(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    "tactile_0": image_spec,  # Tactile image
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "tactile_0": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                action_prev=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # Previous executed action
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# =============================================================================
# Helper Functions for Visual Input
# =============================================================================

def _extract_and_stack_visual_views(images_dict):
    """
    Extract 3-view visual images from observation and stack them.

    Unit-Align Prophet expects: [B, 3, 3, 224, 224]
    (batch, num_views=3, channels=3, height=224, width=224)

    Args:
        images_dict: Dictionary of image arrays from obs.images

    Returns:
        visual_3views: Stacked 3-view visual [B, 3, 3, 224, 224], or None if views unavailable
    """
    view_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    views = []

    for view_name in view_names:
        view = images_dict.get(view_name, None)
        if view is not None:
            views.append(view)

    if len(views) != 3:
        logger.debug(
            f"Could only find {len(views)}/3 visual views. "
            f"Available: {[k for k in images_dict.keys() if 'rgb' in k]}"
        )
        return None

    # Stack views: [B, 3, 224, 224] × 3 → [B, 3, 3, 224, 224]
    visual_3views = jnp.stack(views, axis=1)
    return visual_3views


# =============================================================================
# Main Model
# =============================================================================

class Pi0_ResTac(_model.BaseModel):
    """
    ResTacVLA: ForceVLA-style Tactile Fusion Model (Pi05-based).

    Uses Pi05 backend with adaRMS for timestep injection.

    Fusion pattern (ForceVLA-style):
        concat([prefix_out, tactile]) → Self-Attention → take_last_50 → Gate → + suffix_out

    Comparison with ForceVLA:
        | Component | ForceVLA | ResTac |
        |-----------|----------|--------|
        | Input | prefix_out + force | prefix_out + tactile |
        | Processing | LIMoE (Self-Attn + MOE) | Self-Attn + Gate |
        | Take tokens | last 50 | last 50 (same) |
        | Modulation | MOE expert selection | Gate switch |
        | Output | + suffix_out | + suffix_out (same) |
    """

    def __init__(self, config: Pi0_ResTacConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # Store config
        self.config = config

        # Get model configs
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Dimension constants
        self.vlm_dim = paligemma_config.width           # 2048
        self.action_dim_internal = action_expert_config.width  # 1024
        self.fusion_dim = config.fusion_dim             # 512

        # ============ PaliGemma (VLM + Image Encoder) ============
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,  # Pi05 uses adaRMS
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ============ Action Projections ============
        self.action_in_proj = nnx.Linear(config.action_dim, self.action_dim_internal, rngs=rngs)
        self.action_out_proj = nnx.Linear(self.action_dim_internal, config.action_dim, rngs=rngs)

        # ============ Time MLP for adaRMS (Pi05) ============
        self.time_mlp_in = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)
        self.time_mlp_out = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)

        # ============ Tactile Encoder (VQVAE-based) ============
        # Requires a valid VQVAE checkpoint - will raise error if not provided
        if not config.residual_vqvae_checkpoint:
            raise ValueError(
                "Pi0_ResTac requires 'residual_vqvae_checkpoint' in config. "
                "Please provide a valid Unit-Align VQVAE checkpoint path."
            )

        try:
            vqvae_wrapper = ResidualVQVAEWrapper(
                checkpoint_path=config.residual_vqvae_checkpoint,
                frozen=True
            )
            logger.info(f"Loaded ResidualVQVAE from {config.residual_vqvae_checkpoint}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ResidualVQVAE from {config.residual_vqvae_checkpoint}: {e}"
            )

        self.tactile_encoder = TactileEncoder(
            fusion_dim=config.fusion_dim,
            vqvae_wrapper=vqvae_wrapper,
            rngs=rngs
        )

        # ============ Necessity Gate Network ============
        # g = necessity(σ) = sigmoid((σ - threshold) / temperature)
        # Higher uncertainty (logvar) → higher gate value → more tactile correction
        self.gate = NecessityGate(rngs=rngs)

        # ============ ForceVLA-style Fusion Modules ============
        # 1. Project tactile from fusion_dim to VLM dimension
        self.tactile_to_vlm_proj = nnx.Linear(config.fusion_dim, self.vlm_dim, rngs=rngs)
        # fusion_dim=512 → vlm_dim=2048

        # 2. Self-Attention for fusion (replaces LIMoE in ForceVLA)
        # Input: [B, S+1, 2048] → Output: [B, S+1, 1024]
        self.fusion_self_attn = nnx.MultiHeadAttention(
            num_heads=config.num_cross_attn_heads,  # 8
            in_features=self.vlm_dim,               # 2048
            out_features=self.action_dim_internal,  # 1024
            decode=False,
            rngs=rngs
        )

        logger.info(f"Pi0_ResTac initialized (ForceVLA-style): vlm_dim={self.vlm_dim}, action_dim={self.action_dim_internal}, fusion_dim={self.fusion_dim}")

    # =========================================================================
    # Tactile Encoding
    # =========================================================================

    def encode_tactile(
        self,
        tactile_image: jax.Array,  # [B, H, W, C]
        visual_3views: jax.Array,  # [B, 3, 3, 224, 224]
        action_prev: jax.Array,    # [B, 7]
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Encode tactile image and compute gate value using NecessityGate.

        The gate value is computed as:
            g = necessity(logvar) = sigmoid((logvar - threshold) / temperature)

        Higher logvar (more uncertainty in visual prediction) → higher gate value.

        Args:
            tactile_image: Tactile image [B, H, W, C]
            visual_3views: 3-view visual [B, 3, 3, 224, 224]
            action_prev: Previous action [B, 7]

        Returns:
            tactile_features: Features in fusion space [B, 1, fusion_dim]
            gate_value: Gate value g ∈ [0, 1], shape [B, 1]
            logvar: Log-variance (uncertainty) [B, 1]
        """
        # 1. Encode tactile image (returns features already in fusion_dim and logvar)
        tactile_features, logvar = self.tactile_encoder(tactile_image, visual_3views, action_prev)
        # tactile_features: [B, 1, fusion_dim], logvar: [B, 1]

        # 2. Compute gate value based on uncertainty (logvar)
        # g = necessity(logvar), higher uncertainty → higher gate
        features_flat = tactile_features.squeeze(1)  # [B, fusion_dim]
        gate_value = self.gate(features_flat, logvar)  # [B, 1]

        return tactile_features, gate_value, logvar

    # =========================================================================
    # ForceVLA-style Tactile Fusion
    # =========================================================================

    def tactile_fusion_forcevla_style(
        self,
        tactile_features: jax.Array,   # [B, 1, fusion_dim=512]
        prefix_out: jax.Array,         # [B, S_prefix, vlm_dim=2048]
        suffix_out: jax.Array,         # [B, S_suffix, action_dim_internal=1024]
        gate_value: jax.Array,         # [B, 1]
        deterministic: bool = True
    ) -> jax.Array:
        """
        ForceVLA-style tactile fusion with gate instead of MOE.

        Pattern: concat([prefix_out, tactile]) → self-attn → take_last_50 → gate → add

        Data flow:
            prefix_out [B, S, 2048]  +  tactile_proj [B, 1, 2048]
                            │
                            ▼ concat
                    fused [B, S+1, 2048]
                            │
                            ▼ Self-Attention (replaces LIMoE)
                    attn_out [B, S+1, 1024]
                            │
                            ▼ take last 50 (ForceVLA style)
                    correction [B, 50, 1024]
                            │
                            ▼ × gate (replaces MOE)
                    gated_correction [B, 50, 1024]

        Args:
            tactile_features: Encoded tactile features [B, 1, fusion_dim=512]
            prefix_out: VLM output (visual + language context) [B, S_prefix, vlm_dim=2048]
            suffix_out: Action Expert output [B, S_suffix, action_dim_internal=1024]
            gate_value: Gate value g [B, 1]
            deterministic: Whether to apply dropout

        Returns:
            gated_correction: Gated correction [B, action_horizon, action_dim_internal]
        """
        # 1. Project tactile to VLM dimension
        tactile_vlm = self.tactile_to_vlm_proj(tactile_features)  # [B, 1, 2048]

        # 2. Concatenate [prefix_out, tactile]
        fused = jnp.concatenate([prefix_out, tactile_vlm], axis=1)  # [B, S+1, 2048]

        # 3. Self-Attention (ForceVLA uses LIMoE, we use simple self-attn)
        attn_out = self.fusion_self_attn(fused, deterministic=deterministic)
        # [B, S+1, 1024]

        # 4. Take last action_horizon tokens (ForceVLA style)
        correction = attn_out[:, -self.action_horizon:, :]  # [B, 50, 1024]

        # 5. Gate (replaces MOE) - optional based on config
        if self.config.use_gate:
            gate_expanded = gate_value[:, :, None]  # [B, 1, 1]
            correction = gate_expanded * correction  # [B, 50, 1024]

        return correction

    # =========================================================================
    # Embed Prefix (from Pi0)
    # =========================================================================

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed visual and language inputs (prefix)."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images (excluding tactile)
        for name in obs.images:
            if name == "tactile_0":
                continue  # Skip tactile image, handle separately
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Add language (tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    # =========================================================================
    # Embed Suffix (Pi05-based)
    # =========================================================================

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"]
    ) -> Tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],         # adarms_cond
        at.Float[at.Array, "b 1 fusion"],    # tactile_features
        at.Float[at.Array, "b 1"],           # gate_value
        at.Float[at.Array, "b 1"]            # logvar
    ]:
        """
        Embed suffix (actions) and extract tactile features.

        Pi05: No state token in suffix (state is discretized in prefix).
        Uses time_mlp for adaRMS conditioning.

        Returns:
            tokens: Suffix tokens [B, action_horizon, action_dim_internal]
            input_mask: Input mask [B, action_horizon]
            ar_mask: Autoregressive mask [action_horizon]
            adarms_cond: adaRMS conditioning [B, action_dim_internal]
            tactile_features: Tactile features [B, 1, fusion_dim]
            gate_value: Gate value [B, 1]
            logvar: Log-variance from tactile encoder [B, 1]
        """
        # 1. Extract and encode tactile image
        # For VQVAE, we need:
        # - tactile_image: current tactile observation
        # - visual_image: 3-view visual observations (stacked)
        # - action_prev: previous executed action (state_t - state_t-1)

        tactile_image = obs.images.get("tactile_0", None)
        # Extract and stack 3-view visual for Prophet input: [B, 3, 3, 224, 224]
        visual_3views = _extract_and_stack_visual_views(obs.images)

        # Get action_prev from observation (set by ResTacInputs transform)
        action_prev = obs.action_prev if obs.action_prev is not None else jnp.zeros((obs.state.shape[0], self.action_dim))

        if tactile_image is not None:
            tactile_features, gate_value, logvar = self.encode_tactile(tactile_image, visual_3views, action_prev)
        else:
            # If no tactile image, use zeros
            batch_size = obs.state.shape[0]
            tactile_features = jnp.zeros((batch_size, 1, self.fusion_dim))
            gate_value = jnp.zeros((batch_size, 1))
            logvar = jnp.zeros((batch_size, 1))

        # 2. Action tokens
        action_tokens = self.action_in_proj(noisy_actions)

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
        adarms_cond = nnx.swish(time_emb)

        # 4. Build masks
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + ([False] * (self.action_horizon - 1)))

        return action_tokens, input_mask, ar_mask, adarms_cond, tactile_features, gate_value, logvar

    # =========================================================================
    # Compute Loss
    # =========================================================================

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """
        Compute loss including:
        1. Flow Matching Loss (main loss)
        2. Sparsity Loss (regularization to force g → 0)

        Returns:
            loss: Per-timestep loss [B, action_horizon]
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        # Flow matching targets
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ============ Encode Prefix (visual + language) ============
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # ============ Encode Suffix (state + actions) + Extract Tactile ============
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, tactile_features, gate_value, logvar = \
            self.embed_suffix(observation, x_t, time)

        # ============ Transformer Forward ============
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond]  # Pi05: pass adaRMS conditioning
        )

        # ============ ForceVLA-style Tactile Fusion ============
        tactile_correction = self.tactile_fusion_forcevla_style(
            tactile_features=tactile_features,
            prefix_out=prefix_out,
            suffix_out=suffix_out,
            gate_value=gate_value,
            deterministic=not train
        )

        # ============ Residual + Output ============
        # Add tactile correction to suffix_out's last action_horizon tokens
        action_out = suffix_out[:, -self.action_horizon:] + tactile_correction
        v_t = self.action_out_proj(action_out)

        # ============ Compute Losses ============
        # 1. Flow Matching Loss (main loss)
        flow_loss = jnp.square(v_t - u_t)  # [B, 50, action_dim]
        total_loss = jnp.mean(flow_loss, axis=-1)  # [B, 50]

        # 2. Sparsity Loss (optional, disabled by default)
        # Encourages gate to stay closed when tactile correction is not needed
        if self.config.use_sparse_loss:
            # gate_value: [B, 1], take mean across batch to get scalar
            sparse_loss = jnp.mean(gate_value)
            total_loss = total_loss + self.config.sparse_loss_weight * sparse_loss

        return total_loss

    # =========================================================================
    # Sample Actions (Inference)
    # =========================================================================

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        Sample actions using diffusion reverse process.
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ============ Cache Prefix ============
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out_cached, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions
        )

        # Extract tactile features once (doesn't change during sampling)
        tactile_image = observation.images.get("tactile_0", None)
        if tactile_image is None:
            raise ValueError("Pi0_ResTac requires tactile image ('tactile_0') in observation.images")

        # Extract and stack 3-view visual for VQVAE Prophet input: [B, 3, 3, 224, 224]
        visual_3views = _extract_and_stack_visual_views(observation.images)
        if visual_3views is None:
            raise ValueError("Pi0_ResTac requires 3 visual views (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)")

        # Get action_prev from observation (previous executed action)
        action_prev = observation.action_prev
        if action_prev is None:
            raise ValueError("Pi0_ResTac requires action_prev in observation")

        # Encode tactile (VQVAE-based)
        tactile_features, gate_value, logvar = self.encode_tactile(tactile_image, visual_3views, action_prev)

        def step(carry):
            x_t, time = carry

            # Encode suffix (returns 7 values including adarms_cond and logvar)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, _, _, _ = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # Build attention mask for suffix
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # Transformer forward (using cached prefix)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond]  # Pi05: pass adaRMS conditioning
            )

            # ForceVLA-style tactile fusion
            tactile_correction = self.tactile_fusion_forcevla_style(
                tactile_features=tactile_features,
                prefix_out=prefix_out_cached,
                suffix_out=suffix_out,
                gate_value=gate_value,
                deterministic=True
            )

            # Residual + output
            action_out = suffix_out[:, -self.action_horizon:] + tactile_correction
            v_t = self.action_out_proj(action_out)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


# =============================================================================
# Token-in-AE Configuration
# =============================================================================

@dataclasses.dataclass(frozen=True)
class Pi0_ResTac_TokenInAEConfig(_model.BaseModelConfig):
    """
    Configuration for ResTacVLA Token-in-AE variant.

    This variant injects tactile information as a surprise token directly into
    the Action Expert input, rather than using ForceVLA-style fusion.

    Key differences from Pi0_ResTacConfig:
    - Tactile token is projected to AE dimension (1024) and prepended to suffix
    - Uses gate-based fusion: gate × tactile_token + (1-gate) × default_embedding
    - No cross-attention or residual structure
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Model dimensions (Pi05 defaults)
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200

    # Sparsity loss configuration
    use_sparse_loss: bool = False
    sparse_loss_weight: float = 0.01

    # Unit-Align Residual VQVAE checkpoint (required)
    residual_vqvae_checkpoint: str | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_ResTac_TokenInAE":
        return Pi0_ResTac_TokenInAE(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    "tactile_0": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "tactile_0": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                action_prev=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# =============================================================================
# Token-in-AE Model
# =============================================================================

class Pi0_ResTac_TokenInAE(_model.BaseModel):
    """
    ResTacVLA Token-in-AE: Tactile Token Injection into Action Expert.

    This model injects tactile information as a "surprise token" directly into
    the Action Expert's input sequence, rather than using ForceVLA-style fusion.

    Architecture:
        1. VQVAE extracts q_event [B, 64] and logvar [B, 1] (unchanged)
        2. NecessityGate computes gate value from logvar (unchanged)
        3. Project q_event to AE dimension: tactile_token [B, 1, 1024]
        4. Compute surprise token: gate × tactile_token + (1-gate) × default_embedding
        5. Prepend surprise token to action tokens: [surprise, action_0, ..., action_49]
        6. Action Expert processes the full sequence
        7. Extract last action_horizon tokens as output (no residual)

    Key properties:
        - gate=1 (high uncertainty): use real tactile token
        - gate=0 (low uncertainty): use learned default embedding (no tactile info)
        - Surprise token is visible to all action tokens (ar_mask=False)
    """

    def __init__(self, config: Pi0_ResTac_TokenInAEConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.config = config

        # Get model configs
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Dimension constants
        self.vlm_dim = paligemma_config.width           # 2048
        self.action_dim_internal = action_expert_config.width  # 1024

        # ============ PaliGemma (VLM + Image Encoder) ============
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ============ Action Projections ============
        self.action_in_proj = nnx.Linear(config.action_dim, self.action_dim_internal, rngs=rngs)
        self.action_out_proj = nnx.Linear(self.action_dim_internal, config.action_dim, rngs=rngs)

        # ============ Time MLP for adaRMS (Pi05) ============
        self.time_mlp_in = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)
        self.time_mlp_out = nnx.Linear(self.action_dim_internal, self.action_dim_internal, rngs=rngs)

        # ============ VQVAE Wrapper ============
        if not config.residual_vqvae_checkpoint:
            raise ValueError(
                "Pi0_ResTac_TokenInAE requires 'residual_vqvae_checkpoint' in config. "
                "Please provide a valid Unit-Align VQVAE checkpoint path."
            )

        try:
            self.vqvae_wrapper = ResidualVQVAEWrapper(
                checkpoint_path=config.residual_vqvae_checkpoint,
                frozen=True
            )
            logger.info(f"Loaded ResidualVQVAE from {config.residual_vqvae_checkpoint}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ResidualVQVAE from {config.residual_vqvae_checkpoint}: {e}"
            )

        # ============ Necessity Gate Network ============
        self.gate = NecessityGate(rngs=rngs)

        # ============ Token-in-AE Specific Components ============
        # Project q_event [B, 64] → tactile_token [B, 1024]
        self.tactile_token_proj = nnx.Linear(64, self.action_dim_internal, rngs=rngs)

        # Learnable default embedding for when gate=0 (no tactile event)
        # Initialized to zeros, will be learned during training
        self.default_tactile_embedding = nnx.Param(
            jnp.zeros((1, self.action_dim_internal))  # [1, 1024]
        )

        logger.info(
            f"Pi0_ResTac_TokenInAE initialized: "
            f"vlm_dim={self.vlm_dim}, action_dim_internal={self.action_dim_internal}"
        )

    # =========================================================================
    # Tactile Encoding (returns raw q_event, not fusion_dim features)
    # =========================================================================

    def encode_tactile_raw(
        self,
        tactile_image: jax.Array,
        visual_3views: jax.Array,
        action_prev: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Encode tactile image and compute gate value.

        Returns raw q_event (not projected) and gate value.

        Args:
            tactile_image: [B, H, W, C]
            visual_3views: [B, 3, 3, 224, 224]
            action_prev: [B, 7]

        Returns:
            q_event: Raw VQ event codes [B, 64]
            gate_value: Gate value [B, 1]
            logvar: Log-variance [B, 1]
        """
        # Get q_event and logvar from VQVAE
        q_event, logvar = self.vqvae_wrapper.forward(tactile_image, visual_3views, action_prev)
        # q_event: [B, 64], logvar: [B, 1]

        # Compute gate value based on uncertainty
        gate_value = self.gate(q_event, logvar)  # [B, 1]

        return q_event, gate_value, logvar

    def compute_surprise_token(
        self,
        q_event: jax.Array,      # [B, 64]
        gate_value: jax.Array,   # [B, 1]
    ) -> jax.Array:
        """
        Compute surprise tactile token using gate-based fusion.

        Formula: surprise_token = gate × tactile_token + (1-gate) × default_embedding

        Args:
            q_event: Raw VQ event codes [B, 64]
            gate_value: Gate value [B, 1]

        Returns:
            surprise_token: Fused tactile token [B, 1, 1024]
        """
        # Project q_event to AE dimension: [B, 64] → [B, 1024]
        tactile_token = self.tactile_token_proj(q_event)  # [B, 1024]

        # Get default embedding (broadcast to batch size)
        batch_size = q_event.shape[0]
        default_emb = jnp.broadcast_to(
            self.default_tactile_embedding.value,  # [1, 1024]
            (batch_size, self.action_dim_internal)  # [B, 1024]
        )

        # Gate-based fusion: gate × tactile + (1-gate) × default
        # gate_value: [B, 1], tactile_token: [B, 1024], default_emb: [B, 1024]
        surprise_token = gate_value * tactile_token + (1 - gate_value) * default_emb
        # [B, 1024]

        # Reshape to [B, 1, 1024] for concatenation
        surprise_token = surprise_token[:, None, :]

        return surprise_token

    # =========================================================================
    # Embed Prefix (same as Pi0_ResTac)
    # =========================================================================

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed visual and language inputs (prefix)."""
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            if name == "tactile_0":
                continue
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    # =========================================================================
    # Embed Suffix (Token-in-AE version)
    # =========================================================================

    def embed_suffix_with_tactile(
        self,
        obs: _model.Observation,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        surprise_token: jax.Array,  # [B, 1, 1024]
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Embed suffix with surprise tactile token prepended.

        Sequence: [surprise_token, action_0, action_1, ..., action_{H-1}]
        Total length: 1 + action_horizon

        Args:
            obs: Observation
            noisy_actions: Noisy action sequence [B, action_horizon, action_dim]
            timestep: Diffusion timestep [B]
            surprise_token: Tactile surprise token [B, 1, 1024]

        Returns:
            suffix_tokens: [B, 1+action_horizon, 1024]
            suffix_mask: [B, 1+action_horizon]
            suffix_ar_mask: [1+action_horizon]
            adarms_cond: [B, 1024]
        """
        # 1. Project action tokens
        action_tokens = self.action_in_proj(noisy_actions)  # [B, action_horizon, 1024]

        # 2. Prepend surprise token
        # suffix_tokens = [surprise_token, action_tokens]
        suffix_tokens = jnp.concatenate([surprise_token, action_tokens], axis=1)
        # [B, 1 + action_horizon, 1024]

        # 3. Time embedding for adaRMS
        time_emb = posemb_sincos(
            timestep,
            self.action_dim_internal,
            min_period=4e-3,
            max_period=4.0
        )
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = nnx.swish(time_emb)  # [B, 1024]

        # 4. Build masks
        # Input mask: all ones (all tokens are valid)
        suffix_mask = jnp.ones((suffix_tokens.shape[0], suffix_tokens.shape[1]), dtype=jnp.bool_)

        # AR mask: [False, True, False, False, ...]
        # - surprise_token: False (bidirectional, visible to all)
        # - first action token: True (AR)
        # - remaining action tokens: False (bidirectional within suffix)
        suffix_ar_mask = jnp.array(
            [False] +  # surprise token: visible to all
            [True] +   # first action token: AR
            [False] * (self.action_horizon - 1)  # remaining: bidirectional
        )

        return suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond

    # =========================================================================
    # Compute Loss
    # =========================================================================

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """
        Compute flow matching loss.

        Returns:
            loss: Per-timestep loss [B, action_horizon]
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        # Flow matching targets
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ============ Extract Tactile Information ============
        tactile_image = observation.images.get("tactile_0", None)
        visual_3views = _extract_and_stack_visual_views(observation.images)
        action_prev = observation.action_prev if observation.action_prev is not None else jnp.zeros((observation.state.shape[0], self.action_dim))

        if tactile_image is not None:
            q_event, gate_value, logvar = self.encode_tactile_raw(tactile_image, visual_3views, action_prev)
            surprise_token = self.compute_surprise_token(q_event, gate_value)
        else:
            batch_size = observation.state.shape[0]
            # Use default embedding when no tactile
            surprise_token = jnp.broadcast_to(
                self.default_tactile_embedding.value[None, :, :],
                (batch_size, 1, self.action_dim_internal)
            )
            gate_value = jnp.zeros((batch_size, 1))

        # ============ Encode Prefix ============
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # ============ Encode Suffix (with surprise token) ============
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_tactile(
            observation, x_t, time, surprise_token
        )

        # ============ Transformer Forward ============
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond]
        )

        # ============ Extract Action Output ============
        # suffix_out: [B, 1+action_horizon, 1024]
        # Take last action_horizon tokens (skip surprise token)
        action_out = suffix_out[:, -self.action_horizon:, :]  # [B, action_horizon, 1024]
        v_t = self.action_out_proj(action_out)  # [B, action_horizon, action_dim]

        # ============ Compute Loss ============
        flow_loss = jnp.square(v_t - u_t)
        total_loss = jnp.mean(flow_loss, axis=-1)

        # Optional sparsity loss
        if self.config.use_sparse_loss:
            sparse_loss = jnp.mean(gate_value)
            total_loss = total_loss + self.config.sparse_loss_weight * sparse_loss

        return total_loss

    # =========================================================================
    # Sample Actions
    # =========================================================================

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using diffusion reverse process."""
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ============ Cache Prefix ============
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out_cached, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions
        )

        # ============ Compute Surprise Token (once) ============
        tactile_image = observation.images.get("tactile_0", None)
        if tactile_image is None:
            raise ValueError("Pi0_ResTac_TokenInAE requires tactile image ('tactile_0')")

        visual_3views = _extract_and_stack_visual_views(observation.images)
        if visual_3views is None:
            raise ValueError("Requires 3 visual views (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)")

        action_prev = observation.action_prev
        if action_prev is None:
            raise ValueError("Requires action_prev in observation")

        q_event, gate_value, logvar = self.encode_tactile_raw(tactile_image, visual_3views, action_prev)
        surprise_token = self.compute_surprise_token(q_event, gate_value)
        # [B, 1, 1024]

        def step(carry):
            x_t, time = carry

            # Encode suffix with surprise token
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_tactile(
                observation, x_t, jnp.broadcast_to(time, batch_size), surprise_token
            )

            # Build attention mask
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # Transformer forward
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond]
            )

            # Extract action output (last action_horizon tokens)
            action_out = suffix_out[:, -self.action_horizon:, :]
            v_t = self.action_out_proj(action_out)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
