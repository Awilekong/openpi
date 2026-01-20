"""
ResTacVLA: Two-Stage Cross-Attention Tactile Fusion for Vision-Language-Action Models

This module implements the ResTacVLA architecture, which uses a two-stage cross-attention
mechanism to fuse tactile information with visual and action features:

Stage 1: Tactile queries visual context (prefix_out) to understand "what was touched"
Stage 2: Enriched tactile queries action features (suffix_out) to determine "how to correct"

Key features:
- Self-gated sparse activation (g → 0 when no tactile event)
- Explicit semantic understanding before action correction
- Interpretable attention weights for visualization
"""

import dataclasses
import logging
from typing import Tuple

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
# Tactile Encoder Interface (Placeholder)
# =============================================================================

class TactileEncoderPlaceholder(nnx.Module):
    """
    Placeholder for tactile encoder.
    To be replaced with a real pre-trained tactile encoder later.

    The real encoder should output:
    - features: [B, 1, output_dim] - tactile features
    - sigma: [B, 1] - standard deviation indicating tactile importance

    Args:
        output_dim: Output feature dimension
        rngs: Random number generators
    """

    def __init__(self, output_dim: int, rngs: nnx.Rngs = None):
        self.output_dim = output_dim
        # Placeholder: simple MLP that processes flattened tactile image
        self.proj = nnx.Linear(224 * 224 * 3, output_dim, rngs=rngs)
        # Placeholder sigma estimator
        self.sigma_proj = nnx.Linear(224 * 224 * 3, 1, rngs=rngs)

    def __call__(self, tactile_image: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Encode tactile image to features and sigma.

        Args:
            tactile_image: [B, H, W, C] tactile image (assumed 224x224x3)

        Returns:
            features: [B, 1, output_dim] tactile features
            sigma: [B, 1] standard deviation (importance indicator)
        """
        batch_size = tactile_image.shape[0]
        # Flatten
        x = tactile_image.reshape(batch_size, -1)
        # Project to features
        features = self.proj(x)
        # Estimate sigma (use softplus to ensure positive)
        sigma = jax.nn.softplus(self.sigma_proj(x))  # [B, 1]
        return features[:, None, :], sigma  # [B, 1, output_dim], [B, 1]


# =============================================================================
# Factorized Gate Network
# =============================================================================

class FactorizedGate(nnx.Module):
    """
    Factorized Gate: g = necessity(σ) × modulation(features)

    The gate output g is positively correlated with sigma input, while
    the modulation network learns task-specific adjustments.

    Components:
    - necessity(σ) = sigmoid((σ - threshold) / temperature)
      - Ensures g is positively correlated with sigma
      - threshold and temperature are learnable parameters
    - modulation(features) = sigmoid(MLP(features))
      - Learns when to activate based on tactile content

    Args:
        tactile_dim: Dimension of tactile features
        hidden_dim: Hidden dimension for modulation MLP
        rngs: Random number generators
    """

    def __init__(
        self,
        tactile_dim: int,
        hidden_dim: int = 256,
        rngs: nnx.Rngs = None
    ):
        # Learnable parameters for necessity function
        # Initialize threshold to 0, temperature to 1
        self.sigma_threshold = nnx.Param(jnp.array(0.0))
        self.sigma_temperature = nnx.Param(jnp.array(1.0))

        # Modulation MLP
        self.mod_mlp = nnx.Linear(tactile_dim, hidden_dim, rngs=rngs)
        self.mod_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        tactile_features: jax.Array,  # [B, tactile_dim]
        sigma: jax.Array              # [B, 1]
    ) -> jax.Array:
        """
        Compute factorized gate value.

        Args:
            tactile_features: Tactile features [B, tactile_dim]
            sigma: Standard deviation from tactile encoder [B, 1]

        Returns:
            g: Gate value in [0, 1], shape [B, 1]
        """
        # Necessity: positively correlated with sigma
        # Use softplus to ensure positive temperature
        temp = jax.nn.softplus(self.sigma_temperature) + 1e-6
        necessity = jax.nn.sigmoid((sigma - self.sigma_threshold) / temp)  # [B, 1]

        # Modulation: learned from features
        h = nnx.relu(self.mod_mlp(tactile_features))  # [B, hidden_dim]
        modulation = jax.nn.sigmoid(self.mod_out(h))   # [B, 1]

        # Factorized gate: g = necessity × modulation
        g = necessity * modulation

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
    tactile_encoder_dim: int = 512      # Output dim of tactile encoder (placeholder)
    fusion_dim: int = 512               # Dimension of fusion space
    num_cross_attn_heads: int = 8       # Number of attention heads
    cross_attn_dropout: float = 0.1     # Dropout rate

    # Sparsity loss weight
    sparse_loss_weight: float = 0.01

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
# Main Model
# =============================================================================

class Pi0_ResTac(_model.BaseModel):
    """
    ResTacVLA: Two-Stage Cross-Attention Tactile Fusion Model (Pi05-based).

    Uses Pi05 backend with adaRMS for timestep injection.

    Stage 1: Tactile Query × prefix_out (VLM) → Understand "what was touched"
    Stage 2: Enriched tactile Query × suffix_out (Action) → Determine "how to correct"
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

        # ============ Tactile Encoder (Placeholder) ============
        # TODO: Replace with real pre-trained tactile encoder
        self.tactile_encoder = TactileEncoderPlaceholder(
            output_dim=config.tactile_encoder_dim,
            rngs=rngs
        )

        # Tactile projection to fusion space
        self.tactile_proj = nnx.Linear(config.tactile_encoder_dim, config.fusion_dim, rngs=rngs)

        # ============ Factorized Gate Network ============
        # g = necessity(σ) × modulation(features)
        # necessity ensures positive correlation with sigma
        # modulation learns task-specific adjustments
        self.gate = FactorizedGate(
            tactile_dim=config.tactile_encoder_dim,
            hidden_dim=256,
            rngs=rngs
        )

        # ============ Stage 1: Cross-Attention (Tactile × VLM Context) ============
        # Tactile as Query, visual context (prefix_out) as Key/Value
        self.stage1_cross_attn = CrossAttentionBlock(
            query_dim=config.fusion_dim,
            kv_dim=self.vlm_dim,           # 2048
            out_dim=config.fusion_dim,
            num_heads=config.num_cross_attn_heads,
            dropout=config.cross_attn_dropout,
            rngs=rngs
        )

        # ============ Stage 2: Cross-Attention (Enriched Tactile × Action Features) ============
        # Stage 1 output as Query, action features (suffix_out) as Key/Value
        self.stage2_cross_attn = CrossAttentionBlock(
            query_dim=config.fusion_dim,
            kv_dim=self.action_dim_internal,   # 1024
            out_dim=self.action_dim_internal,  # Output same as suffix_out for residual
            num_heads=config.num_cross_attn_heads,
            dropout=config.cross_attn_dropout,
            rngs=rngs
        )

        logger.info(f"Pi0_ResTac initialized: vlm_dim={self.vlm_dim}, action_dim={self.action_dim_internal}, fusion_dim={self.fusion_dim}")

    # =========================================================================
    # Tactile Encoding
    # =========================================================================

    def encode_tactile(
        self,
        tactile_image: jax.Array,  # [B, H, W, C]
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Encode tactile image and compute gate value using Factorized Gate.

        The gate value is computed as:
            g = necessity(σ) × modulation(features)

        Where σ (sigma) comes from the tactile encoder and indicates tactile importance.

        Args:
            tactile_image: Tactile image [B, H, W, C]

        Returns:
            tactile_features: Features in fusion space [B, 1, fusion_dim]
            gate_value: Gate value g ∈ [0, 1], shape [B, 1]
        """
        # 1. Encode tactile image (returns features and sigma)
        raw_features, sigma = self.tactile_encoder(tactile_image)
        # raw_features: [B, 1, tactile_encoder_dim], sigma: [B, 1]
        raw_features_flat = raw_features.squeeze(1)  # [B, tactile_encoder_dim]

        # 2. Project to fusion space
        tactile_features = self.tactile_proj(raw_features_flat)  # [B, fusion_dim]
        tactile_features = tactile_features[:, None, :]  # [B, 1, fusion_dim]

        # 3. Compute factorized gate value
        # g = necessity(σ) × modulation(features)
        gate_value = self.gate(raw_features_flat, sigma)  # [B, 1]

        return tactile_features, gate_value

    # =========================================================================
    # Two-Stage Tactile Fusion
    # =========================================================================

    def two_stage_tactile_fusion(
        self,
        tactile_features: jax.Array,   # [B, 1, fusion_dim]
        prefix_out: jax.Array,         # [B, S_prefix, vlm_dim]
        suffix_out: jax.Array,         # [B, S_suffix, action_dim_internal]
        gate_value: jax.Array,         # [B, 1]
        deterministic: bool = True
    ) -> jax.Array:
        """
        Two-stage Cross-Attention tactile fusion.

        Stage 1: Tactile Query × prefix_out → Semantic understanding
        Stage 2: Enriched tactile Query × suffix_out → Action correction

        Args:
            tactile_features: Encoded tactile features [B, 1, fusion_dim]
            prefix_out: VLM output (visual + language context) [B, S_prefix, vlm_dim]
            suffix_out: Action Expert output [B, S_suffix, action_dim_internal]
            gate_value: Gate value g [B, 1]
            deterministic: Whether to apply dropout

        Returns:
            gated_correction: Gated correction [B, action_horizon, action_dim_internal]
        """
        # ========== Stage 1: Semantic Understanding ==========
        # Tactile asks visual context: "What did I touch?"
        enriched_tactile, stage1_attn = self.stage1_cross_attn(
            query=tactile_features,      # [B, 1, fusion_dim]
            key_value=prefix_out,        # [B, S_prefix, 2048]
            deterministic=deterministic
        )  # [B, 1, fusion_dim]

        # ========== Stage 2: Action Correction ==========
        # Extract action features (last action_horizon tokens of suffix_out)
        action_features = suffix_out[:, -self.action_horizon:, :]  # [B, 50, 1024]

        # Enriched tactile asks action: "How should I correct?"
        correction, stage2_attn = self.stage2_cross_attn(
            query=enriched_tactile,      # [B, 1, fusion_dim]
            key_value=action_features,   # [B, 50, 1024]
            deterministic=deterministic
        )  # [B, 1, 1024]

        # ========== Broadcast + Gate ==========
        # Broadcast to all timesteps
        correction = einops.repeat(
            correction,
            'b 1 d -> b t d',
            t=self.action_horizon
        )  # [B, 50, 1024]

        # Apply gate
        gate_expanded = gate_value[:, :, None]  # [B, 1, 1]
        gated_correction = gate_expanded * correction  # [B, 50, 1024]

        return gated_correction

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
        at.Float[at.Array, "b 1"]            # gate_value
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
        """
        # 1. Extract and encode tactile image
        tactile_image = obs.images.get("tactile_0", None)
        if tactile_image is not None:
            tactile_features, gate_value = self.encode_tactile(tactile_image)
        else:
            # If no tactile image, use zeros
            batch_size = obs.state.shape[0]
            tactile_features = jnp.zeros((batch_size, 1, self.fusion_dim))
            gate_value = jnp.zeros((batch_size, 1))

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

        return action_tokens, input_mask, ar_mask, adarms_cond, tactile_features, gate_value

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
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, tactile_features, gate_value = \
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

        # ============ Two-Stage Tactile Fusion ============
        tactile_correction = self.two_stage_tactile_fusion(
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
        # 1. Flow Matching Loss
        flow_loss = jnp.square(v_t - u_t)  # [B, 50, action_dim]

        # 2. Sparsity Loss (force g → 0)
        sparse_loss = jnp.mean(gate_value)  # scalar

        # Combine losses
        # Note: Original framework expects [B, action_horizon] shape
        # Add sparse_loss to mean of each timestep's loss
        total_loss = jnp.mean(flow_loss, axis=-1) + self.config.sparse_loss_weight * sparse_loss

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
        if tactile_image is not None:
            tactile_features, gate_value = self.encode_tactile(tactile_image)
        else:
            tactile_features = jnp.zeros((batch_size, 1, self.fusion_dim))
            gate_value = jnp.zeros((batch_size, 1))

        def step(carry):
            x_t, time = carry

            # Encode suffix (now returns 6 values including adarms_cond)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, _, _ = self.embed_suffix(
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

            # Two-stage tactile fusion
            tactile_correction = self.two_stage_tactile_fusion(
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
