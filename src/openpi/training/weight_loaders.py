import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0ResTacWeightLoader(WeightLoader):
    """Loads weights from a checkpoint for ResTacVLA models.

    Loads base Pi0/Pi05 weights and preserves new tactile-related parameters.

    Supported models:
    1. Pi0_ResTac (ForceVLA-style fusion):
       - tactile_encoder, tactile_to_vlm_proj: Tactile encoding and projection
       - fusion_self_attn: Self-attention for ForceVLA-style fusion
       - gate (NecessityGate): Gate network (sigma_threshold, sigma_temperature)

    2. Pi0_ResTac_TokenInAE (Token injection into AE):
       - tactile_token_proj: Projects q_event [64] to AE dimension [1024]
       - default_tactile_embedding: Learnable default embedding for gate=0
       - gate (NecessityGate): Gate network

    Common modules:
       - time_mlp (Pi05): Time MLP for adaRMS conditioning
       - lora: LoRA adaptation weights

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load base checkpoint
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray
        )
        # Preserve new tactile-related modules, fusion modules, time_mlp (Pi05), and LoRA weights
        # - .*tactile.*: tactile_encoder, tactile_to_vlm_proj, tactile_token_proj, default_tactile_embedding
        # - .*gate.*: NecessityGate (sigma_threshold, sigma_temperature)
        # - .*fusion.*: fusion_self_attn (ForceVLA-style)
        # - .*time_mlp.*: time_mlp_in, time_mlp_out (Pi05 adaRMS)
        # - .*lora.*: LoRA adaptation weights
        # - .*default.*embedding.*: default_tactile_embedding (Token-in-AE)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*lora.*|.*tactile.*|.*gate.*|.*fusion.*|.*time_mlp.*|.*default.*embedding.*"
        )


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
