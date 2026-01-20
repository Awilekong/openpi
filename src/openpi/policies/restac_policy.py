"""
ResTacVLA Policy: Data transformation for tactile-enabled VLA model.

This module provides input/output transformations for the ResTacVLA model,
handling both visual observations and tactile images.
"""

import dataclasses
import logging

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import normalize as _normalize

logger = logging.getLogger("openpi")


def _parse_image(image: np.ndarray | None) -> np.ndarray | None:
    """Parse and validate image input."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    return np.asarray(image, dtype=np.float32)


def make_restac_example() -> dict:
    """Create a random input example compatible with ResTacVLA."""
    return {
        "state": np.ones((7,), dtype=np.float32),  # 7D robot state
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "tactile_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ResTacInputs(transforms.DataTransformFn):
    """
    ResTacVLA input transformation.

    Handles:
    - Robot state (7D: xyz + rpy + gripper)
    - Visual images (base + wrist)
    - Tactile image (new input channel)
    - Language prompt

    Args:
        action_dim: Action dimension for padding
        model_type: Model type (PI0 or VLA)
    """
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # 1. Parse robot state (7D)
        state = data.get("state", np.zeros(7, dtype=np.float32))
        if len(state) < 7:
            state = np.pad(state, (0, 7 - len(state)))
        state = transforms.pad_to_dim(state, self.action_dim)

        # 2. Parse visual images
        base_image = _parse_image(data.get("image"))
        wrist_image = _parse_image(data.get("wrist_image"))

        # Handle missing images
        if base_image is None:
            base_image = np.zeros((480, 640, 3), dtype=np.float32)
        if wrist_image is None:
            wrist_image = np.zeros((480, 640, 3), dtype=np.float32)

        # 3. Parse tactile image
        tactile_image = _parse_image(data.get("tactile_image"))
        if tactile_image is None:
            tactile_image = np.zeros((224, 224, 3), dtype=np.float32)

        # 4. Build inputs dict
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # Placeholder
                "tactile_0": tactile_image,  # Tactile image
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "tactile_0": np.True_,  # Tactile always enabled
            },
        }

        # 5. Handle actions (training only)
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # 6. Handle prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ResTacOutputs(transforms.DataTransformFn):
    """
    ResTacVLA output transformation.

    Returns only the first 7 dimensions of actions (xyz + rpy + gripper).
    """

    def __call__(self, data: dict) -> dict:
        # Only return first 7 dimensions (xyz + rpy + gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class ResTacNormalization(transforms.DataTransformFn):
    """
    Normalization transform for ResTacVLA.

    Normalizes state and actions using provided statistics.
    """
    norm_stats: _normalize.NormStats
    state_keys: tuple = ("state",)
    action_keys: tuple = ("actions",)

    def __call__(self, data: dict) -> dict:
        result = dict(data)

        # Normalize state
        for key in self.state_keys:
            if key in result:
                result[key] = _normalize.normalize(
                    result[key],
                    self.norm_stats,
                    key
                )

        # Normalize actions
        for key in self.action_keys:
            if key in result:
                result[key] = _normalize.normalize(
                    result[key],
                    self.norm_stats,
                    key
                )

        return result


@dataclasses.dataclass(frozen=True)
class ResTacDenormalization(transforms.DataTransformFn):
    """
    Denormalization transform for ResTacVLA outputs.
    """
    norm_stats: _normalize.NormStats
    action_keys: tuple = ("actions",)

    def __call__(self, data: dict) -> dict:
        result = dict(data)

        for key in self.action_keys:
            if key in result:
                result[key] = _normalize.denormalize(
                    result[key],
                    self.norm_stats,
                    key
                )

        return result
