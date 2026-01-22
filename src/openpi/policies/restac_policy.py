"""
ResTacVLA Policy: Data transformation for tactile-enabled VLA model.

This module provides input/output transformations for the ResTacVLA model,
handling both visual observations and tactile images.

For Franka robot:
- State: 7-dim [ee_x, ee_y, ee_z, ee_rot_x, ee_rot_y, ee_rot_z, gripper]
- Images: 3 cameras (main_realsense, side_realsense, handeye_realsense) + tactile
- Action: 7-dim EE pose delta
- Action_prev: 7-dim (state_t - state_t-1)
"""

import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import normalize as _normalize

logger = logging.getLogger("openpi")


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format.

    Handles conversion from various formats:
    - LeRobot uint8 (H, W, C) or (C, H, W)
    - Float32 (0-1) or (0-255) range
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _compute_action_prev(state: np.ndarray, state_prev: np.ndarray | None) -> np.ndarray:
    """
    Compute the previous action as state_t - state_t-1.

    Args:
        state: Current state [B, state_dim] or [state_dim]
        state_prev: Previous state [B, state_dim] or [state_dim], or None

    Returns:
        action_prev: Previous action [B, state_dim] or [state_dim]
    """
    if state_prev is None:
        # If no previous state, assume action_prev was zero
        return np.zeros_like(state)

    state = np.asarray(state, dtype=np.float32)
    state_prev = np.asarray(state_prev, dtype=np.float32)

    action_prev = state - state_prev
    return action_prev.astype(np.float32)


def make_restac_example() -> dict:
    """Create a random input example compatible with ResTacVLA."""
    return {
        "observation/state": np.random.rand(7),
        "observation/images/main_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/side_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/handeye_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/gelsight_left_rgb": np.random.randint(256, size=(128, 160, 3), dtype=np.uint8),
        "action_prev": np.zeros(7),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ResTacInputs(transforms.DataTransformFn):
    """
    ResTacVLA input transformation.

    Converts from LeRobot dataset format to model input format.

    Handles:
    - Robot state (7D: xyz + rpy + gripper)
    - Visual images (3 cameras: main, side, handeye)
    - Tactile image (GelSight)
    - Previous action (action_prev = state_t - state_t-1)
    - Language prompt

    Args:
        action_dim: Action dimension for padding
        model_type: Model type (PI0, PI05, or PI0_FAST)
    """
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 1. Parse robot state (7D)
        state = data.get("observation/state", np.zeros(7, dtype=np.float32))
        state = np.asarray(state, dtype=np.float32)
        if len(state) < 7:
            state = np.pad(state, (0, 7 - len(state)))
        state = transforms.pad_to_dim(state, self.action_dim)

        # 2. Parse action_prev (already computed in dataset as state_t - state_t-1)
        action_prev = data.get("action_prev", np.zeros(7, dtype=np.float32))
        action_prev = np.asarray(action_prev, dtype=np.float32)
        if len(action_prev) < 7:
            action_prev = np.pad(action_prev, (0, 7 - len(action_prev)))
        action_prev = transforms.pad_to_dim(action_prev, self.action_dim)

        # 3. Parse visual images from the three cameras
        main_image = _parse_image(data.get("observation/images/main_realsense_rgb"))
        side_image = _parse_image(data.get("observation/images/side_realsense_rgb"))
        handeye_image = _parse_image(data.get("observation/images/handeye_realsense_rgb"))

        # Map images based on model type
        # For PI0 and PI05: base_0_rgb (main), left_wrist_0_rgb (handeye), right_wrist_0_rgb (side)
        # For PI0_FAST: base_0_rgb (main), base_1_rgb (side), wrist_0_rgb (handeye)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "tactile_0")
                images = (main_image, handeye_image, side_image, None)  # tactile will be added below
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb", "tactile_0")
                images = (main_image, side_image, handeye_image, None)  # tactile will be added below
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # 4. Parse tactile image
        tactile_image = _parse_image(data.get("observation/images/gelsight_left_rgb"))

        # Convert images to list to modify tactile position
        image_list = list(images)
        image_list[-1] = tactile_image
        images = tuple(image_list)

        # 5. Create inputs dict with image mapping
        inputs = {
            "state": state,
            "action_prev": action_prev,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # 6. Handle actions (training only)
        # Data from LeRobot uses "action" key, but model expects "actions"
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if len(actions) < self.action_dim:
                actions = np.pad(actions, (0, self.action_dim - len(actions)))
            inputs["actions"] = actions

        # 7. Handle prompt
        if "prompt" in data:
            # Handle bytes prompt (from RLDS datasets)
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ResTacOutputs(transforms.DataTransformFn):
    """
    ResTacVLA output transformation.

    Returns only the first 7 dimensions of actions (xyz + rpy + gripper),
    since the model may pad to larger dimensions.
    """

    def __call__(self, data: dict) -> dict:
        # Only return first 7 actions (ee_x, ee_y, ee_z, ee_rot_x, ee_rot_y, ee_rot_z, gripper)
        # Model provides "actions" (plural), convert to dataset format "action" (singular)
        return {"action": np.asarray(data["actions"][:, :7])}


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
