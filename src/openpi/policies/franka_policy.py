import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/images/main_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/side_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/handeye_realsense_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. 
    It is used for both training and inference.

    For Franka robot:
    - State: 7-dim [ee_x, ee_y, ee_z, ee_rot_x, ee_rot_y, ee_rot_z, gripper]
    - Images: 3 cameras (main_realsense, side_realsense, handeye_realsense)
    - Action: 7-dim EE pose delta (same as state)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        main_image = _parse_image(data["observation/images/main_realsense_rgb"])
        side_image = _parse_image(data["observation/images/side_realsense_rgb"])
        handeye_image = _parse_image(data["observation/images/handeye_realsense_rgb"])

        # Map images based on model type
        # For pi0 and pi05: base_0_rgb (main camera), left_wrist_0_rgb (handeye), right_wrist_0_rgb (side camera)
        # For pi0-FAST: base_0_rgb (main camera), base_1_rgb (side camera), wrist_0_rgb (handeye)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (main_image, handeye_image, side_image)
                # All three images are available, so mask all as True
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (main_image, side_image, handeye_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            # Handle bytes prompt (from RLDS datasets)
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. 
    It is used for inference only.

    For Franka robot, we only return the first 7 actions (ee_x, ee_y, ee_z, ee_rot_x, ee_rot_y, ee_rot_z, gripper).
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 actions -- since we may have padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Franka, we only return the first 7 actions.
        return {"actions": np.asarray(data["actions"][:, :7])}
