"""
Gate Effect Visualization Script

Generates videos showing tactile images alongside gate value curves
to verify the learned gate mechanism is working correctly.

Usage:
    uv run scripts/visualize_gate.py --checkpoint <ckpt_path> --config <config_name> \
        --episodes 0 1 --output <output_dir>
"""

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Sequence

import cv2
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.training.config as config_lib
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def create_gate_curve_panel(
    all_gate_values: np.ndarray,
    current_idx: int,
    width: int = 400,
    height: int = 200,
    dpi: int = 100,
) -> np.ndarray:
    """Create gate value curve visualization (dark style).

    Auto-scales Y-axis to show actual value range when values are small.
    """
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    T = len(all_gate_values)
    x = np.arange(T)

    # Gate curve with glow effect
    color = '#00ff88'
    ax.plot(x, all_gate_values, color=color, linewidth=1.5, alpha=0.9)
    ax.plot(x, all_gate_values, color=color, linewidth=4, alpha=0.2)  # glow

    # Current position line
    ax.axvline(x=current_idx, color='#ff4757', linestyle='-', linewidth=1.5, alpha=0.8)

    # Current value marker
    if current_idx < len(all_gate_values):
        current_val = all_gate_values[current_idx]
        ax.scatter([current_idx], [current_val], color='#ff4757', s=60, zorder=5,
                   edgecolors='white', linewidths=0.5)
        # Display current value with more precision for small values
        if current_val < 0.01:
            ax.text(current_idx + 2, current_val, f'{current_val:.6f}',
                    color='#ff4757', fontsize=9, fontweight='bold')
        else:
            ax.text(current_idx + 2, current_val, f'{current_val:.3f}',
                    color='#ff4757', fontsize=10, fontweight='bold')

    ax.set_ylabel('Gate Value', fontsize=10, color='#a0a0a0')
    ax.set_xlabel('Frame', fontsize=10, color='#a0a0a0')
    ax.set_xlim(0, T)

    # Auto-scale Y-axis based on actual value range
    g_min, g_max = all_gate_values.min(), all_gate_values.max()
    g_range = g_max - g_min

    if g_max < 0.1:
        # Small values: zoom in to show variation with padding
        padding = max(g_range * 0.2, g_max * 0.1)  # 20% padding or 10% of max
        y_min = max(0, g_min - padding)
        y_max = g_max + padding
        ax.set_ylim(y_min, y_max)
        # Use scientific notation for small values
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, -3))
        title_suffix = f" [range: {g_min:.2e} - {g_max:.2e}]"
    else:
        # Normal range: use [0, 1]
        ax.set_ylim(-0.05, 1.05)
        title_suffix = ""

    ax.tick_params(axis='both', labelsize=8, colors='#606060')
    ax.grid(True, alpha=0.15, color='#404040')

    # Title with range info for small values
    ax.set_title(f'Gate Value (g){title_suffix}', fontsize=11, color='#00ff88', pad=10)

    for spine in ax.spines.values():
        spine.set_color('#404040')

    plt.tight_layout()

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    plt.style.use('default')

    return img


def create_info_panel(
    gate_value: float,
    logvar: float,
    episode_idx: int,
    frame_idx: int,
    total_frames: int,
    width: int = 200,
    height: int = 200,
) -> np.ndarray:
    """Create info panel with current values (dark style)."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 26, 22)  # BGR dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    y = 30

    # Colors (BGR)
    cyan = (255, 255, 0)
    green = (136, 255, 0)
    magenta = (255, 0, 255)

    # Episode and Frame info
    cv2.putText(panel, f"Episode: {episode_idx}", (10, y), font, 0.55, cyan, thickness)
    y += 28
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (10, y), font, 0.55, cyan, thickness)
    y += 40

    # Gate value (use scientific notation for small values)
    cv2.putText(panel, "GATE VALUE", (10, y), font, 0.5, green, thickness)
    y += 25
    if gate_value < 0.01:
        cv2.putText(panel, f"g = {gate_value:.2e}", (10, y), font, 0.55, green, thickness)
    else:
        cv2.putText(panel, f"g = {gate_value:.4f}", (10, y), font, 0.6, green, thickness)
    y += 35

    # Logvar
    cv2.putText(panel, "LOGVAR (sigma)", (10, y), font, 0.5, magenta, thickness)
    y += 25
    cv2.putText(panel, f"s = {logvar:.4f}", (10, y), font, 0.6, magenta, thickness)

    return panel


def run_inference_and_collect_gates(
    model: _model.BaseModel,
    dataset: LeRobotDataset,
    episode_idx: int,
    data_config: config_lib.DataConfig,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Run inference on an episode and collect gate values.

    Returns:
        tactile_images: List of tactile images (original, not resized)
        gate_values: List of gate values per frame
        logvars: List of logvar values per frame
    """
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()

    tactile_images = []
    gate_values = []
    logvars = []

    # Get camera keys
    camera_keys = dataset.meta.camera_keys
    tactile_key = None
    for key in camera_keys:
        if 'gelsight' in key.lower() or 'tactile' in key.lower():
            tactile_key = key
            break

    if tactile_key is None:
        logger.warning("No tactile camera found, using first camera")
        tactile_key = camera_keys[0] if camera_keys else None

    logger.info(f"Processing episode {episode_idx}: frames {from_idx} to {to_idx}")
    logger.info(f"Tactile key: {tactile_key}")

    for idx in tqdm(range(from_idx, to_idx), desc=f"Episode {episode_idx}"):
        frame_data = dataset[idx]

        # Get tactile image (original for display)
        if tactile_key:
            tactile_img = frame_data[tactile_key]
            if hasattr(tactile_img, 'numpy'):
                tactile_img = tactile_img.numpy()
            if tactile_img.ndim == 3 and tactile_img.shape[0] == 3:
                tactile_img = np.transpose(tactile_img, (1, 2, 0))
            if tactile_img.dtype == np.float32 or tactile_img.dtype == np.float64:
                tactile_img = (tactile_img * 255).astype(np.uint8)
            tactile_images.append(tactile_img)

        # Prepare data for model inference - apply transforms
        sample = dict(frame_data)

        # Convert tensors to numpy
        for key in sample:
            if hasattr(sample[key], 'numpy'):
                sample[key] = sample[key].numpy()

        # Add missing fields that transforms may expect
        if 'prompt' not in sample:
            sample['prompt'] = "do something"
        if 'action_prev' not in sample:
            sample['action_prev'] = np.zeros(7, dtype=np.float32)

        # Apply repack transforms
        for transform in data_config.repack_transforms.inputs:
            sample = transform(sample)

        # Apply data transforms (only inputs, skip DeltaActions which needs batch dimension)
        from openpi.transforms import DeltaActions
        for transform in data_config.data_transforms.inputs:
            if not isinstance(transform, DeltaActions):
                sample = transform(sample)

        # Apply model transforms (only inputs, skip tokenization and normalization which we don't need)
        from openpi.transforms import ResizeImages
        from openpi.policies.restac_policy import ResTacNormalizeActionPrev
        for transform in data_config.model_transforms.inputs:
            # Only apply ResizeImages and skip others that may cause issues
            if isinstance(transform, ResizeImages):
                sample = transform(sample)

        # Extract model inputs
        tactile_for_model = sample['image'].get('tactile_0')
        if tactile_for_model is None:
            logger.warning(f"No tactile_0 in sample at idx {idx}")
            gate_values.append(0.0)
            logvars.append(0.0)
            continue

        # Tactile encoder expects 128x160, but ResizeImages resizes all to 224x224
        # Resize tactile back to the expected dimensions
        tactile_h, tactile_w = tactile_for_model.shape[:2]
        if (tactile_h, tactile_w) != (128, 160):
            tactile_for_model = cv2.resize(tactile_for_model, (160, 128), interpolation=cv2.INTER_LINEAR)

        # Add batch dimension
        tactile_for_model = np.expand_dims(tactile_for_model, 0)

        # Get visual images for prophet
        visual_images = []
        for vkey in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            if vkey in sample['image']:
                visual_images.append(sample['image'][vkey])

        if len(visual_images) == 3:
            # Stack to [3, H, W, C] then transpose to [3, C, H, W]
            visual_3views = np.stack(visual_images, axis=0)
            visual_3views = np.transpose(visual_3views, (0, 3, 1, 2))
            visual_3views = np.expand_dims(visual_3views, 0)  # [1, 3, C, H, W]
        else:
            visual_3views = None

        # Get action_prev
        action_prev = sample.get('action_prev', np.zeros(7, dtype=np.float32))
        action_prev = np.expand_dims(action_prev, 0)  # [1, 7]

        # Run tactile encoding to get gate value
        try:
            # Convert to JAX arrays
            tactile_jax = jnp.array(tactile_for_model)
            visual_jax = jnp.array(visual_3views) if visual_3views is not None else None
            action_jax = jnp.array(action_prev)

            # Check model type and call appropriate method
            if hasattr(model, 'encode_tactile'):
                # Pi0_ResTac model
                q_event, gate_value, logvar = model.encode_tactile(tactile_jax, visual_jax, action_jax)
            elif hasattr(model, 'encode_tactile_raw'):
                # Pi0_ResTac_TokenInAE model
                q_event, gate_value, logvar = model.encode_tactile_raw(tactile_jax, visual_jax, action_jax)
            else:
                logger.warning("Model does not have encode_tactile method")
                gate_value = jnp.zeros((1, 1))
                logvar = jnp.zeros((1, 1))

            gate_values.append(float(gate_value[0, 0]))
            logvars.append(float(logvar[0, 0]))

        except Exception as e:
            logger.warning(f"Error during inference at idx {idx}: {e}")
            gate_values.append(0.0)
            logvars.append(0.0)

    # Print statistics
    gate_arr = np.array(gate_values)
    logvar_arr = np.array(logvars)
    print(f"\n=== Episode {episode_idx} Statistics ===")
    print(f"Gate values: min={gate_arr.min():.6f}, max={gate_arr.max():.6f}, mean={gate_arr.mean():.6f}, std={gate_arr.std():.6f}")
    print(f"Logvar values: min={logvar_arr.min():.4f}, max={logvar_arr.max():.4f}, mean={logvar_arr.mean():.4f}")
    print(f"==============================\n")

    return tactile_images, gate_values, logvars


def generate_gate_video(
    tactile_images: list[np.ndarray],
    gate_values: list[float],
    logvars: list[float],
    episode_idx: int,
    output_path: Path,
    fps: int = 10,
    target_size: tuple[int, int] = (320, 256),
) -> None:
    """Generate video with tactile images and gate curve."""
    total_frames = len(tactile_images)
    gate_array = np.array(gate_values)

    # Layout
    img_w, img_h = target_size
    curve_w, curve_h = 400, 200
    info_w, info_h = 200, 200

    canvas_w = img_w + curve_w + info_w
    canvas_h = max(img_h, curve_h + info_h)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    logger.info(f"Generating video: {output_path}")
    logger.info(f"  Frames: {total_frames}, Size: {canvas_w}x{canvas_h}")

    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (30, 26, 22)  # Dark background

        # 1. Tactile image
        img = tactile_images[frame_idx].copy()
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Add label
        cv2.putText(img_bgr, "Tactile", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        canvas[0:img_h, 0:img_w] = img_bgr

        # 2. Gate curve
        curve_img = create_gate_curve_panel(gate_array, frame_idx, width=curve_w, height=curve_h)
        curve_img_bgr = cv2.cvtColor(curve_img, cv2.COLOR_RGB2BGR)
        canvas[0:curve_h, img_w:img_w+curve_w] = curve_img_bgr

        # 3. Info panel
        info_panel = create_info_panel(
            gate_values[frame_idx],
            logvars[frame_idx],
            episode_idx,
            frame_idx,
            total_frames,
            width=info_w,
            height=info_h,
        )
        canvas[curve_h:curve_h+info_h, img_w:img_w+info_w] = info_panel

        video_writer.write(canvas)

    video_writer.release()
    logger.info(f"Video saved: {output_path}")


def load_model_from_checkpoint(
    checkpoint_dir: str,
    config_name: str,
) -> tuple[_model.BaseModel, config_lib.DataConfig]:
    """
    Load model from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/pi05_restac/exp_name)
        config_name: Training config name

    Returns:
        model: Loaded model
        data_config: Data configuration
    """
    import flax.nnx as nnx

    checkpoint_dir = Path(checkpoint_dir)

    # Get config
    train_config = config_lib.get_config(config_name)
    model_config = train_config.model

    # Find latest step
    steps = []
    for p in checkpoint_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            steps.append(int(p.name))

    if not steps:
        raise ValueError(f"No checkpoint steps found in {checkpoint_dir}")

    latest_step = max(steps)
    step_dir = checkpoint_dir / str(latest_step)
    params_dir = step_dir / "params"

    logger.info(f"Loading checkpoint from step {latest_step}: {params_dir}")

    # Initialize model
    rng = jax.random.key(0)
    model = model_config.create(rng)

    # Create checkpoint handler and restore (let orbax infer structure)
    handler = ocp.PyTreeCheckpointHandler()
    restored = handler.restore(params_dir)

    # The restored structure is {"params": {...}} where each leaf is {"value": array}
    # We need to extract just the values
    def extract_values(tree):
        if isinstance(tree, dict):
            if "value" in tree and len(tree) == 1:
                return tree["value"]
            return {k: extract_values(v) for k, v in tree.items()}
        return tree

    params = extract_values(restored["params"])

    # Load params into model
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)

    # Setup data config
    def _is_empty_dir(path: Path) -> bool:
        """Check if directory doesn't exist or is empty."""
        return not path.exists() or not any(path.iterdir())

    assets_dir = step_dir / "assets"
    if _is_empty_dir(assets_dir):
        assets_dir = checkpoint_dir / "assets"
    if _is_empty_dir(assets_dir):
        # For ResTac configs, norm_stats are in the dataset directory
        # Use repo_id directly as assets path (the norm_stats.json is at repo_id/norm_stats.json)
        repo_id = getattr(train_config.data, 'repo_id', None)
        if repo_id:
            repo_path = Path(repo_id)
            if repo_path.exists() and (repo_path / "norm_stats.json").exists():
                # Use repo's parent as assets_dir, with repo basename as asset_id
                assets_dir = repo_path.parent
                # Override asset_id in the data config factory
                train_config = dataclasses.replace(
                    train_config,
                    data=dataclasses.replace(
                        train_config.data,
                        assets=config_lib.AssetsConfig(
                            assets_dir=str(repo_path.parent),
                            asset_id=repo_path.name
                        )
                    )
                )

    data_config = train_config.data.create(assets_dir, model_config)

    return model, data_config


def visualize_gate_for_episodes(
    checkpoint_path: str,
    config_name: str,
    episode_indices: Sequence[int],
    output_dir: str,
    fps: int = 10,
) -> list[str]:
    """
    Main function to visualize gate values for specified episodes.

    Args:
        checkpoint_path: Path to model checkpoint directory
        config_name: Training config name (e.g., 'pi05_restac_token_in_ae')
        episode_indices: List of episode indices to visualize
        output_dir: Output directory for videos
        fps: Video frame rate

    Returns:
        List of output video paths
    """
    # Load model
    model, data_config = load_model_from_checkpoint(checkpoint_path, config_name)

    # Load dataset
    repo_id = data_config.repo_id
    logger.info(f"Loading dataset: {repo_id}")

    # LeRobotDataset handles local paths when repo_id is a valid path
    dataset = LeRobotDataset(repo_id)
    num_episodes = dataset.num_episodes
    logger.info(f"Dataset has {num_episodes} episodes")

    # Process episodes
    output_paths = []
    output_dir = Path(output_dir)

    for ep_idx in episode_indices:
        if ep_idx >= num_episodes:
            logger.warning(f"Episode {ep_idx} out of range (max: {num_episodes-1}), skipping")
            continue

        # Run inference and collect gates
        tactile_images, gate_values, logvars = run_inference_and_collect_gates(
            model, dataset, ep_idx, data_config
        )

        if not tactile_images:
            logger.warning(f"No frames collected for episode {ep_idx}, skipping")
            continue

        # Generate video
        output_path = output_dir / f"gate_episode_{ep_idx}.mp4"
        generate_gate_video(
            tactile_images, gate_values, logvars,
            ep_idx, output_path, fps=fps
        )
        output_paths.append(str(output_path))

    return output_paths


def visualize_after_training(
    checkpoint_dir: str,
    config_name: str,
    num_episodes: int = 2,
    fps: int = 10,
) -> list[str]:
    """
    Convenience function to visualize gate after training.
    Called automatically at the end of training.

    Args:
        checkpoint_dir: Training checkpoint directory
        config_name: Training config name
        num_episodes: Number of episodes to visualize
        fps: Video frame rate

    Returns:
        List of output video paths
    """
    import random

    # Get config to find dataset
    train_config = config_lib.get_config(config_name)
    data_config_factory = train_config.data

    # Get repo_id from config
    repo_id = getattr(data_config_factory, 'repo_id', None)
    if repo_id is None:
        logger.warning("Could not determine repo_id from config")
        return []

    # Load dataset to get episode count
    dataset = LeRobotDataset(repo_id)
    total_episodes = dataset.num_episodes

    # Randomly select episodes
    if total_episodes <= num_episodes:
        episode_indices = list(range(total_episodes))
    else:
        episode_indices = random.sample(range(total_episodes), num_episodes)

    logger.info(f"Visualizing episodes: {episode_indices}")

    # Output to checkpoint dir
    output_dir = Path(checkpoint_dir) / "gate_visualizations"

    return visualize_gate_for_episodes(
        checkpoint_path=checkpoint_dir,
        config_name=config_name,
        episode_indices=episode_indices,
        output_dir=str(output_dir),
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize gate values for ResTac model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory path")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., pi05_restac_token_in_ae)")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1], help="Episode indices to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: checkpoint/gate_visualizations)")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    output_dir = args.output or str(Path(args.checkpoint) / "gate_visualizations")

    output_paths = visualize_gate_for_episodes(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        episode_indices=args.episodes,
        output_dir=output_dir,
        fps=args.fps,
    )

    print(f"\nGenerated {len(output_paths)} videos:")
    for p in output_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
