"""
LeRobot 数据集可视化脚本

功能：
- 随机选取一个 episode，生成包含相机图像和 state 曲线的视频
- 7个 state 维度分别绘制
- 支持多相机拼接显示
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def create_state_curves(
    all_states: np.ndarray,
    current_idx: int,
    state_names: list,
    width: int = 400,
    height: int = 600,
    dpi: int = 100,
):
    """创建 state 曲线图，每个维度单独一行 (暗黑风格)"""
    n_dims = all_states.shape[1]

    # 暗黑风格
    plt.style.use('dark_background')

    fig, axes = plt.subplots(n_dims, 1, figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')

    if n_dims == 1:
        axes = [axes]

    T = len(all_states)
    x = np.arange(T)

    # 霓虹色系
    neon_colors = ['#00fff5', '#ff00ff', '#00ff88', '#ffff00', '#ff6b6b', '#4ecdc4', '#a855f7']

    for i, ax in enumerate(axes):
        ax.set_facecolor('#16213e')

        name = state_names[i] if i < len(state_names) else f"s{i}"
        color = neon_colors[i % len(neon_colors)]

        # 绘制曲线，带发光效果
        ax.plot(x, all_states[:, i], color=color, linewidth=1.2, alpha=0.9)
        ax.plot(x, all_states[:, i], color=color, linewidth=3, alpha=0.2)  # 发光

        # 当前位置竖线
        ax.axvline(x=current_idx, color='#ff4757', linestyle='-', linewidth=1.5, alpha=0.8)

        # 当前值标记
        current_val = all_states[current_idx, i]
        ax.scatter([current_idx], [current_val], color='#ff4757', s=40, zorder=5, edgecolors='white', linewidths=0.5)

        ax.set_ylabel(name, fontsize=8, color='#a0a0a0')
        ax.set_xlim(0, T)
        ax.tick_params(axis='both', labelsize=6, colors='#606060')
        ax.grid(True, alpha=0.15, color='#404040')

        # 边框颜色
        for spine in ax.spines.values():
            spine.set_color('#404040')

        if i < n_dims - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Frame', fontsize=8, color='#a0a0a0')
    plt.tight_layout()

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    # 重置样式
    plt.style.use('default')

    return img


def create_info_panel(
    current_state: np.ndarray,
    state_names: list,
    episode_idx: int,
    frame_idx: int,
    total_frames: int,
    width: int = 200,
    height: int = 300,
):
    """创建信息面板 (暗黑风格)"""
    # 深色背景 - 与曲线图背景一致
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 26, 22)  # BGR: #16213e 类似色

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = 22
    y = 30

    # 霓虹色系 (BGR)
    cyan = (255, 255, 0)      # #00ffff
    magenta = (255, 0, 255)   # #ff00ff
    green = (136, 255, 0)     # #00ff88
    red = (87, 71, 255)       # #ff4757
    gray = (160, 160, 160)    # #a0a0a0

    # Episode 和 Frame 信息
    cv2.putText(panel, f"Episode: {episode_idx}", (10, y), font, 0.55, cyan, thickness)
    y += 28
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (10, y), font, 0.55, cyan, thickness)
    y += 35

    # State 标题
    cv2.putText(panel, "STATE", (10, y), font, 0.5, green, thickness)
    y += line_height + 5

    # 霓虹色列表
    neon_colors_bgr = [
        (245, 255, 0),    # #00fff5
        (255, 0, 255),    # #ff00ff
        (136, 255, 0),    # #00ff88
        (0, 255, 255),    # #ffff00
        (107, 107, 255),  # #ff6b6b
        (196, 205, 78),   # #4ecdc4
        (247, 85, 168),   # #a855f7
    ]

    for i, val in enumerate(current_state):
        name = state_names[i] if i < len(state_names) else f"s{i}"
        color = neon_colors_bgr[i % len(neon_colors_bgr)]
        text = f"{name}: {val:+.4f}"
        cv2.putText(panel, text, (10, y), font, 0.4, color, thickness)
        y += line_height

    return panel


def visualize_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    output_path: Path,
    fps: int = 10,
    target_size: tuple = (224, 224),
):
    """将指定 episode 的相机图像和曲线拼接成视频"""

    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    total_frames = to_idx - from_idx

    camera_keys = dataset.meta.camera_keys
    print(f"Camera keys: {camera_keys}")

    # 预加载所有数据
    print("Loading episode data...")
    all_states = []
    all_frames = {key: [] for key in camera_keys}

    for idx in tqdm(range(from_idx, to_idx), desc="Loading data"):
        frame_data = dataset[idx]

        if "observation.state" in frame_data:
            state = frame_data["observation.state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            all_states.append(state)

        for key in camera_keys:
            img = frame_data[key]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0)
                img = (img * 255).numpy().astype(np.uint8)
            all_frames[key].append(img)

    all_states = np.array(all_states) if all_states else np.zeros((total_frames, 7))

    state_names = ["x", "y", "z", "rx", "ry", "rz", "grip"][:all_states.shape[1]]

    # 布局计算
    n_cams = len(camera_keys)
    n_cols = 2
    n_rows = (n_cams + 1) // 2

    img_h, img_w = target_size
    cam_area_h = img_h * n_rows
    cam_area_w = img_w * n_cols

    # 曲线和信息面板尺寸
    curve_w = 350
    curve_h = cam_area_h
    info_w = 180
    info_h = cam_area_h

    canvas_w = cam_area_w + curve_w + info_w
    canvas_h = cam_area_h

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    print(f"Generating video for episode {episode_index}")
    print(f"  Frames: {total_frames}, FPS: {fps}")
    print(f"  Output: {output_path}")
    print(f"  Size: {canvas_w}x{canvas_h}")

    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 1. 相机图像
        for cam_idx, key in enumerate(camera_keys):
            img = all_frames[key][frame_idx].copy()

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.concatenate([img] * 3, axis=-1)

            if img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cam_label = key.split('.')[-1]
            cv2.putText(img_bgr, cam_label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            row = cam_idx // n_cols
            col = cam_idx % n_cols
            y1, y2 = row * img_h, (row + 1) * img_h
            x1, x2 = col * img_w, (col + 1) * img_w
            canvas[y1:y2, x1:x2] = img_bgr

        # 2. State 曲线
        curve_img = create_state_curves(
            all_states, frame_idx, state_names,
            width=curve_w, height=curve_h
        )
        curve_img_bgr = cv2.cvtColor(curve_img, cv2.COLOR_RGB2BGR)
        canvas[0:curve_h, cam_area_w:cam_area_w+curve_w] = curve_img_bgr

        # 3. 信息面板
        info_panel = create_info_panel(
            all_states[frame_idx], state_names,
            episode_index, frame_idx, total_frames,
            width=info_w, height=info_h
        )
        canvas[0:info_h, cam_area_w+curve_w:canvas_w] = info_panel

        video_writer.write(canvas)

    video_writer.release()
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset episode as video")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--episode", type=int, default=None, help="Episode index (random if not specified)")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    parser.add_argument("--size", type=int, default=224, help="Image size for each camera view")

    args = parser.parse_args()

    dataset_path = args.root / args.repo_id
    print(f"Loading dataset: {args.repo_id} from {dataset_path}")
    dataset = LeRobotDataset(args.repo_id, root=dataset_path)

    num_episodes = dataset.num_episodes
    print(f"Dataset has {num_episodes} episodes")

    if args.episode is not None:
        episode_index = args.episode
    else:
        episode_index = random.randint(0, num_episodes - 1)
        print(f"Randomly selected episode: {episode_index}")

    if args.output is None:
        output_path = dataset_path / f"episode_{episode_index}.mp4"
    else:
        output_path = args.output

    visualize_episode(
        dataset, episode_index, output_path,
        args.fps, target_size=(args.size, args.size)
    )


if __name__ == "__main__":
    main()
