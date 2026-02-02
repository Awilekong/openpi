"""
Franka 单臂数据转换为 LeRobot 格式

功能:
    - 从 Franka 机器人采集的原始数据(jsonl + mp4)转换为 LeRobot 数据集格式
    - 支持多相机(main_realsense, side_realsense, handeye_realsense)
    - 自动处理数据增量、帧提取、图像 resize 等
    - 可配置数据处理策略和采样间隔
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import jsonlines
import numpy as np
import torch
import tqdm
from scipy.spatial.transform import Rotation as R
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 线程控制，避免过度并行
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ===== 动作空间定义 =====

class ActionSpace(Enum):
    """动作空间类型"""
    JOINT_POSITION_GLOBAL = "joint_position_global"  # state=joint_pos, action=next_joint_pos
    JOINT_POSITION_DELTA = "joint_position_delta"    # state=joint_pos, action=delta_joint_pos
    EE_POSE_GLOBAL = "ee_pose_global"                # state=ee_pose(xyz+rotvec), action=next_ee_pose
    EE_POSE_DELTA = "ee_pose_delta"                  # state=ee_pose(xyz+rotvec), action=delta_ee_pose


# ===== 旋转转换工具 =====

def quaternion_to_rotation_vector(quaternions: np.ndarray) -> np.ndarray:
    """
    四元数转旋转向量 (axis-angle representation)
    
    Args:
        quaternions: (T, 4) 四元数数组 [qw, qx, qy, qz] 或 [qx, qy, qz, qw]
    
    Returns:
        rotation_vectors: (T, 3) 旋转向量数组
    """
    # Franka 使用 [qw, qx, qy, qz] 格式
    # scipy 需要 [qx, qy, qz, qw] 格式
    if quaternions.shape[-1] == 4:
        # 转换格式: [qw, qx, qy, qz] -> [qx, qy, qz, qw]
        quaternions_scipy = np.concatenate([
            quaternions[..., 1:4],  # qx, qy, qz
            quaternions[..., 0:1]   # qw
        ], axis=-1)
    else:
        quaternions_scipy = quaternions
    
    # 使用 scipy 转换
    rotations = R.from_quat(quaternions_scipy)
    rotation_vectors = rotations.as_rotvec()
    
    return rotation_vectors.astype(np.float32)


# ===== 配置 =====

@dataclass
class Config:
    """全局配置"""
    # 数据路径列表 (支持多个源目录合并)
    source_paths: List[Path] = None

    # 输出配置
    repo_id: str = "wipe_plate"
    output_dir: Path = Path("/home/dataset-local/data/megvii_post/tac")  # 数据集保存目录
    target_size: Tuple[int, int] = (224, 224)  # (H, W)
    tactile_size: Tuple[int, int] = (128, 160)  # 触觉图像目标尺寸 (H, W)

    # 动作空间配置
    action_space: ActionSpace = ActionSpace.EE_POSE_DELTA

    # 数据处理
    stride: int = 1  # 采样间隔

    # 帧过滤配置
    enable_frame_filtering: bool = True  # 是否启用帧过滤
    frame_filter_threshold: float = 1e-10  # 帧过滤阈值：state 变化幅度 (默认只过滤静止帧)
    min_frames_per_episode: int = 10  # 每个 episode 最少保留的帧数

    # 夹爪配置
    enable_gripper_binarization: bool = True  # 是否启用夹爪二值化
    gripper_binarization_threshold: float = 0.079  # 夹爪二值化阈值

    # 提示词配置
    prompt: Optional[str] = None  # 如果指定，将覆盖 meta.json 中的 prompt

    # None 表示转换所有，否则转换前n个
    max_episodes: Optional[int] = None

    # 相机配置
    camera_names: List[str] = None  # None表示自动检测

    # 触觉数据配置
    require_tactile: bool = True  # 是否要求触觉数据完整
    tactile_delay_offset: float = 0.3  # 触觉硬件延迟补偿 (秒)，触觉实际时间 = 记录时间 - offset

    def __post_init__(self):
        if self.source_paths is None:
            # 默认路径列表
            self.source_paths = [
                Path("/home/dataset-local/data/megvii/wipe_plate"),
            ]

        if self.camera_names is None:
            self.camera_names = ["main_realsense_rgb", "side_realsense_rgb", "handeye_realsense_rgb"]

        # 根据动作空间自动更新 repo_id
        action_suffix = self.action_space.value
        if action_suffix not in self.repo_id:
            self.repo_id = f"{self.repo_id}_{action_suffix}"


# ===== 数据读取模块 =====

class FrankaDataLoader:
    """Franka 数据加载器 (支持多数据源)"""

    def __init__(self, config: Config):
        self.config = config

    def get_all_episodes(self) -> List[Path]:
        """获取所有 episode 的完整路径"""
        all_episodes = []

        for source_path in self.config.source_paths:
            if not source_path.exists():
                print(f"Warning: Source path not found: {source_path}")
                continue

            # 获取该源下的所有 episode 目录
            episodes = sorted([d for d in source_path.iterdir() if d.is_dir()])
            print(f"Found {len(episodes)} episodes in {source_path}")
            all_episodes.extend(episodes)

        if self.config.max_episodes:
            all_episodes = all_episodes[:self.config.max_episodes]

        return all_episodes

    def load_meta(self, episode_path: Path) -> Dict:
        """加载 meta.json"""
        meta_path = episode_path / "v1" / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_robot_data(self, episode_path: Path) -> Dict[str, np.ndarray]:
        """从 jsonl 加载机器人数据"""
        jsonl_path = episode_path / "v1" / "data" / "Franka_4_arms_arm.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Robot data not found: {jsonl_path}")

        with jsonlines.open(jsonl_path) as reader:
            data = list(reader)

        # 提取字段 (Unix 时间戳需要 float64 精度，float32 会丢失约 50 秒精度)
        timestamps = np.array([d['timestamp'] for d in data], dtype=np.float64)
        joint_positions = np.array([d['joint_positions'] for d in data], dtype=np.float32)  # (T, 7)
        ee_positions = np.array([d['ee_positions'] for d in data], dtype=np.float32)  # (T, 7)
        gripper = np.array([d['gripper'] for d in data], dtype=np.float32)  # (T, 2)
        gripper_width = np.array([d['gripper_width'][0] for d in data], dtype=np.float32)  # (T,)

        return {
            'timestamps': timestamps,
            'joint_positions': joint_positions,  # (T, 7) - Franka 有 7 个关节
            'ee_positions': ee_positions,  # (T, 7) - xyz(3) + quaternion(4)
            'gripper': gripper,  # (T, 2) - 两个手指
            'gripper_width': gripper_width,  # (T,) - 夹爪宽度
        }

    def load_camera_timestamps(self, episode_path: Path, camera_name: str = "main_realsense") -> np.ndarray:
        """加载相机时间戳文件

        Args:
            episode_path: episode 路径
            camera_name: 相机名称 (不带 _rgb 后缀)

        Returns:
            timestamps: (T,) float64 时间戳数组 (单位: 秒)
        """
        ts_path = episode_path / "v1" / "origin_data" / f"{camera_name}_timestamps.jsonl"
        if not ts_path.exists():
            raise FileNotFoundError(f"Camera timestamps not found: {ts_path}")

        timestamps = []
        with jsonlines.open(ts_path) as reader:
            for record in reader:
                # timestamp 是毫秒，转换为秒
                ts = record['timestamp'] / 1000.0
                timestamps.append(ts)

        return np.array(timestamps, dtype=np.float64)

    def load_video_frames(self, episode_path: Path, frame_indices: List[int]) -> Dict[str, np.ndarray]:
        """加载指定帧的图像（并行加载多个相机）"""
        video_dir = episode_path / "v1" / "videos"

        def load_camera(cam_name):
            video_path = video_dir / f"{cam_name}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            return cam_name, self._extract_frames(str(video_path), frame_indices)

        # 并行加载所有相机
        frames = {}
        with ThreadPoolExecutor(max_workers=len(self.config.camera_names)) as executor:
            results = executor.map(load_camera, self.config.camera_names)
            for cam_name, cam_frames in results:
                frames[cam_name] = cam_frames

        return frames

    def _extract_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """从视频提取指定索引的帧，返回 (T, H, W, 3) RGB uint8"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                print(f"Warning: Failed to read frame {idx} from {video_path}")

        cap.release()
        return np.array(frames, dtype=np.uint8)

    def load_tactile_video(self, episode_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """加载 GelSight 触觉视频和对应的时间戳

        Returns:
            tactile_frames: (T_tactile, H, W, 3) RGB uint8
            tactile_timestamps: (T_tactile,) float64 Unix时间戳
        """
        video_dir = episode_path / "v1" / "gelsight"
        video_path = video_dir / "gelsight_left_rgb.mp4"

        if not video_path.exists():
            # 触觉数据可选，返回空数组
            return np.array([]), np.array([])

        # 读取视频帧
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open tactile video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        tactile_frames = np.array(frames, dtype=np.uint8)

        # 读取时间戳文件 (使用 float64 保持精度，Unix 时间戳需要高精度)
        ts_path = episode_path / "v1" / "origin_data" / "gelsight_left_timestamps.txt"
        if ts_path.exists():
            tactile_timestamps = np.loadtxt(ts_path, dtype=np.float64)
        else:
            # 备用：从视频FPS估算
            fps = 10.0  # GelSight 标准频率
            tactile_timestamps = np.arange(len(frames), dtype=np.float64) / fps

        return tactile_frames, tactile_timestamps


# ===== 数据处理模块 =====

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_episode(
        self,
        robot_data: Dict[str, np.ndarray],
        video_frames: Dict[str, np.ndarray],
        camera_timestamps: Optional[np.ndarray] = None,
        tactile_frames: Optional[np.ndarray] = None,
        tactile_timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """处理单个 episode 的数据

        处理流程 (统一以相机时间戳为基准):
        1. 步骤1: 在相机帧上进行 stride 采样
        2. 步骤2: 将机器人状态对齐到相机时间戳
        3. 步骤3: 【可选】帧过滤 - 根据 state 变化过滤关键帧
        4. 步骤4: 计算 action - 基于过滤后的帧计算 action (delta 或 next_frame)
        5. 步骤5: 处理视频帧和触觉帧（触觉需要 -0.3s 延迟补偿）
        """
        stride = self.config.stride
        action_space = self.config.action_space

        if camera_timestamps is None:
            raise ValueError("Camera timestamps are required for proper alignment")

        # === 步骤1: 在相机帧上进行 stride 采样 ===
        T_camera = len(camera_timestamps)
        stride_cam_indices = list(range(0, T_camera - stride, stride))

        # 检查视频帧数量是否匹配
        first_cam_frames = list(video_frames.values())[0]
        if len(first_cam_frames) < len(stride_cam_indices):
            # 视频帧数量与预期不符，裁剪到最小长度
            stride_cam_indices = stride_cam_indices[:len(first_cam_frames)]

        # === 步骤2: 将机器人状态对齐到相机时间戳 ===
        aligned_robot_data = self.align_robot_to_camera(
            robot_data,
            camera_timestamps,
            stride_cam_indices
        )

        # 根据动作空间提取 state 数据
        if action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.JOINT_POSITION_DELTA]:
            # 使用关节空间
            stride_state_data = aligned_robot_data['joint_positions']  # (T_stride, 7)
        elif action_space in [ActionSpace.EE_POSE_GLOBAL, ActionSpace.EE_POSE_DELTA]:
            # 使用末端位姿空间 - 需要先转换为 xyz + rotation_vector
            ee_positions = aligned_robot_data['ee_positions']  # (T_stride, 7) - xyz(3) + quaternion(4)
            ee_xyz = ee_positions[:, :3]  # (T_stride, 3)
            ee_quat = ee_positions[:, 3:]  # (T_stride, 4)
            ee_rotvec = quaternion_to_rotation_vector(ee_quat)  # (T_stride, 3)
            stride_state_data = np.concatenate([ee_xyz, ee_rotvec], axis=1)  # (T_stride, 6)

        # === 步骤3: 【可选】帧过滤 ===
        # 在 stride 采样后的数据上进行过滤
        if self.config.enable_frame_filtering:
            # 返回在 stride_cam_indices 中的索引位置
            keep_mask = self._filter_frames_by_motion(stride_state_data)
            final_cam_indices = [stride_cam_indices[i] for i in range(len(stride_cam_indices)) if keep_mask[i]]
            final_state_data = stride_state_data[keep_mask]
            final_gripper_width = aligned_robot_data['gripper_width'][keep_mask]
        else:
            final_cam_indices = stride_cam_indices
            final_state_data = stride_state_data
            final_gripper_width = aligned_robot_data['gripper_width']

        # === 步骤4: 处理数据 ===
        processed = {}
        processed['timestamps'] = camera_timestamps[final_cam_indices]  # 使用相机时间戳
        processed['state_data'] = final_state_data

        # 计算 action（在最终的 state 数据上计算相邻帧的差异）
        if action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.EE_POSE_GLOBAL]:
            # Action: 下一帧的状态
            processed['action_data'] = self._compute_next_frame_from_states(final_state_data)
        else:  # DELTA 模式
            # Action: 状态增量
            processed['action_data'] = self._compute_delta_from_states(final_state_data)

        # 夹爪处理
        processed['gripper'] = self._binarize_gripper(final_gripper_width)

        # === 步骤5: 处理视频帧 ===
        processed['video_frames'] = {}
        for cam_name, frames in video_frames.items():
            if self.config.enable_frame_filtering:
                # 计算 final_cam_indices 在 stride_cam_indices 中的位置
                video_indices = [stride_cam_indices.index(idx) for idx in final_cam_indices]
                processed['video_frames'][cam_name] = self._resize_frames(frames[video_indices])
            else:
                processed['video_frames'][cam_name] = self._resize_frames(frames[:len(final_cam_indices)])

        # === 步骤5.5: 处理触觉帧（使用相机时间戳对齐，包含延迟补偿） ===
        if tactile_frames is not None and len(tactile_frames) > 0 and tactile_timestamps is not None:
            aligned_tactile = self.align_tactile_to_camera(
                tactile_frames,
                tactile_timestamps,
                camera_timestamps,
                final_cam_indices  # 直接使用最终的相机帧索引
            )
            processed['tactile_frames'] = aligned_tactile
        else:
            # 创建占位触觉帧（空数组表示无触觉数据）
            processed['tactile_frames'] = np.array([])

        # === 步骤6: 计算 action_prev (state_t - state_t-1) ===
        processed['action_prev'] = self._compute_action_prev(final_state_data)

        return processed
    
    def _compute_next_frame_from_states(self, state_data: np.ndarray) -> np.ndarray:
        """从 state 序列计算下一帧 (用于 global action)
        
        Args:
            state_data: 已经过 stride 采样和过滤的 state 数据 (T, D)
        
        Returns:
            next_frames: 下一帧的 state (T, D)
        """
        next_frames = np.zeros_like(state_data)
        next_frames[:-1] = state_data[1:]  # 前 T-1 帧的 action 是下一帧的 state
        next_frames[-1] = state_data[-1]   # 最后一帧保持不变
        return next_frames.astype(np.float32)
    
    def _compute_delta_from_states(self, state_data: np.ndarray) -> np.ndarray:
        """从 state 序列计算增量 (用于 delta action)
        
        Args:
            state_data: 已经过 stride 采样和过滤的 state 数据 (T, D)
        
        Returns:
            deltas: 相邻帧的增量 (T, D)
        """
        deltas = np.zeros_like(state_data)
        deltas[:-1] = state_data[1:] - state_data[:-1]  # 前 T-1 帧的 delta
        deltas[-1] = 0.0  # 最后一帧增量为0
        return deltas.astype(np.float32)
    
    def _filter_frames_by_motion(self, stride_state_data: np.ndarray) -> np.ndarray:
        """在 stride 采样后的数据上，根据相邻帧的 state 变化幅度过滤关键帧
        
        标准流程: stride 采样 -> 帧过滤 -> 计算 action
        
        Args:
            stride_state_data: 已经过 stride 采样的 state 数据 (T_stride, D)
        
        Returns:
            keep_mask: 布尔数组，指示哪些帧应该保留 (T_stride,)
        """
        T = len(stride_state_data)
        if T < 2:
            return np.ones(T, dtype=bool)
        
        # 计算 stride 采样后相邻帧之间的 state 变化幅度
        motion_magnitudes = np.zeros(T)
        for i in range(T - 1):
            delta = stride_state_data[i + 1] - stride_state_data[i]
            motion_magnitudes[i] = np.linalg.norm(delta)
        # 最后一帧的幅度设为0（因为没有下一帧）
        motion_magnitudes[-1] = 0.0
        
        # 根据阈值过滤
        threshold = self.config.frame_filter_threshold
        keep_mask = motion_magnitudes >= threshold
        
        # 确保至少保留 min_frames_per_episode 帧
        n_valid = keep_mask.sum()
        min_frames = self.config.min_frames_per_episode
        
        if n_valid < min_frames:
            # 保留动作幅度最大的 min_frames 帧
            top_indices = np.argsort(motion_magnitudes)[-min_frames:]
            keep_mask = np.zeros(T, dtype=bool)
            keep_mask[top_indices] = True
        else:
            # 始终保留第一帧和最后一帧
            keep_mask[0] = True
            keep_mask[-1] = True
        
        # 记录过滤信息
        n_kept = keep_mask.sum()
        filter_ratio = n_kept / T if T > 0 else 0
        
        if n_kept < T:
            print(f"  [Frame Filter] {T} -> {n_kept} frames "
                  f"(kept {filter_ratio:.1%}, threshold={threshold:.6f})")
        
        return keep_mask
    
    def _compute_next_frame(self, data: np.ndarray, stride: int, frame_indices: List[int]) -> np.ndarray:
        """获取下一帧的数据 (用于 global action) - 已弃用，使用 _compute_next_frame_filtered"""
        next_frames = []
        for i in frame_indices:
            if i + stride < len(data):
                next_frame = data[i + stride]
            else:
                # 最后一帧，使用当前帧 (或者可以用前一帧的增量推断)
                next_frame = data[i]
            next_frames.append(next_frame)
        return np.array(next_frames, dtype=np.float32)
    
    def _compute_delta(self, data: np.ndarray, stride: int, frame_indices: List[int]) -> np.ndarray:
        """计算增量"""
        deltas = []
        for i in frame_indices:
            if i + stride < len(data):
                delta = data[i + stride] - data[i]
            else:
                delta = np.zeros_like(data[i])
            deltas.append(delta)
        return np.array(deltas, dtype=np.float32)
    
    def _binarize_gripper(self, gripper: np.ndarray) -> np.ndarray:
        """夹爪处理 - 可配置是否二值化"""
        if not self.config.enable_gripper_binarization:
            return gripper.astype(np.float32)

        threshold = self.config.gripper_binarization_threshold
        return np.where(gripper < threshold, 0.0, 1.0).astype(np.float32)
    
    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize 视频帧到目标尺寸，保持 (T, H, W, 3) 格式"""
        H, W = self.config.target_size
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            resized.append(resized_frame)
        return np.array(resized, dtype=np.uint8)

    def _compute_action_prev(self, state_data: np.ndarray) -> np.ndarray:
        """计算 action_prev = state[t] - state[t-1]

        边界情况处理：第一帧没有前一状态，action_prev[0] = 0

        Args:
            state_data: (T, D) 状态序列

        Returns:
            action_prev: (T, D) 其中 action_prev[0] = 0
        """
        T = len(state_data)
        action_prev = np.zeros_like(state_data)

        # action_prev[t] = state[t] - state[t-1]，第一帧保持为0
        action_prev[1:] = state_data[1:] - state_data[:-1]

        return action_prev.astype(np.float32)

    def align_robot_to_camera(
        self,
        robot_data: Dict[str, np.ndarray],
        camera_timestamps: np.ndarray,
        frame_indices: List[int]
    ) -> Dict[str, np.ndarray]:
        """使用相机时间戳对齐机器人数据

        统一以相机时间戳为基准，为每个相机帧找最近的机器人状态

        Args:
            robot_data: 机器人数据字典，包含 timestamps, joint_positions, ee_positions, gripper_width 等
            camera_timestamps: (T_camera,) 相机时间戳 (秒)
            frame_indices: 选中的相机帧索引列表

        Returns:
            aligned_robot_data: 对齐后的机器人数据字典
        """
        robot_ts = robot_data['timestamps'].astype(np.float64)

        # 为每个相机帧找最近的机器人状态索引
        aligned_indices = []
        for cam_idx in frame_indices:
            cam_ts = camera_timestamps[cam_idx]

            # 找最近的机器人时间戳
            time_diffs = np.abs(robot_ts - cam_ts)
            nearest_idx = np.argmin(time_diffs)

            # 如果对齐误差较大（超过 0.2 秒，约 3 帧间隔），打印警告
            if time_diffs[nearest_idx] > 0.2:
                print(f"  [Robot Align] Large gap: {time_diffs[nearest_idx]:.3f}s at cam_idx {cam_idx}")

            aligned_indices.append(nearest_idx)

        # 提取对齐后的数据
        aligned_data = {}
        aligned_data['timestamps'] = camera_timestamps[frame_indices]  # 使用相机时间戳作为最终时间戳
        aligned_data['joint_positions'] = robot_data['joint_positions'][aligned_indices]
        aligned_data['ee_positions'] = robot_data['ee_positions'][aligned_indices]
        aligned_data['gripper'] = robot_data['gripper'][aligned_indices]
        aligned_data['gripper_width'] = robot_data['gripper_width'][aligned_indices]

        return aligned_data

    def align_tactile_to_camera(
        self,
        tactile_frames: np.ndarray,
        tactile_timestamps: np.ndarray,
        camera_timestamps: np.ndarray,
        frame_indices: List[int]
    ) -> np.ndarray:
        """使用相机时间戳对齐触觉帧

        注意：触觉数据存在硬件延迟，实际采集时间 = 记录时间 - delay_offset
        因此在对齐时需要将触觉时间戳减去延迟补偿值

        Args:
            tactile_frames: (T_tactile, H, W, 3)
            tactile_timestamps: (T_tactile,) 触觉时间戳 (秒)
            camera_timestamps: (T_camera,) 相机时间戳 (秒)
            frame_indices: 选中的相机帧索引列表

        Returns:
            aligned_tactile: (T_final, H, W, 3) 对齐后的触觉帧
        """
        if len(tactile_frames) == 0:
            return np.array([])

        # 应用触觉硬件延迟补偿：实际采集时间 = 记录时间 - delay_offset
        # 例如：触觉记录时间为 t，实际是在 t-0.3s 时采集的
        # 所以要找相机时间 cam_ts 对应的触觉帧，应该找 tactile_ts - offset 最接近 cam_ts 的
        delay_offset = self.config.tactile_delay_offset
        corrected_tactile_ts = tactile_timestamps - delay_offset

        # 为每个相机帧找最近的触觉帧
        aligned_frames = []
        for cam_idx in frame_indices:
            cam_ts = camera_timestamps[cam_idx]

            # 找最近的触觉时间戳（使用校正后的时间戳）
            time_diffs = np.abs(corrected_tactile_ts - cam_ts)
            nearest_idx = np.argmin(time_diffs)

            # 如果对齐误差较大（超过 0.15 秒，约 1.5 个触觉帧间隔），打印警告
            if time_diffs[nearest_idx] > 0.15:
                print(f"  [Tactile Align] Large gap: {time_diffs[nearest_idx]:.3f}s at cam_idx {cam_idx} "
                      f"(delay_offset={delay_offset}s)")

            aligned_frames.append(tactile_frames[nearest_idx])

        return np.array(aligned_frames, dtype=np.uint8)


# ===== LeRobot 转换模块 =====

class LeRobotConverter:
    """LeRobot 数据集转换器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dataset: Optional[LeRobotDataset] = None
    
    def create_dataset(self, first_processed_data: Dict, first_meta: Dict) -> LeRobotDataset:
        """创建数据集骨架"""
        # 估计 FPS
        fps = self._estimate_fps(first_processed_data['timestamps'])
        
        # 根据动作空间确定维度
        action_space = self.config.action_space
        
        if action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.JOINT_POSITION_DELTA]:
            # 关节空间: 7 个关节 + 1 个夹爪 = 8
            state_dim = 8
            action_dim = 8
            state_names = [["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]]
            action_names = [["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]]
        else:  # EE_POSE_GLOBAL or EE_POSE_DELTA
            # 末端位姿空间: 6 (xyz + rotation_vector) + 1 个夹爪 = 7
            state_dim = 7
            action_dim = 7
            state_names = [["ee_x", "ee_y", "ee_z", "ee_rot_x", "ee_rot_y", "ee_rot_z", "gripper"]]
            action_names = [["ee_x", "ee_y", "ee_z", "ee_rot_x", "ee_rot_y", "ee_rot_z", "gripper"]]
        
        # 定义特征
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names,
            },
            "action_prev": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names,
            },
        }
        
        # 添加相机特征 - lerobot 0.1.0 使用 (H, W, C) 格式
        H, W, C = self.config.target_size[0], self.config.target_size[1], 3
        for cam_name in self.config.camera_names:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "image",
                "shape": (H, W, C),
                "names": ["height", "width", "channels"],
            }

        # 添加触觉特征
        tac_H, tac_W = self.config.tactile_size
        features["observation.images.gelsight_left_rgb"] = {
            "dtype": "image",
            "shape": (tac_H, tac_W, C),  # 触觉图像尺寸
            "names": ["height", "width", "channels"],
        }

        # 清理旧数据
        out_root = self.config.output_dir / self.config.repo_id
        if out_root.exists():
            import shutil
            shutil.rmtree(out_root)
        
        # 创建数据集（LeRobotDataset 会自动创建需要的目录）
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=fps,
            root=self.config.output_dir / self.config.repo_id,
            robot_type="franka",
            features=features,
            use_videos=False,
            image_writer_threads=8,
            image_writer_processes=8,
        )
        
        return self.dataset
    
    def add_episode(self, processed_data: Dict, meta: Dict):
        """添加一个 episode（批量添加帧以减少I/O）"""
        if self.dataset is None:
            raise RuntimeError("Dataset not created yet")

        # 构建 state、action 和 action_prev
        state = self._build_state(processed_data)
        action = self._build_action(processed_data)
        action_prev = self._build_action_prev(processed_data)

        # 获取任务描述
        task_str = self._get_task_string(meta)

        # 对齐长度
        T = min(len(state), len(action), len(action_prev))
        for cam_data in processed_data['video_frames'].values():
            T = min(T, len(cam_data))

        # 触觉数据对齐
        tactile_data = processed_data.get('tactile_frames', np.array([]))
        if len(tactile_data) > 0:
            T = min(T, len(tactile_data))

        # 批量添加帧 - lerobot 0.1.0 使用 (H, W, C) 图像格式
        for t in range(T):
            frame = {
                "observation.state": torch.from_numpy(state[t]),
                "action": torch.from_numpy(action[t]),
                "action_prev": torch.from_numpy(action_prev[t]),
                "task": task_str,
            }

            # 添加图像 - 直接使用 (H, W, C) 格式
            for cam_name in self.config.camera_names:
                frame[f"observation.images.{cam_name}"] = processed_data['video_frames'][cam_name][t]

            # 添加触觉图像 (resize 到配置的尺寸)
            if len(tactile_data) > 0:
                tactile_img = tactile_data[t]
                tac_H, tac_W = self.config.tactile_size
                tactile_resized = cv2.resize(tactile_img, (tac_W, tac_H), interpolation=cv2.INTER_AREA)
                frame["observation.images.gelsight_left_rgb"] = tactile_resized

            self.dataset.add_frame(frame)

        self.dataset.save_episode()
    
    def finalize(self):
        """完成数据集"""
        if self.dataset:
            # lerobot 0.1.0 数据已经在 save_episode 时保存，不需要额外步骤
            print(f"Dataset finalized with {self.dataset.num_episodes} episodes")
    
    def _build_state(self, processed_data: Dict) -> np.ndarray:
        """构建 state: state_data + gripper
        
        state_data 的内容取决于 action_space:
        - JOINT_POSITION_*: 当前关节位置 (7,) -> state (8,)
        - EE_POSE_*: 当前末端位姿 (6,) xyz + rotation_vector -> state (7,)
        """
        state_data = processed_data['state_data']  # (T, 7) for joint or (T, 6) for ee
        gripper = processed_data['gripper'][:, None]  # (T, 1)
        return np.concatenate([state_data, gripper], axis=1).astype(np.float32)
    
    def _build_action(self, processed_data: Dict) -> np.ndarray:
        """构建 action: action_data + gripper

        action_data 的内容取决于 action_space:
        - JOINT_POSITION_GLOBAL: 下一帧关节位置 (7,) -> action (8,)
        - JOINT_POSITION_DELTA: 关节位置增量 (7,) -> action (8,)
        - EE_POSE_GLOBAL: 下一帧末端位姿 (6,) xyz+rotvec -> action (7,)
        - EE_POSE_DELTA: 末端位姿增量 (6,) xyz+rotvec -> action (7,)
        """
        action_data = processed_data['action_data']  # (T, 7) for joint or (T, 6) for ee
        gripper = processed_data['gripper'][:, None]  # (T, 1)
        return np.concatenate([action_data, gripper], axis=1).astype(np.float32)

    def _build_action_prev(self, processed_data: Dict) -> np.ndarray:
        """构建 action_prev: (state[t] - state[t-1]) + 当前gripper状态

        第一帧 action_prev 中的位置部分为0，gripper 为当前值
        """
        action_prev_data = processed_data['action_prev']  # (T, 6) for ee_pose or (T, 7) for joint
        gripper = processed_data['gripper'][:, None]  # (T, 1)
        return np.concatenate([action_prev_data, gripper], axis=1).astype(np.float32)
    
    def _estimate_fps(self, timestamps: np.ndarray) -> float:
        """从时间戳估计 FPS"""
        if len(timestamps) < 2:
            return 30.0
        dt = float(np.median(np.diff(timestamps)))
        return 1.0 / dt if dt > 0 else 30.0
    
    def _get_task_string(self, meta: Dict) -> str:
        """提取任务描述字符串"""
        task_meta = meta.get("task_meta", {})
        task_name = task_meta.get("task_name", "unknown")

        # 优先使用配置中的 prompt
        if self.config.prompt is not None:
            prompt = self.config.prompt
        else:
            prompt = task_meta.get("prompt", "")

        robot_model = meta.get("robot_meta", {}).get("robots", [{}])[0].get("robot_model", "")

        return f"{task_name} | {prompt} | robot={robot_model}".strip()


# ===== 主流程 =====

def _check_tactile_complete(episode_path: Path) -> bool:
    """检查触觉数据是否完整

    Returns:
        True 如果 gelsight 视频和时间戳都存在
    """
    video_path = episode_path / "v1" / "gelsight" / "gelsight_left_rgb.mp4"
    ts_path = episode_path / "v1" / "origin_data" / "gelsight_left_timestamps.txt"

    return video_path.exists() and ts_path.exists()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="Franka to LeRobot Converter with Tactile Support")
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt for all episodes")
    parser.add_argument("--input", type=Path, nargs='+', help="Input directories (multiple supported)")
    parser.add_argument("--output", type=Path, default=None, help="Output root directory")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo ID")
    parser.add_argument("--no-filter", action="store_true", help="Disable frame filtering")
    parser.add_argument("--filter-threshold", type=float, default=1e-10, help="Frame filter threshold")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride")
    parser.add_argument("--no-binarize-gripper", action="store_false", dest="binarize_gripper",
                        help="Disable gripper binarization")
    parser.add_argument("--gripper-threshold", type=float, default=0.079,
                        help="Threshold for gripper binarization (default: 0.079)")
    parser.add_argument("--no-require-tactile", action="store_false", dest="require_tactile",
                        help="Don't require tactile data (process episodes without tactile)")
    parser.add_argument("--tactile-size", type=int, nargs=2, default=[128, 160],
                        help="Tactile image size (H W), default: 128 160")
    parser.add_argument("--tactile-delay", type=float, default=0.3,
                        help="Tactile hardware delay offset in seconds (default: 0.3)")
    parser.set_defaults(binarize_gripper=True, require_tactile=True)

    args = parser.parse_args()

    config = Config()

    # 应用命令行参数
    if args.prompt is not None:
        config.prompt = args.prompt

    if args.input:
        config.source_paths = args.input

    if args.output:
        config.output_dir = args.output

    if args.repo_id:
        config.repo_id = args.repo_id

    if args.no_filter:
        config.enable_frame_filtering = False

    config.frame_filter_threshold = args.filter_threshold
    config.stride = args.stride
    config.enable_gripper_binarization = args.binarize_gripper
    config.gripper_binarization_threshold = args.gripper_threshold
    config.require_tactile = args.require_tactile
    config.tactile_size = tuple(args.tactile_size)
    config.tactile_delay_offset = args.tactile_delay

    print(f"Action space: {config.action_space.value}")
    print(f"Output repo: {config.repo_id}")
    print(f"Source paths: {config.source_paths}")
    print(f"Tactile delay offset: {config.tactile_delay_offset}s (tactile_real_ts = recorded_ts - {config.tactile_delay_offset}s)")

    # 初始化各模块
    loader = FrankaDataLoader(config)
    processor = DataProcessor(config)
    converter = LeRobotConverter(config)

    # 获取所有 episodes
    all_episode_paths = loader.get_all_episodes()
    print(f"Total episodes found: {len(all_episode_paths)}")

    # 根据配置过滤 episodes
    episode_paths = []
    skipped_episodes = []

    if config.require_tactile:
        for ep_path in all_episode_paths:
            if _check_tactile_complete(ep_path):
                episode_paths.append(ep_path)
            else:
                skipped_episodes.append(ep_path)
        print(f"  - Tactile data complete: {len(episode_paths)} episodes")
        print(f"  - Skipped (incomplete tactile): {len(skipped_episodes)} episodes")
    else:
        episode_paths = all_episode_paths
        print(f"  - Tactile not required, processing all {len(episode_paths)} episodes")

    if not episode_paths:
        raise ValueError("No valid episodes found")

    # 处理第一个 episode 以创建数据集骨架
    print(f"\nProcessing first episode: {episode_paths[0]}")
    first_meta = loader.load_meta(episode_paths[0])
    first_robot_data = loader.load_robot_data(episode_paths[0])

    # 加载触觉数据
    first_tactile_frames, first_tactile_ts = loader.load_tactile_video(episode_paths[0])

    # 加载相机时间戳 (作为主时间基准)
    first_camera_ts = loader.load_camera_timestamps(episode_paths[0], "main_realsense")

    # 计算帧索引 (基于相机时间戳，而不是机器人数据)
    T_camera = len(first_camera_ts)
    frame_indices = list(range(0, T_camera - config.stride, config.stride))

    first_video_frames = loader.load_video_frames(episode_paths[0], frame_indices)
    first_processed = processor.process_episode(
        first_robot_data,
        first_video_frames,
        camera_timestamps=first_camera_ts,
        tactile_frames=first_tactile_frames,
        tactile_timestamps=first_tactile_ts
    )

    # 创建数据集
    converter.create_dataset(first_processed, first_meta)

    # 转换所有 episodes (带错误处理)
    success_count = 0
    error_count = 0

    for episode_path in tqdm.tqdm(episode_paths, desc="Converting episodes"):
        try:
            meta = loader.load_meta(episode_path)
            robot_data = loader.load_robot_data(episode_path)

            # 加载触觉数据
            tactile_frames, tactile_ts = loader.load_tactile_video(episode_path)

            # 加载相机时间戳 (作为主时间基准)
            camera_ts = loader.load_camera_timestamps(episode_path, "main_realsense")

            # 计算帧索引 (基于相机时间戳)
            T_camera = len(camera_ts)
            frame_indices = list(range(0, T_camera - config.stride, config.stride))

            video_frames = loader.load_video_frames(episode_path, frame_indices)
            processed_data = processor.process_episode(
                robot_data,
                video_frames,
                camera_timestamps=camera_ts,
                tactile_frames=tactile_frames,
                tactile_timestamps=tactile_ts
            )

            converter.add_episode(processed_data, meta)
            success_count += 1
        except Exception as e:
            print(f"\nError processing episode {episode_path}: {e}")
            error_count += 1
            continue

    # 完成
    converter.finalize()

    out_dir = config.output_dir / config.repo_id
    print(f"\n✓ Done! Dataset saved to: {out_dir}")
    print(f"  Action space used: {config.action_space.value}")
    print(f"\n  Episodes converted: {success_count}")
    print(f"  Episodes failed: {error_count}")
    if config.require_tactile:
        print(f"  Episodes skipped (incomplete tactile): {len(skipped_episodes)}")


if __name__ == "__main__":
    main()
