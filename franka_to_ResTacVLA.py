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
    # 数据路径
    data_root: Path = Path("/home/zpw/ws_zpw/vla/data/2025_11_18")
    task_folder: str = "peg_in_hole1"
    
    # 输出配置
    repo_id: str = "franka/peg_in_hole"
    output_dir: Path = Path.home() / "ws_zpw" / "vla" / "data" / "lerobot_data" # 数据集保存目录
    target_size: Tuple[int, int] = (224, 224)  # (H, W)
    
    # 动作空间配置
    action_space: ActionSpace = ActionSpace.EE_POSE_DELTA
    
    # 数据处理
    stride: int = 1  # 采样间隔
    
    # 帧过滤配置
    enable_frame_filtering: bool = True  # 是否启用帧过滤
    frame_filter_threshold: float = 1e-10  # 帧过滤阈值：state 变化幅度 (默认只过滤静止帧)
    min_frames_per_episode: int = 10  # 每个 episode 最少保留的帧数
    
    # None 表示转换所有，否则转换前n个s
    max_episodes: Optional[int] = None
    
    # 相机配置
    camera_names: List[str] = None  # None表示自动检测
    
    def __post_init__(self):
        if self.camera_names is None:
            self.camera_names = ["main_realsense_rgb", "side_realsense_rgb", "handeye_realsense_rgb"]
        
        # 根据动作空间自动更新 repo_id
        action_suffix = self.action_space.value
        if action_suffix not in self.repo_id:
            self.repo_id = f"{self.repo_id}_{action_suffix}"


# ===== 数据读取模块 =====

class FrankaDataLoader:
    """Franka 数据加载器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_root = config.data_root / config.task_folder
        
    def get_all_episodes(self) -> List[str]:
        """获取所有 episode 时间文件夹"""
        if not self.data_root.exists():
            raise ValueError(f"Task folder not found: {self.data_root}")
        
        episodes = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        
        if self.config.max_episodes:
            episodes = episodes[:self.config.max_episodes]
            
        return episodes
    
    def load_meta(self, episode: str) -> Dict:
        """加载 meta.json"""
        meta_path = self.data_root / episode / "v1" / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_robot_data(self, episode: str) -> Dict[str, np.ndarray]:
        """从 jsonl 加载机器人数据"""
        jsonl_path = self.data_root / episode / "v1" / "data" / "Franka_4_arms_arm.jsonl"
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Robot data not found: {jsonl_path}")
        
        with jsonlines.open(jsonl_path) as reader:
            data = list(reader)
        
        # 提取字段
        timestamps = np.array([d['timestamp'] for d in data], dtype=np.float32)
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
    
    def load_video_frames(self, episode: str, frame_indices: List[int]) -> Dict[str, np.ndarray]:
        """加载指定帧的图像（并行加载多个相机）"""
        video_dir = self.data_root / episode / "v1" / "videos"
        
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

    def load_tactile_video(self, episode: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载 GelSight 触觉视频和对应的时间戳

        Returns:
            tactile_frames: (T_tactile, H, W, 3) RGB uint8
            tactile_timestamps: (T_tactile,) float32 Unix时间戳
        """
        video_dir = self.data_root / episode / "v1" / "gelsight"
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

        # 读取时间戳文件
        ts_path = self.data_root / episode / "v1" / "origin_data" / "gelsight_left_timestamps.txt"
        if ts_path.exists():
            tactile_timestamps = np.loadtxt(ts_path, dtype=np.float32)
        else:
            # 备用：从视频FPS估算
            fps = 10.0  # GelSight 标准频率
            tactile_timestamps = np.arange(len(frames), dtype=np.float32) / fps

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
        tactile_frames: Optional[np.ndarray] = None,
        tactile_timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """处理单个 episode 的数据

        处理流程:
        1. 步骤1: Stride 采样 - 从原始数据中按 stride 间隔采样
        2. 步骤2: 【可选】帧过滤 - 在 stride 采样后的数据上，根据相邻帧的 state 变化过滤关键帧
        3. 步骤3: 计算 action - 基于过滤后的帧计算 action (delta 或 next_frame)
        4. 步骤4: 处理 gripper、视频帧、和触觉帧
        """
        stride = self.config.stride
        action_space = self.config.action_space
        
        # === 步骤1: Stride 采样 ===
        T_raw = len(robot_data['timestamps'])
        stride_indices = list(range(0, T_raw - stride, stride))
        
        # 检查视频帧数量是否匹配
        first_cam_frames = list(video_frames.values())[0]
        if len(first_cam_frames) != len(stride_indices):
            # 视频帧数量与预期不符，裁剪到最小长度
            min_len = min(len(stride_indices), len(first_cam_frames))
            stride_indices = stride_indices[:min_len]
        
        # 根据动作空间提取原始 state 数据
        if action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.JOINT_POSITION_DELTA]:
            # 使用关节空间
            raw_state_data = robot_data['joint_positions']  # (T_raw, 7)
        elif action_space in [ActionSpace.EE_POSE_GLOBAL, ActionSpace.EE_POSE_DELTA]:
            # 使用末端位姿空间 - 需要先转换为 xyz + rotation_vector
            ee_positions = robot_data['ee_positions']  # (T_raw, 7) - xyz(3) + quaternion(4)
            ee_xyz = ee_positions[:, :3]  # (T_raw, 3)
            ee_quat = ee_positions[:, 3:]  # (T_raw, 4)
            ee_rotvec = quaternion_to_rotation_vector(ee_quat)  # (T_raw, 3)
            raw_state_data = np.concatenate([ee_xyz, ee_rotvec], axis=1)  # (T_raw, 6)
        
        # 先进行 stride 采样
        stride_state_data = raw_state_data[stride_indices]  # (T_stride, D)
        
        # === 步骤2: 【可选】帧过滤 ===
        # 在 stride 采样后的数据上进行过滤
        if self.config.enable_frame_filtering:
            # 返回在 stride_indices 中的索引位置
            keep_mask = self._filter_frames_by_motion(stride_state_data)
            final_indices = [stride_indices[i] for i in range(len(stride_indices)) if keep_mask[i]]
            final_state_data = stride_state_data[keep_mask]
        else:
            final_indices = stride_indices
            final_state_data = stride_state_data
        
        # === 步骤3: 处理数据 ===
        processed = {}
        processed['timestamps'] = robot_data['timestamps'][final_indices]
        processed['state_data'] = final_state_data
        
        # 计算 action（在最终的 state 数据上计算相邻帧的差异）
        if action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.EE_POSE_GLOBAL]:
            # Action: 下一帧的状态
            processed['action_data'] = self._compute_next_frame_from_states(final_state_data)
        else:  # DELTA 模式
            # Action: 状态增量
            processed['action_data'] = self._compute_delta_from_states(final_state_data)
        
        # 夹爪处理
        gripper_width = robot_data['gripper_width']
        gripper_sampled = gripper_width[final_indices]
        processed['gripper'] = self._binarize_gripper(gripper_sampled)
        
        # === 步骤4: 处理视频帧 ===
        processed['video_frames'] = {}
        for cam_name, frames in video_frames.items():
            if self.config.enable_frame_filtering:
                # 计算 final_indices 在 stride_indices 中的位置
                video_indices = [stride_indices.index(idx) for idx in final_indices]
                processed['video_frames'][cam_name] = self._resize_frames(frames[video_indices])
            else:
                processed['video_frames'][cam_name] = self._resize_frames(frames[:len(final_indices)])

        # === 步骤4.5: 处理触觉帧（时间对齐） ===
        if tactile_frames is not None and len(tactile_frames) > 0:
            aligned_tactile = self.align_tactile_to_robot(
                tactile_frames,
                tactile_timestamps,
                robot_data['timestamps'],
                final_indices
            )
            processed['tactile_frames'] = aligned_tactile
        else:
            # 创建占位触觉帧（空数组表示无触觉数据）
            processed['tactile_frames'] = np.array([])

        # === 步骤5: 计算 action_prev (state_t - state_t-1) ===
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
        """夹爪二值化 - 基于数据分析确定阈值
        
        数据分析结果 (基于 50 个 episodes, 49587 个数据点):
        - 双峰分布:
          * 合 (closed): 19-24mm (57.3% 的数据)
          * 开 (open): 70-85mm (40.6% 的数据)
          * 中间过渡: 25-70mm (仅 2.1% 的数据)
        
        - 最常见的值:
          * 85.00mm: 39.3% (完全开启)
          * 20.22mm: 31.5% (抓紧物体)
          * 19.85mm: 23.8% (最紧)
        """
        threshold = 0.025  # 25mm
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

    def align_tactile_to_robot(
        self,
        tactile_frames: np.ndarray,
        tactile_timestamps: np.ndarray,
        robot_timestamps: np.ndarray,
        final_indices: List[int]
    ) -> np.ndarray:
        """使用最近邻插值将触觉帧对齐到最终的机器人时间线

        Args:
            tactile_frames: (T_tactile, H, W, 3)
            tactile_timestamps: (T_tactile,) 触觉时间戳
            robot_timestamps: (T_raw,) 机器人原始时间戳
            final_indices: 最终选中的帧索引列表（从 process_episode 中获取）

        Returns:
            aligned_tactile: (T_final, H, W, 3) 对齐后的触觉帧
        """
        if len(tactile_frames) == 0:
            return np.array([])

        # 为每个最终帧找最近的触觉帧
        aligned_frames = []
        for robot_idx in final_indices:
            robot_ts = robot_timestamps[robot_idx]

            # 找最近的触觉时间戳
            time_diffs = np.abs(tactile_timestamps - robot_ts)
            nearest_idx = np.argmin(time_diffs)

            # 如果对齐误差较大，打印警告
            if time_diffs[nearest_idx] > 0.1:
                print(f"  [Tactile Align] Large gap: {time_diffs[nearest_idx]:.3f}s at robot_idx {robot_idx}")

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
        features["observation.images.gelsight_left_rgb"] = {
            "dtype": "image",
            "shape": (128, 160, C),  # GelSight 原生分辨率
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

            # 添加触觉图像
            if len(tactile_data) > 0:
                frame["observation.images.gelsight_left_rgb"] = tactile_data[t]

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
        # 使用固定的 prompt，不从 meta.json 读取
        prompt = "wipe the pen marks off the plate"
        robot_model = meta.get("robot_meta", {}).get("robots", [{}])[0].get("robot_model", "")

        return f"{task_name} | {prompt} | robot={robot_model}".strip()


# ===== 主流程 =====

def _check_tactile_complete(loader: FrankaDataLoader, episode: str) -> bool:
    """检查触觉数据是否完整

    Returns:
        True 如果 gelsight 视频和时间戳都存在
    """
    video_path = loader.data_root / episode / "v1" / "gelsight" / "gelsight_left_rgb.mp4"
    ts_path = loader.data_root / episode / "v1" / "origin_data" / "gelsight_left_timestamps.txt"

    return video_path.exists() and ts_path.exists()


def main():
    """主函数"""
    config = Config()

    # 示例: 如何修改动作空间
    # config.action_space = ActionSpace.JOINT_POSITION_GLOBAL
    # config.action_space = ActionSpace.EE_POSE_GLOBAL
    # config.action_space = ActionSpace.EE_POSE_DELTA

    print(f"Action space: {config.action_space.value}")
    print(f"Output repo: {config.repo_id}")

    # 初始化各模块
    loader = FrankaDataLoader(config)
    processor = DataProcessor(config)
    converter = LeRobotConverter(config)

    # 获取所有 episodes
    all_episodes = loader.get_all_episodes()
    print(f"Found {len(all_episodes)} episodes")

    # 过滤出触觉数据完整的 episodes
    episodes = []
    skipped_episodes = []
    for ep in all_episodes:
        if _check_tactile_complete(loader, ep):
            episodes.append(ep)
        else:
            skipped_episodes.append(ep)

    print(f"  - Tactile data complete: {len(episodes)} episodes")
    print(f"  - Skipped (incomplete tactile): {len(skipped_episodes)} episodes")

    if not episodes:
        raise ValueError("No episodes with complete tactile data found")

    # 处理第一个 episode 以创建数据集骨架
    print(f"\nProcessing first episode: {episodes[0]}")
    first_meta = loader.load_meta(episodes[0])
    first_robot_data = loader.load_robot_data(episodes[0])

    # 加载触觉数据
    first_tactile_frames, first_tactile_ts = loader.load_tactile_video(episodes[0])

    # 计算帧索引
    T_raw = len(first_robot_data['timestamps'])
    frame_indices = list(range(0, T_raw - config.stride, config.stride))

    first_video_frames = loader.load_video_frames(episodes[0], frame_indices)
    first_processed = processor.process_episode(
        first_robot_data,
        first_video_frames,
        tactile_frames=first_tactile_frames,
        tactile_timestamps=first_tactile_ts
    )

    # 创建数据集
    converter.create_dataset(first_processed, first_meta)

    # 转换所有 episodes
    for episode in tqdm.tqdm(episodes, desc="Converting episodes"):
        meta = loader.load_meta(episode)
        robot_data = loader.load_robot_data(episode)

        # 加载触觉数据
        tactile_frames, tactile_ts = loader.load_tactile_video(episode)

        # 计算帧索引
        T_raw = len(robot_data['timestamps'])
        frame_indices = list(range(0, T_raw - config.stride, config.stride))

        video_frames = loader.load_video_frames(episode, frame_indices)
        processed_data = processor.process_episode(
            robot_data,
            video_frames,
            tactile_frames=tactile_frames,
            tactile_timestamps=tactile_ts
        )

        converter.add_episode(processed_data, meta)

    # 完成
    converter.finalize()

    out_dir = config.output_dir / config.repo_id
    print(f"\n✓ Done! Dataset saved to: {out_dir}")
    print(f"  Action space used: {config.action_space.value}")
    print(f"\n  Episodes converted: {len(episodes)}")
    print(f"  Episodes skipped (incomplete tactile): {len(skipped_episodes)}")


if __name__ == "__main__":
    main()
