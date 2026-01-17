# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
使用远程策略服务器运行 Isaac Lab 环境的脚本（支持相机输入）。

该脚本将环境观察（包括相机图像）发送到远程策略服务器，接收动作并在 Isaac Lab 仿真环境中执行。
支持与 OpenPI 等 VLA 策略服务器配合使用。

使用方法：
    # 1. 在有 GPU 的机器上启动策略服务器
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_kinova --policy.dir=<checkpoint_path>

    # 2. 运行此脚本（使用带相机的任务）
    python scripts/remote_policy/play.py \\
        --task Gen3-Reach-Camera-v0 \\
        --remote-host <server_ip> \\
        --remote-port 8000 \\
        --prompt "reach the target"

相机说明：
    - 使用 Gen3-Reach-Camera-v0 任务，该任务包含两个相机：
      - external_camera: 外部固定相机（第三人称视角）
      - wrist_camera: 腕部相机（末端执行器视角）
    - 相机图像会自动 resize 到 224x224 发送给策略服务器
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with remote policy server in Isaac Lab environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--sim-dt",
    type=float,
    default=None,
    help="Override simulation timestep (dt) in seconds. If not set, uses default from environment config.",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Override decimation value. If not set, uses default from environment config.",
)

# Remote policy server arguments
parser.add_argument("--remote-host", type=str, default="0.0.0.0", help="Remote policy server IP address.")
parser.add_argument("--remote-port", type=int, default=8000, help="Remote policy server port.")
parser.add_argument(
    "--open-loop-horizon",
    type=int,
    default=1,
    help="Number of actions to execute from each action chunk before querying server again.",
)
parser.add_argument("--max-timesteps", type=int, default=1000, help="Maximum number of timesteps to run.")

# Task prompt for VLA models
parser.add_argument(
    "--prompt",
    type=str,
    default="",
    help="Task prompt/instruction to send to the policy server (for VLA models).",
)

# Camera configuration
parser.add_argument(
    "--image-size",
    type=int,
    default=224,
    help="Size to resize camera images to (images will be resized to image_size x image_size).",
)
parser.add_argument(
    "--external-camera-key",
    type=str,
    default="external_camera",
    help="Key name for external camera in the scene.",
)
parser.add_argument(
    "--wrist-camera-key",
    type=str,
    default="wrist_camera",
    help="Key name for wrist camera in the scene.",
)
parser.add_argument(
    "--save-debug-images",
    action="store_true",
    default=False,
    help="Save debug images to disk.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import numpy as np
import torch
from PIL import Image

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import gen3.tasks  # noqa: F401

# Optional: Import image tools if your policy uses images
try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    HAS_OPENPI_CLIENT = True
except ImportError:
    HAS_OPENPI_CLIENT = False
    print("[WARNING] openpi_client not found. Install with: pip install openpi-client")
    print("[WARNING] Falling back to simple WebSocket client mode.")


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """
    临时阻止键盘中断，延迟处理直到受保护的代码执行完毕。
    用于防止在等待策略服务器响应时被 Ctrl+C 中断导致连接问题。
    """
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class SimpleWebsocketPolicyClient:
    """
    简单的 WebSocket 策略客户端，作为 openpi_client 不可用时的备选方案。
    使用与 OpenPI 服务器兼容的 msgpack 序列化格式。
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._connect()

    @staticmethod
    def _pack_array(obj):
        """OpenPI 兼容的 numpy 数组序列化"""
        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }
        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }
        return obj

    @staticmethod
    def _unpack_array(obj):
        """OpenPI 兼容的 numpy 数组反序列化"""
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    def _connect(self):
        import websockets.sync.client
        import msgpack
        
        print(f"[INFO] Connecting to WebSocket server at {self._uri}...")
        self._ws = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None
        )
        # 接收服务器元数据
        metadata = msgpack.unpackb(self._ws.recv(), object_hook=self._unpack_array)
        print(f"[INFO] Connected! Server metadata: {metadata}")

    def infer(self, request_data: dict) -> dict:
        import msgpack

        # 使用 OpenPI 兼容的 msgpack 序列化
        data = msgpack.packb(request_data, default=self._pack_array)
        self._ws.send(data)
        response = self._ws.recv()
        
        if isinstance(response, str):
            raise RuntimeError(f"Error from server: {response}")
        
        result = msgpack.unpackb(response, object_hook=self._unpack_array)
        return result


def resize_image_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    将图像 resize 到目标尺寸，保持长宽比并用黑色填充。

    Args:
        image: 输入图像 [H, W, C] 格式
        target_height: 目标高度
        target_width: 目标宽度

    Returns:
        resize 后的图像
    """
    if HAS_OPENPI_CLIENT:
        return image_tools.resize_with_pad(image, target_height, target_width)

    # Fallback implementation
    from PIL import Image

    pil_image = Image.fromarray(image)
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create padded image
    result = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    result.paste(pil_image, (paste_x, paste_y))

    return np.array(result)


def get_camera_images(env, env_idx: int = 0, timestep: int = 0) -> dict:
    """
    从环境中获取相机图像。

    Args:
        env: Isaac Lab 环境
        env_idx: 环境索引
        timestep: 当前时间步（用于调试）

    Returns:
        包含相机图像的字典
    """
    # #region agent log
    import json
    import pathlib
    LOG_DIR = pathlib.Path(__file__).parent.parent.parent / ".cursor"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = str(LOG_DIR / "debug.log")
    # #endregion
    
    images = {}
    unwrapped_env = env.unwrapped

    if not hasattr(unwrapped_env, "scene"):
        return images

    scene = unwrapped_env.scene
    sensors_dict = getattr(scene, "sensors", {}) or {}
    
    # #region agent log
    # 假设1: 检查是否需要手动 update 相机
    if timestep < 5:  # 只在前5帧记录详细日志
        for cam_key in [args_cli.external_camera_key, args_cli.wrist_camera_key]:
            if cam_key in sensors_dict:
                cam = sensors_dict[cam_key]
                log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H1_camera_update", "location": "play.py:get_camera_images", "message": f"Camera {cam_key} state", "data": {"timestep": timestep, "has_update_method": hasattr(cam, 'update'), "cam_type": str(type(cam))}}
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    # #endregion

    def _get_camera_rgb(camera, cam_name=""):
        """从相机传感器获取 RGB 图像"""
        if camera is None or not hasattr(camera, "data"):
            return None
        camera_data = camera.data
        rgb_data = None
        if hasattr(camera_data, "output"):
            rgb_data = camera_data.output.get("rgb")
        elif hasattr(camera_data, "rgb"):
            rgb_data = camera_data.rgb
        if rgb_data is not None:
            img = rgb_data[env_idx].cpu().numpy()
            
            # #region agent log
            # 假设2/3: 检查图像数据是否变化
            if timestep < 10:
                img_hash = hash(img.tobytes()[:1000])  # 只哈希前1000字节
                img_mean = float(img.mean())
                img_std = float(img.std())
                log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H2_image_data", "location": "play.py:_get_camera_rgb", "message": f"Image data for {cam_name}", "data": {"timestep": timestep, "img_hash": img_hash, "img_mean": img_mean, "img_std": img_std, "shape": list(img.shape), "rgb_data_id": id(rgb_data)}}
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            # #endregion
            
            if img.shape[-1] == 4:
                img = img[..., :3]
            return img.astype(np.uint8)
        return None

    # 获取物理时间步长用于更新传感器
    dt = getattr(unwrapped_env, "physics_dt", 1.0 / 60.0)
    
    # #region agent log
    # H7/H8/H9: 尝试不同的更新方式
    if timestep < 3:
        # 检查相机配置
        for cam_key in [args_cli.external_camera_key, args_cli.wrist_camera_key]:
            if cam_key in sensors_dict:
                cam = sensors_dict[cam_key]
                cfg_info = {}
                if hasattr(cam, 'cfg'):
                    cfg = cam.cfg
                    cfg_info["update_period"] = getattr(cfg, 'update_period', 'N/A')
                    cfg_info["data_types"] = getattr(cfg, 'data_types', 'N/A')
                log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H8_config", "location": "play.py:get_camera_images", "message": f"Camera {cam_key} config", "data": {"timestep": timestep, "cfg": cfg_info}}
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    # #endregion
    
    # H7: 尝试先渲染
    if hasattr(unwrapped_env, 'sim') and hasattr(unwrapped_env.sim, 'render'):
        unwrapped_env.sim.render()
    
    # H9: 尝试更新整个场景
    if hasattr(scene, 'update'):
        scene.update(dt)
    
    # 通过 sensors 字典访问相机
    if args_cli.external_camera_key in sensors_dict:
        cam = sensors_dict[args_cli.external_camera_key]
        # 手动更新相机传感器以刷新图像数据
        if hasattr(cam, 'update'):
            cam.update(dt)
        img = _get_camera_rgb(cam, "external")
        if img is not None:
            images["external_image"] = img

    if args_cli.wrist_camera_key in sensors_dict:
        cam = sensors_dict[args_cli.wrist_camera_key]
        # 手动更新相机传感器以刷新图像数据
        if hasattr(cam, 'update'):
            cam.update(dt)
        img = _get_camera_rgb(cam, "wrist")
        if img is not None:
            images["wrist_image"] = img

    return images


def get_robot_state(env, env_idx: int = 0) -> dict:
    """
    从环境中获取机器人状态。

    Args:
        env: Isaac Lab 环境
        env_idx: 环境索引

    Returns:
        包含机器人状态的字典
    """
    state = {}
    unwrapped_env = env.unwrapped

    if not hasattr(unwrapped_env, "scene"):
        return state

    scene = unwrapped_env.scene
    robot = None

    # 通过 scene.robot 或 articulations 字典获取机器人
    if hasattr(scene, "robot"):
        robot = scene.robot
    elif hasattr(scene, "articulations") and isinstance(scene.articulations, dict):
        if "robot" in scene.articulations:
            robot = scene.articulations["robot"]

    if robot is not None and hasattr(robot, 'data'):
        # 获取关节位置
        if hasattr(robot.data, "joint_pos"):
            state["joint_position"] = robot.data.joint_pos[env_idx].cpu().numpy()
            state["joint_position_target"] = robot.data.joint_pos_target[env_idx].cpu().numpy()

        # 获取关节速度
        # if hasattr(robot.data, "joint_vel"):
        #     state["joint_velocity"] = robot.data.joint_vel[env_idx].cpu().numpy()

        # 获取末端执行器位置
        if hasattr(robot.data, "body_pos_w"):
            state["ee_position"] = robot.data.body_pos_w[env_idx, -1].cpu().numpy()

    return state


def prepare_request_data(
        camera_images: dict,
        robot_state: dict,
        prompt: str = "",
        image_size: int = 224,
        gripper_position: float = 1.0,
) -> dict:
    """
    准备发送到策略服务器的请求数据。

    Args:
        camera_images: 相机图像字典
        robot_state: 机器人状态字典
        prompt: 任务提示
        image_size: 图像 resize 尺寸

    Returns:
        准备好的请求数据字典
    """
    request_data = {}

    # 处理相机图像
    if "external_image" in camera_images:
        external_img = resize_image_with_pad(camera_images["external_image"], image_size, image_size)
        request_data["observation/exterior_image_1_left"] = external_img

    if "wrist_image" in camera_images:
        wrist_img = resize_image_with_pad(camera_images["wrist_image"], image_size, image_size)
        request_data["observation/wrist_image_left"] = wrist_img

    # 处理机器人状态
    if "joint_position" in robot_state:
        request_data["observation/joint_position"] = robot_state["joint_position"]

    if "joint_velocity" in robot_state:
        request_data["observation/joint_velocity"] = robot_state["joint_velocity"]

    # 模拟夹爪位置（如果策略需要）
    # 注意：Kinova Gen3 的夹爪可能需要单独处理
    request_data["observation/gripper_position"] = np.array([gripper_position])

    # 添加任务提示
    if prompt:
        request_data["prompt"] = prompt

    return request_data


def save_debug_images(camera_images: dict, step: int):
    """保存调试图像到磁盘。"""
    os.makedirs("debug_images", exist_ok=True)

    if "external_image" in camera_images:
        img = Image.fromarray(camera_images["external_image"])
        img.save(f"debug_images/external_step_{step:04d}.png")

    if "wrist_image" in camera_images:
        img = Image.fromarray(camera_images["wrist_image"])
        img.save(f"debug_images/wrist_step_{step:04d}.png")

    # 创建组合图像
    if "external_image" in camera_images and "wrist_image" in camera_images:
        combined = np.concatenate(
            [camera_images["external_image"], camera_images["wrist_image"]], axis=1
        )
        combined_img = Image.fromarray(combined)
        combined_img.save(f"debug_images/combined_step_{step:04d}.png")


def main():
    """Play with remote policy server."""
    # Create policy client
    if HAS_OPENPI_CLIENT:
        print(f"[INFO] Connecting to remote policy server: {args_cli.remote_host}:{args_cli.remote_port}")
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            args_cli.remote_host, args_cli.remote_port
        )
        print("[INFO] Connected to policy server!")
    else:
        print(f"[INFO] Using simple WebSocket client: {args_cli.remote_host}:{args_cli.remote_port}")
        policy_client = SimpleWebsocketPolicyClient(args_cli.remote_host, args_cli.remote_port)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Override physics parameters if specified
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = args_cli.sim_dt
        print(f"[INFO] Overriding sim.dt to {args_cli.sim_dt}")
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
        print(f"[INFO] Overriding decimation to {args_cli.decimation}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_dir = os.path.join("logs", "remote_policy", "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # print info
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Action space: {env.action_space.shape}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Open-loop horizon: {args_cli.open_loop_horizon}")
    print(f"[INFO] Max timesteps: {args_cli.max_timesteps}")
    if args_cli.prompt:
        print(f"[INFO] Prompt: {args_cli.prompt}")

    # Check for cameras
    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, "scene"):
        scene = unwrapped_env.scene
        sensors_dict = getattr(scene, "sensors", {}) or {}
        has_external = args_cli.external_camera_key in sensors_dict
        has_wrist = args_cli.wrist_camera_key in sensors_dict
        
        print(f"[INFO] External camera: {'Found' if has_external else 'Not found'}")
        print(f"[INFO] Wrist camera: {'Found' if has_wrist else 'Not found'}")

        if not has_external and not has_wrist:
            print("[WARNING] No cameras found! Make sure to use a task with camera sensors.")

    dt = env.unwrapped.physics_dt
    
    print(f"[INFO] Simulation timestep: {dt}")
    print(f"[INFO] Decimation: {env_cfg.decimation}")
    # Action chunk management
    pred_action_chunk = None
    actions_from_chunk_completed = 0

    # reset environment
    obs, info = env.reset()

    timestep = 0

    print("\n[INFO] Starting evaluation loop...")
    print("[INFO] Press Ctrl+C to stop.\n")

    # simulate environment
    gripper_position = 1.0
    while simulation_app.is_running() and timestep < args_cli.max_timesteps:
        start_time = time.time()

        try:
            # run everything in inference mode
            with torch.inference_mode():
                # #region agent log
                # 假设4: 检查 step 前后的差异
                import json
                import pathlib
                LOG_DIR = pathlib.Path(__file__).parent.parent.parent / ".cursor"
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                LOG_PATH = str(LOG_DIR / "debug.log")
                if timestep < 5:
                    log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H4_timing", "location": "play.py:main_loop", "message": "Before get_camera_images", "data": {"timestep": timestep}}
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                # #endregion
                
                # Get camera images and robot state
                camera_images = get_camera_images(env, env_idx=0, timestep=timestep)
                robot_state = get_robot_state(env, env_idx=0)

                # Save debug images if requested
                if args_cli.save_debug_images and timestep % 10 == 0:
                    save_debug_images(camera_images, timestep)
                # Query remote server for new action chunk if needed
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args_cli.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # Prepare request data
                    request_data = prepare_request_data(
                        camera_images=camera_images,
                        robot_state=robot_state,
                        prompt=args_cli.prompt,
                        image_size=args_cli.image_size,
                        gripper_position=gripper_position,
                    )
                    # Query remote policy server
                    with prevent_keyboard_interrupt():
                        response = policy_client.infer(request_data)

                    pred_action_chunk = response["actions"]
                    #print(f"pred_action_chunk: {pred_action_chunk}")

                    # Ensure action chunk has correct shape
                    if pred_action_chunk.ndim == 1:
                        pred_action_chunk = pred_action_chunk[np.newaxis, :]

                # Get current action from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # 裁剪动作维度：OpenPI 输出 8 维（7关节 + 1夹爪），环境只需要 7 维
                # 丢弃最后一维（夹爪）
                if action.shape[-1] == 8:
                    action = action[..., :7]
                    gripper_position = action[-1]

                # Debug: 打印 action 和 joint_pos 的关系
                if timestep < 10:
                    joint_pos_before = robot_state.get("joint_position", None)
                    # 计算预期的目标位置（根据配置：default_pos + action * scale）
                    # Default positions from env.yaml: [0.0, 0.65, 0.0, 1.89, 0.0, 0.6, -1.57]
                    default_pos = np.array([0.0, 0.65, 0.0, 1.89, 0.0, 0.6, -1.57])
                    action_scaled = action * 0.5  # scale=0.5
                    target_joint_pos = default_pos + action_scaled
                    print(f"  Raw action: {action}")

                # Convert to torch tensor and expand for all environments
                action_tensor = torch.tensor(action, device=env.unwrapped.device, dtype=torch.float32)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)  # [1, action_dim]
                if action_tensor.shape[0] < args_cli.num_envs:
                    # Repeat action for all environments
                    action_tensor = action_tensor.repeat(args_cli.num_envs, 1)

                # Step environment
                obs, rewards, terminated, truncated, info = env.step(action_tensor)
                
                # Debug: 打印 step 后的 joint_pos 变化
                if timestep < 10:
                    robot_state_after = get_robot_state(env, env_idx=0)
                    joint_pos_after = robot_state_after.get("joint_position", None)
                    joint_pos_before = robot_state.get("joint_position", None)
                    if joint_pos_after is not None and joint_pos_before is not None:
                        print(f"  Joint pos after: {joint_pos_after}")
                
                # #region agent log
                # 假设5: 检查 step 后的状态和图像
                if timestep < 5:
                    robot_state_after = get_robot_state(env, env_idx=0)
                    joint_pos_after = robot_state_after.get("joint_position", None)
                    print(f"  Joint pos after: {joint_pos_after}")
                    joint_pos_before = robot_state.get("joint_position", None)
                    pos_changed = "N/A"
                    if joint_pos_after is not None and joint_pos_before is not None:
                        pos_changed = not np.allclose(joint_pos_after, joint_pos_before, atol=1e-6)
                    log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H5_step_effect", "location": "play.py:after_step", "message": "After env.step", "data": {"timestep": timestep, "joint_pos_changed": pos_changed, "action_applied": action.tolist()[:3] if action is not None else None}}
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                    
                    # 在 step 后再次获取相机图像，检查是否不同
                    camera_images_after = get_camera_images(env, env_idx=0, timestep=timestep)
                    if "external_image" in camera_images and "external_image" in camera_images_after:
                        img_same = np.array_equal(camera_images["external_image"], camera_images_after["external_image"])
                        log_entry = {"timestamp": time.time()*1000, "sessionId": "debug-session", "hypothesisId": "H6_before_after", "location": "play.py:compare_images", "message": "Compare before/after step", "data": {"timestep": timestep, "images_identical": img_same}}
                        with open(LOG_PATH, "a") as f:
                            f.write(json.dumps(log_entry) + "\n")
                # #endregion

            timestep += 1

            # Exit if recording video and reached target length
            if args_cli.video and timestep >= args_cli.video_length:
                print(f"[INFO] Video recording complete ({timestep} steps).")
                break

            # Time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

            # Print progress every 100 steps
            if timestep % 100 == 0:
                print(f"[INFO] Step {timestep}/{args_cli.max_timesteps}")

        except KeyboardInterrupt:
            print("\n[INFO] User interrupted. Stopping...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback

            traceback.print_exc()
            break

    print(f"\n[INFO] Evaluation complete. Total steps: {timestep}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
