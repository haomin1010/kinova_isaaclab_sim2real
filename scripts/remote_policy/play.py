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


def get_camera_images(env, env_idx: int = 0, debug: bool = False) -> dict:
    """
    从环境中获取相机图像。

    Args:
        env: Isaac Lab 环境
        env_idx: 环境索引
        debug: 是否打印调试信息

    Returns:
        包含相机图像的字典
    """
    images = {}
    unwrapped_env = env.unwrapped

    # 尝试从场景中获取相机传感器
    if hasattr(unwrapped_env, "scene"):
        scene = unwrapped_env.scene

        if debug:
            # 打印场景中所有可用的属性
            print(f"[DEBUG] Scene type: {type(scene)}")
            print(f"[DEBUG] Scene attributes: {[attr for attr in dir(scene) if not attr.startswith('_')]}")
            # 打印 sensors 字典内容
            if hasattr(scene, "sensors"):
                print(f"[DEBUG] Scene sensors keys: {list(scene.sensors.keys())}")

        # 定义一个辅助函数来获取相机图像
        def _get_camera_rgb(camera, camera_name: str):
            if camera is None:
                return None
            if debug:
                print(f"[DEBUG] {camera_name} type: {type(camera)}")
                print(f"[DEBUG] {camera_name} has data: {hasattr(camera, 'data')}")

            if hasattr(camera, "data"):
                camera_data = camera.data
                if debug:
                    print(f"[DEBUG] {camera_name} data type: {type(camera_data)}")
                    data_attrs = [attr for attr in dir(camera_data) if not attr.startswith('_')]
                    print(f"[DEBUG] {camera_name} data attributes: {data_attrs}")

                # 尝试多种数据访问方式
                rgb_data = None
                if hasattr(camera_data, "output"):
                    rgb_data = camera_data.output.get("rgb")
                    if debug:
                        print(f"[DEBUG] {camera_name} output keys: {list(camera_data.output.keys()) if camera_data.output else 'None'}")
                elif hasattr(camera_data, "rgb"):
                    rgb_data = camera_data.rgb

                if rgb_data is not None:
                    # RGB 数据格式: [num_envs, H, W, C]
                    img = rgb_data[env_idx].cpu().numpy()
                    # 移除 alpha 通道（如果存在）
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    if debug:
                        print(f"[DEBUG] {camera_name} image shape: {img.shape}")
                    return img.astype(np.uint8)
            return None

        external_key = args_cli.external_camera_key
        wrist_key = args_cli.wrist_camera_key

        # 方法 1：直接作为场景属性访问
        if hasattr(scene, external_key):
            camera = getattr(scene, external_key)
            img = _get_camera_rgb(camera, "external_camera")
            if img is not None:
                images["external_image"] = img

        if hasattr(scene, wrist_key):
            camera = getattr(scene, wrist_key)
            img = _get_camera_rgb(camera, "wrist_camera")
            if img is not None:
                images["wrist_image"] = img

        # 方法 2：通过 sensors 字典访问
        if hasattr(scene, "sensors") and isinstance(scene.sensors, dict):
            if external_key in scene.sensors and "external_image" not in images:
                camera = scene.sensors[external_key]
                img = _get_camera_rgb(camera, "external_camera (from sensors)")
                if img is not None:
                    images["external_image"] = img

            if wrist_key in scene.sensors and "wrist_image" not in images:
                camera = scene.sensors[wrist_key]
                img = _get_camera_rgb(camera, "wrist_camera (from sensors)")
                if img is not None:
                    images["wrist_image"] = img

    return images


def get_robot_state(env, env_idx: int = 0, debug: bool = False) -> dict:
    """
    从环境中获取机器人状态。

    Args:
        env: Isaac Lab 环境
        env_idx: 环境索引
        debug: 是否打印调试信息

    Returns:
        包含机器人状态的字典
    """
    state = {}
    unwrapped_env = env.unwrapped

    if debug:
        print(f"[DEBUG] Looking for robot in scene...")

    # 方法 1：直接通过 scene.robot 访问
    robot = None
    if hasattr(unwrapped_env, "scene"):
        scene = unwrapped_env.scene
        
        if hasattr(scene, "robot"):
            robot = scene.robot
            if debug:
                print(f"[DEBUG] Found robot via scene.robot")
        
        # 方法 2：通过 articulations 字典访问
        elif hasattr(scene, "articulations") and isinstance(scene.articulations, dict):
            if debug:
                print(f"[DEBUG] Scene articulations keys: {list(scene.articulations.keys())}")
            # 尝试查找名为 "robot" 的 articulation
            if "robot" in scene.articulations:
                robot = scene.articulations["robot"]
                if debug:
                    print(f"[DEBUG] Found robot via scene.articulations['robot']")
            # 或者取第一个 articulation
            elif len(scene.articulations) > 0:
                robot_key = list(scene.articulations.keys())[0]
                robot = scene.articulations[robot_key]
                if debug:
                    print(f"[DEBUG] Found robot via scene.articulations['{robot_key}']")

    if robot is not None:
        if debug:
            print(f"[DEBUG] Robot type: {type(robot)}")
            print(f"[DEBUG] Robot has data: {hasattr(robot, 'data')}")
            if hasattr(robot, 'data'):
                data_attrs = [attr for attr in dir(robot.data) if not attr.startswith('_')]
                print(f"[DEBUG] Robot data attributes: {data_attrs}")

        # 获取关节位置
        if hasattr(robot, 'data') and hasattr(robot.data, "joint_pos"):
            joint_pos = robot.data.joint_pos[env_idx].cpu().numpy()
            state["joint_position"] = joint_pos
            if debug:
                print(f"[DEBUG] Joint position shape: {joint_pos.shape}")

        # 获取关节速度
        if hasattr(robot, 'data') and hasattr(robot.data, "joint_vel"):
            joint_vel = robot.data.joint_vel[env_idx].cpu().numpy()
            state["joint_velocity"] = joint_vel

        # 获取末端执行器位姿（如果可用）
        if hasattr(robot, 'data') and hasattr(robot.data, "body_pos_w"):
            # 尝试获取末端执行器的位置
            # 注意：这里的索引可能需要根据具体机器人调整
            ee_pos = robot.data.body_pos_w[env_idx, -1].cpu().numpy()
            state["ee_position"] = ee_pos
    else:
        if debug:
            print(f"[DEBUG] Robot not found in scene!")

    return state


def prepare_request_data(
        camera_images: dict,
        robot_state: dict,
        prompt: str = "",
        image_size: int = 224,
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
    request_data["observation/gripper_position"] = np.array([0.0])

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
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Action space dimension: {env.action_space.shape}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    
    # 打印机器人关节信息
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "scene"):
        scene = unwrapped.scene
        if hasattr(scene, "articulations") and "robot" in scene.articulations:
            robot = scene.articulations["robot"]
            print(f"[INFO] Robot joint names: {robot.joint_names}")
            print(f"[INFO] Robot num joints: {len(robot.joint_names)}")
        # 检查 isaaclab_assets 中可用的 Kinova 配置
        try:
            import isaaclab_assets
            kinova_configs = [name for name in dir(isaaclab_assets) if 'KINOVA' in name.upper() or 'GEN3' in name.upper()]
            print(f"[INFO] Available Kinova configs in isaaclab_assets: {kinova_configs}")
        except Exception as e:
            print(f"[WARNING] Could not list isaaclab_assets: {e}")
    print(f"[INFO] Open-loop horizon: {args_cli.open_loop_horizon}")
    print(f"[INFO] Max timesteps: {args_cli.max_timesteps}")
    print(f"[INFO] Image size: {args_cli.image_size}x{args_cli.image_size}")
    if args_cli.prompt:
        print(f"[INFO] Prompt: {args_cli.prompt}")

    # Check for cameras
    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, "scene"):
        scene = unwrapped_env.scene

        # 打印场景配置类型
        print(f"[INFO] Scene config type: {type(scene.cfg).__name__ if hasattr(scene, 'cfg') else 'Unknown'}")

        # 检查 scene.sensors 字典
        print("[INFO] Checking scene.sensors dict...")
        if hasattr(scene, "sensors") and isinstance(scene.sensors, dict):
            print(f"[INFO] scene.sensors keys: {list(scene.sensors.keys())}")
            for key, sensor in scene.sensors.items():
                print(f"       - {key}: {type(sensor).__name__}")
        else:
            print("[WARNING] scene.sensors not found or not a dict!")
            print(f"[DEBUG] scene.sensors type: {type(getattr(scene, 'sensors', None))}")

        # 检查直接属性
        has_external = hasattr(scene, args_cli.external_camera_key)
        has_wrist = hasattr(scene, args_cli.wrist_camera_key)
        
        # 也检查 sensors 字典
        sensors_dict = getattr(scene, "sensors", {}) or {}
        has_external_in_sensors = args_cli.external_camera_key in sensors_dict
        has_wrist_in_sensors = args_cli.wrist_camera_key in sensors_dict
        
        print(f"[INFO] External camera ({args_cli.external_camera_key}):")
        print(f"       - as attribute: {'Found' if has_external else 'Not found'}")
        print(f"       - in sensors dict: {'Found' if has_external_in_sensors else 'Not found'}")
        print(f"[INFO] Wrist camera ({args_cli.wrist_camera_key}):")
        print(f"       - as attribute: {'Found' if has_wrist else 'Not found'}")
        print(f"       - in sensors dict: {'Found' if has_wrist_in_sensors else 'Not found'}")

        if not (has_external or has_external_in_sensors) and not (has_wrist or has_wrist_in_sensors):
            print("[WARNING] No cameras found! Make sure to use a task with camera sensors.")
            print("[WARNING] Example: --task Gen3-Reach-Camera-v0")
            print("[DEBUG] Available scene attributes (non-private):")
            for attr_name in sorted(dir(scene)):
                if not attr_name.startswith('_'):
                    attr = getattr(scene, attr_name, None)
                    print(f"         {attr_name}: {type(attr).__name__ if attr is not None else 'None'}")

    dt = env.unwrapped.physics_dt

    # Action chunk management
    pred_action_chunk = None
    actions_from_chunk_completed = 0

    # reset environment
    obs, info = env.reset()

    timestep = 0

    print("\n[INFO] Starting evaluation loop...")
    print("[INFO] Press Ctrl+C to stop.\n")

    # simulate environment
    while simulation_app.is_running() and timestep < args_cli.max_timesteps:
        start_time = time.time()

        try:
            # run everything in inference mode
            with torch.inference_mode():
                # Get camera images and robot state (debug on first step)
                debug_mode = (timestep == 0)
                camera_images = get_camera_images(env, env_idx=0, debug=debug_mode)
                robot_state = get_robot_state(env, env_idx=0, debug=debug_mode)
 
                if debug_mode:
                    print(f"[DEBUG] Camera images keys: {list(camera_images.keys())}")
                    print(f"[DEBUG] Robot state keys: {list(robot_state.keys())}")

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
                    )
                    # Query remote policy server
                    with prevent_keyboard_interrupt():
                        response = policy_client.infer(request_data)

                    pred_action_chunk = response["actions"]

                    # Ensure action chunk has correct shape
                    if pred_action_chunk.ndim == 1:
                        pred_action_chunk = pred_action_chunk[np.newaxis, :]

                # Get current action from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Convert to torch tensor and expand for all environments
                action_tensor = torch.tensor(action, device=env.unwrapped.device, dtype=torch.float32)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)  # [1, action_dim]
                if action_tensor.shape[0] < args_cli.num_envs:
                    # Repeat action for all environments
                    action_tensor = action_tensor.repeat(args_cli.num_envs, 1)

                # Step environment
                # print(action_tensor)
                obs, rewards, terminated, truncated, info = env.step(action_tensor)

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
