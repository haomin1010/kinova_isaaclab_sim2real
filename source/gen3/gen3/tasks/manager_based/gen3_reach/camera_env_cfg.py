# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
带相机传感器的 Gen3 Reach 环境配置。

该配置在原有 Gen3ReachEnvCfg 的基础上添加了两个相机：
- 外部相机（external_camera）：固定在场景中，提供全局视角
- 腕部相机（wrist_camera）：安装在机器人末端执行器上，随机械臂移动
"""

import math

from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip


##
# Scene configuration with cameras
##


@configclass
class Gen3ReachCameraSceneCfg(InteractiveSceneCfg):
    """场景配置，包含机器人和两个相机传感器。"""

    # 继承基础场景配置需要的属性会在 __post_init__ 中设置

    # 外部相机 - 固定在场景中，提供第三人称视角
    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        update_period=0.1,  # 10 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 1.5, 1.2),  # 相机位置：斜上方观察
            rot=(0.653, 0.271, 0.271, 0.653),  # 朝向机器人工作区
            convention="world",
        ),
    )

    # 腕部相机 - 安装在机器人末端执行器上
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/end_effector_link/WristCamera",
        update_period=0.1,  # 10 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),  # 相对于末端执行器的偏移
            rot=(0.5, -0.5, 0.5, -0.5),  # 朝下看
            convention="ros",
        ),
    )


##
# Environment configuration with cameras
##


@configclass
class Gen3ReachCameraEnvCfg(ReachEnvCfg):
    """带相机的 Gen3 Reach 环境配置。

    该配置用于需要视觉输入的策略，例如 VLA（Vision-Language-Action）模型。
    提供两个相机视角：
    - external_camera: 外部固定相机，提供场景全局视角
    - wrist_camera: 腕部相机，提供末端执行器视角
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 切换为 Kinova Gen3 机器人
        self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 添加相机传感器
        # 外部相机 - 固定在场景中
        self.scene.external_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ExternalCamera",
            update_period=0.0,  # 每个物理步都更新
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.5, 1.5, 1.2),  # 斜上方位置
                rot=(0.653, 0.271, 0.271, 0.653),  # 朝向工作区
                convention="world",
            ),
        )

        # 腕部相机 - 安装在末端执行器上
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector_link/WristCamera",
            update_period=0.0,  # 每个物理步都更新
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.02),  # 相对于末端执行器
                rot=(0.5, -0.5, 0.5, -0.5),  # 朝下看
                convention="ros",
            ),
        )

        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["end_effector_link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # override command generator body
        self.commands.ee_pose.body_name = "end_effector_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
