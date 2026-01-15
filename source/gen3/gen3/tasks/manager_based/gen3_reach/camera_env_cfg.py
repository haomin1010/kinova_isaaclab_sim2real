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

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# 导入 ReachEnvCfg 使用的场景配置
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachSceneCfg

##
# Pre-defined configs
##
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip


##
# Scene configuration with cameras
##


@configclass
class Gen3ReachCameraSceneCfg(ReachSceneCfg):
    """
    带相机的场景配置。

    继承自 ReachSceneCfg，添加两个相机传感器。
    """

    # 外部相机 - 固定在场景中，提供第三人称视角
    # 位置在机器人前方斜上方，朝向机器人基座
    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        update_period=0.0,  # 每个物理步都更新
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # 相机位置：在机器人左前方，拉高拉远以获得更广视野
            pos=(0.8, 1.2, 1.0),
            # 四元数 (w, x, y, z)：让相机朝向 (0.3, 0, 0.1) - 小球附近
            # 方向：(-0.5, -1.2, -0.9)，向下倾斜约 35°
            # 绕 Z 轴旋转约 -113°，再向下倾斜
            rot=(0.529, 0.165, -0.248, -0.794),
            convention="world",
        ),
    )

    # 腕部相机 - 安装在机器人末端执行器上
    # 朝向末端执行器前方
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/end_effector_link/WristCamera",
        update_period=0.0,  # 每个物理步都更新
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # 相对于末端执行器的偏移：稍微向前
            pos=(0.1, 0.0, 0.0),
            # 朝向末端执行器前方（沿 x 轴）
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # 桌子 - 放置物体的平台
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.05),  # 60cm x 80cm x 5cm 厚的桌面
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 桌子固定不动
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # 木头颜色
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025),  # 在机器人前方 0.5m，桌面高度 5cm
        ),
    )

    # 可抓取的小球 - 红色球体，放在桌子上
    target_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBall",
        spawn=sim_utils.SphereCfg(
            radius=0.04,  # 4cm 半径，适合抓取
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 受物理影响
                disable_gravity=False,    # 受重力影响
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.1,  # 100g 质量
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # 启用碰撞
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # 红色
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.1),  # 在桌子上方（桌面高度 0.05 + 球半径 0.04 + 一点余量）
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

    # 使用带相机的场景配置
    scene: Gen3ReachCameraSceneCfg = Gen3ReachCameraSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 切换为 Kinova Gen3 机器人
        self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
