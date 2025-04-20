# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_rel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the joint positions of the robot relative to zero."""
    robot: RigidObject = env.scene[robot_cfg.name]
    return robot.data.joint_pos  # (num_envs, num_joints)


def joint_vel_rel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the joint velocities of the robot."""
    robot: RigidObject = env.scene[robot_cfg.name]
    return robot.data.joint_vel  # (num_envs, num_joints)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def ee_position_world(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """End-effector position in world frame."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_position(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector pozíció a világkoordinátában."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get end-effector orientation as quaternion."""
    robot = env.scene[robot_cfg.name]
    # Megkeressük a megfogó linkjének az indexét
    body_names = robot.data.body_names
    ee_index = body_names.index("robotiq_85_base_link")
    # Kivesszük az orientációs kvaterniót
    ee_quat_w = robot.data.body_quat_w[:, ee_index]
    return ee_quat_w


def ee_to_object_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Az end-effector és az objektum közötti vektor."""
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    return object_pos - ee_pos  # vektor az EE-ből az objektum felé
