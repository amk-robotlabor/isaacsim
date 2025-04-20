# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward if the object is lifted slightly above the table (indicating grasp success)."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > 0.08, 1.0, 0.0)


def ee_above_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the end-effector being right above the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    horizontal_dist = torch.norm(ee_pos[:, :2] - object_pos[:, :2], dim=-1)
    height_diff = ee_pos[:, 2] - object_pos[:, 2]

    # Jutalmazzuk, ha kicsi a vízszintes távolság ÉS a gripper magasan van
    return torch.where((horizontal_dist < 0.02) & (height_diff > 0.05), 1.0, 0.0)


def gripper_above_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the gripper being just slightly above the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    vertical_distance = ee_pos[:, 2] - object_pos[:, 2]

    # reward if the gripper is 0.03 - 0.05 meters above the cube
    return torch.where((vertical_distance > 0.03) & (vertical_distance < 0.07), 1.0, 0.0)


def gripper_closed(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02
) -> torch.Tensor:
    """Reward if the gripper is closed enough (after reaching the object)."""
    # Joint positions of the gripper fingers
    joint_indices = {name: i for i, name in enumerate(env.scene["robot"].data.joint_names)}
    left_index = joint_indices["robotiq_85_left_knuckle_joint"]
    left = env.scene["robot"].data.joint_pos[:, left_index]
    # If both joints are closed (close to commanded closed value)
    return (left > 0.9).float()


def ee_orientation_alignment(
    env: ManagerBasedRLEnv,
    desired_quat: torch.Tensor = torch.tensor([0, 0, 0, 1], device="cuda"),  # x lefelé
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward if the end-effector orientation aligns with the desired orientation."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_rot = ee_frame.data.target_quat_w[..., 0, :]
    cos_sim = torch.abs(torch.sum(ee_rot * desired_quat, dim=-1))
    return cos_sim


def hold_object(env: ManagerBasedRLEnv, minimal_height: float = 0.1) -> torch.Tensor:
    object = env.scene["object"]
    return (object.data.root_pos_w[:, 2] > minimal_height).float()


def gripper_closed_near_object(env, env_ids: Optional[Sequence[int]] = None, threshold=0.04):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 1. End-effector pozíció a FrameTransformer-ből
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]  # shape: [num_envs, 3]

    # 2. Objektum pozíció
    obj_pos = env.scene["object"].data.root_pos_w[env_ids]

    # 3. Távolság számítása
    dist = torch.norm(ee_pos[env_ids] - obj_pos, dim=1)

    # 4. Gripper állapot (joint pozíciók)
    joint_pos = env.scene["robot"].data.joint_pos[env_ids]
    joint_names = env.scene["robot"].data.joint_names
    left_idx = joint_names.index("robotiq_85_left_knuckle_joint")
    right_idx = joint_names.index("robotiq_85_right_knuckle_joint")

    left_val = joint_pos[:, left_idx]
    right_val = joint_pos[:, right_idx]

    # 5. Zárás és közelség ellenőrzés
    is_closed = (left_val > 0.6) & (right_val > 0.6)
    is_near = dist < threshold

    reward = (is_closed & is_near).float()
    return reward