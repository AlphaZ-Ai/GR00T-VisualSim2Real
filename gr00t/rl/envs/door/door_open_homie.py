# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import omni.usd
import torch
import torch.nn.functional as F
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    wrap_to_pi,
)
from pxr import Usd
from typing_extensions import override

from gr00t.rl.envs.base_task.delta_action_base import DeltaActionBase
from gr00t.rl.envs.base_task.finger_primitive_base import FingerPrimitiveBase
from gr00t.rl.envs.base_task.homie_base import HomieBase
from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase
from gr00t.rl.envs.base_task.warped_action_base import WarpedActionBase
from gr00t.rl.envs.door.reset_from_dataset import ResetFromDataset
from gr00t.rl.isaac_utils.rotations import quat_to_tan_norm, wxyz_to_xyzw, xyzw_to_wxyz
from gr00t.rl.utils.torch_utils import torch_rand_float


class DoorPregrasp(
    StagedTaskBase,
    DeltaActionBase,
    WarpedActionBase,
    HomieBase,
    FingerPrimitiveBase,
    ResetFromDataset,
):
    STAGE_WALK_TO_DOOR = 0
    STAGE_PREGRASP = 1
    STAGE_GRASP = 2
    STAGE_OPEN = 3
    STAGE_SWING = 4
    STAGE_THROUGH = 5

    def __init__(self, config, device):
        super().__init__(config, device)

        # finger primitive related
        self._left_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_0,
            device=self.device,
            requires_grad=False,
        )
        self._left_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_1,
            device=self.device,
            requires_grad=False,
        )
        self._right_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_0,
            device=self.device,
            requires_grad=False,
        )
        self._right_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_1,
            device=self.device,
            requires_grad=False,
        )
        self._left_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.left.dof_names
        ]
        self._right_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.right.dof_names
        ]
        self._upper_non_finger_dof_idx = [
            i
            for i in self.upper_dof_indices
            if i not in self._left_hand_dof_idx and i not in self._right_hand_dof_idx
        ]

        # read the door metadata
        stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.door_width = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_handle_height = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_handle_width = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_weight = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_lr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_io = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for env_id in range(self.num_envs):
            door_prim_path = f"/World/envs/env_{env_id}/door"
            door_prim = stage.GetPrimAtPath(door_prim_path)
            door_metadata = door_prim.GetPrim().GetMetadata("customData")
            self.door_width[env_id] = door_metadata["doorWidth"]
            self.door_height[env_id] = door_metadata["doorHeight"]
            self.door_handle_height[env_id] = door_metadata["doorHandleHeight"]
            self.door_handle_width[env_id] = door_metadata["doorHandleWidth"]
            self.door_weight[env_id] = door_metadata["doorWeight"]
            self.door_open_lr[env_id] = door_metadata["doorOpenLR"]

        # body indices
        self.left_palm_idx = self.simulator.body_names.index(self.simulator.robot_config.left_hand_palm_link)
        self.right_palm_idx = self.simulator.body_names.index(self.simulator.robot_config.right_hand_palm_link)
        self.root_idx = self.simulator.body_names.index("pelvis")
        self.left_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.left_hand_body_names
        ]
        self.right_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.right_hand_body_names
        ]
        g1_hand_links = (
            list(self.simulator.robot_config.left_hand_body_names)
            + list(self.simulator.robot_config.right_hand_body_names)
        )
        self.left_hand_indices_tgt_ct_sensor = [
            g1_hand_links.index(link)
            for link in self.simulator.robot_config.left_hand_body_names
        ]
        self.left_hand_indices_convert = [
            self.left_hand_indices.index(self.simulator.body_names.index(g1_hand_links[i]))
            for i in self.left_hand_indices_tgt_ct_sensor
        ]
        self.right_hand_indices_tgt_ct_sensor = [
            g1_hand_links.index(link)
            for link in self.simulator.robot_config.right_hand_body_names
        ]
        self.right_hand_indices_convert = [
            self.right_hand_indices.index(self.simulator.body_names.index(g1_hand_links[i]))
            for i in self.right_hand_indices_tgt_ct_sensor
        ]

        self.left_hand_palm_side_direction = self._parse_palm_side_direction(
            self.simulator.robot_config.left_hand_palm_side_direction
        )
        self.right_hand_palm_side_direction = self._parse_palm_side_direction(
            self.simulator.robot_config.right_hand_palm_side_direction
        )

        # dof indices
        finger_dof_names = (
            list(self.simulator.robot_config.left_hand_dof_names)
            + list(self.simulator.robot_config.right_hand_dof_names)
        )
        self.finger_dof_idx = torch.tensor(
            [self.simulator.dof_names.index(dof) for dof in finger_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.non_finger_dof_idx = [
            self.simulator.dof_names.index(dof)
            for dof in self.simulator.dof_names
            if dof not in finger_dof_names
        ]
        self.wrist_dof_idx = torch.tensor(
            [
                self.simulator.dof_names.index(dof)
                for dof in self.simulator.dof_names
                if "wrist" in dof
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.dof_pos_humanly_lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_lower_limit_list, device=self.device
        )[None, :]
        self.dof_pos_humanly_upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_upper_limit_list, device=self.device
        )[None, :]

        self._left_arm_dof_idx = torch.tensor(self.left_arm_dof_indices, device=self.device)
        self._right_arm_dof_idx = torch.tensor(self.right_arm_dof_indices, device=self.device)

        # Palm-down (top-down) grasp orientation, direction-based. Calibrated IN-SIM: posed the full
        # measured left-arm grasp config (shoulder/elbow/wrist) with the pelvis pinned upright and
        # read the actual wrist_yaw_link world rotation -- the Dex1 palm faces along its local +x
        # axis (local +x -> world-down ~ -0.89 at the grasp pose, both hands). The reward rewards that
        # axis, transformed to world, pointing at world-down: direction-based, tolerates redundancy.
        self._palm_facing_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        self._register_task_state_to_track(self.simulator.scene.articulations["door"], "door")
        self._register_buffer_to_track(
            "delta_actions",
            self._get_delta_actions_buffer_shape(),
            self._store_delta_actions_buffer,
            self._load_delta_actions_buffer,
            dtype=torch.float32,
        )

        self.resting_dof_pos = torch.tensor([self.config.resting_dof_pos], device=self.device)

        self.target_root_pos = torch.tensor(self.config.target_root_pos, device=self.device)[
            None, :
        ]

    def _init_buffers(self):
        super()._init_buffers()
        self.relative_door_pos_buf = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.relative_door_rot_buf = torch.zeros(
            self.num_envs, 4, device=self.device, requires_grad=False
        )

        # door state buffer
        self.door_root_state_buf = torch.zeros(
            self.num_envs, 13, device=self.device, requires_grad=False
        )
        self.door_root_state_buf[:, 3] = 1.0  # w
        self.door_dof_state_buf = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.door_root_state_buf[:, :3] += self.env_origins

    def _pre_compute_observations_callback(self, env_ids=None):
        super()._pre_compute_observations_callback(env_ids)
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids

        current_root_pos = self.simulator.robot_root_states[env_ids, :3].clone()
        current_root_rot = self.simulator.robot_root_states[env_ids, 3:7].clone()
        current_root_rot_wxyz = xyzw_to_wxyz(current_root_rot)

        door_root_pos = self.simulator.get_task_root_state("door")[env_ids, :3].clone()
        door_root_pos[:, 2] = current_root_pos[:, 2]
        door_root_rot_wxyz = self.simulator.get_task_root_state("door")[env_ids, 3:7].clone()

        relative_door_pos, relative_door_rot = subtract_frame_transforms(
            current_root_pos, current_root_rot_wxyz, door_root_pos, door_root_rot_wxyz
        )
        self.relative_door_pos_buf[env_ids] = relative_door_pos
        self.relative_door_rot_buf[env_ids] = wxyz_to_xyzw(relative_door_rot)

    @StagedTaskBase.effective_in_stage(STAGE_WALK_TO_DOOR)
    def _reward_walk_to_door(self):
        current_root_pos = self.simulator.robot_root_states[:, :3].clone()
        door_root_pos = self.simulator.get_task_root_state("door")[:, :3].clone()
        door_root_pos[:, 2] = current_root_pos[:, 2]
        door_direction = door_root_pos - current_root_pos
        target_dir = F.normalize(door_direction, dim=-1)
        current_root_vel = self.simulator.robot_root_states[:, 7:10].clone()

        target_vel = self.config.get("target_root_vel", 0.3) * target_dir

        return self._tracking_reward_util(
            torch.linalg.norm(current_root_vel - target_vel, dim=-1),
            std=0.15,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([STAGE_WALK_TO_DOOR, STAGE_THROUGH])
    def _reward_penalty_upper_body_non_finger_deviation_l1(self):
        """Maintain upper body pose (resting "ready" pose) during walk-to-door and through"""
        return torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
        ).sum(dim=-1)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP])
    def _reward_pregrasp_finger_dof_pos_l1(self):
        # Half-closed "ready" gripper during pregrasp: target the midpoint of the open (p0) and
        # closed (p1) finger primitive positions for the active hand (selected by door_open_lr<0).
        left_half = 0.5 * (self._left_p0 + self._left_p1)
        right_half = 0.5 * (self._right_p0 + self._right_p1)
        left_err = (
            self.simulator.dof_pos[:, self._left_hand_dof_idx] - left_half
        ).abs().sum(dim=-1)
        right_err = (
            self.simulator.dof_pos[:, self._right_hand_dof_idx] - right_half
        ).abs().sum(dim=-1)
        err = torch.where(self.door_open_lr < 0, left_err, right_err)
        return self._tracking_reward_util(err, std=0.02, target=0.0, scale=1.0, offset=0.0)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_penalty_unused_dof_deviation_l1(self):
        """Penalize the deviation of the unused arm dof during door opening"""
        left_diff = (
            self.simulator.dof_pos[:, self._left_arm_dof_idx]
            - self.resting_dof_pos[:, self._left_arm_dof_idx]
        )
        right_diff = (
            self.simulator.dof_pos[:, self._right_arm_dof_idx]
            - self.resting_dof_pos[:, self._right_arm_dof_idx]
        )
        return torch.where(self.door_open_lr[:, None] < 0, right_diff, left_diff).abs().sum(dim=-1)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP])
    def _reward_hand_handle_orientation(self):
        left_q = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.left_palm_idx, :])
        right_q = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.right_palm_idx, :])
        palm_q = torch.where((self.door_open_lr < 0)[:, None], left_q, right_q)
        palm_facing_world = quat_apply(palm_q, self._palm_facing_axis)
        cos = (palm_facing_world * self._world_down).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cos)
        return self._tracking_reward_util(angle, std=0.6, target=0.0, scale=1.0, offset=0.0)

    @override
    def _post_physics_step(self):
        super()._post_physics_step()
        if os.environ.get("CALIB_PALM"): 
            self._calibrate_palm_in_sim()

    def _calibrate_palm_in_sim(self):
        import sys

        c = getattr(self, "_calib_count", 0)
        self._calib_count = c + 1
        env_ids = torch.arange(self.num_envs, device=self.device)
        if c == 0:
            names = list(self.simulator.dof_names)
            dof = self.simulator.dof_pos[env_ids].clone()
            arm_targets = {
                "left_shoulder_pitch_joint": -0.09, "left_shoulder_roll_joint": 0.41,
                "left_shoulder_yaw_joint": 0.38, "left_elbow_joint": -0.05,
                "left_wrist_roll_joint": -1.57, "left_wrist_pitch_joint": 0.39,
                "left_wrist_yaw_joint": 1.57,
                "right_shoulder_pitch_joint": -0.09, "right_shoulder_roll_joint": -0.41,
                "right_shoulder_yaw_joint": -0.38, "right_elbow_joint": -0.05,
                "right_wrist_roll_joint": 1.57, "right_wrist_pitch_joint": 0.39,
                "right_wrist_yaw_joint": -1.57,
            }
            for jname, val in arm_targets.items():
                dof[:, names.index(jname)] = val
            self._calib_dof = dof
            self._calib_root = self.simulator.robot_root_states[env_ids].clone()
            self._calib_root[:, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            )  # force pelvis perfectly upright (wxyz identity) so "down" read is clean
            self._calib_root[:, 7:13] = 0.0  # zero base velocity so it stays put
        self.simulator.write_joint_state_to_sim(
            self._calib_dof, torch.zeros_like(self._calib_dof), env_ids
        )
        self.simulator.write_root_state_to_sim(self._calib_root, env_ids)
        if c < 4:
            return  # let the writes propagate + sensors update

        def show(tag, q):
            for axname, ax in [("+x", [1.0, 0, 0]), ("+y", [0, 1.0, 0]), ("+z", [0, 0, 1.0])]:
                v = quat_apply(q, torch.tensor([ax], device=self.device))[0]
                print(f"[CALIB] {tag} local {axname} -> world {[round(x,3) for x in v.tolist()]} "
                      f"(z={v[2].item():+.3f})")

        print("\n================ PALM CALIBRATION (in-sim) ================")
        pq = xyzw_to_wxyz(self.simulator._rigid_body_rot[0:1, self.root_idx, :])
        print(f"[CALIB] pelvis world quat wxyz = {[round(x,3) for x in pq[0].tolist()]} "
              f"(upright ~ [1,0,0,0]; if not, base tilted -> reread)")
        for side, idx in [("LEFT palm", self.left_palm_idx), ("RIGHT palm", self.right_palm_idx)]:
            q = xyzw_to_wxyz(self.simulator._rigid_body_rot[0:1, idx, :])
            show(side, q)
            print(f"[CALIB] {side} world quat wxyz = {[round(x,3) for x in q[0].tolist()]}")
        lh = self.simulator.left_hand_transform_rot[0, 0, :]
        rh = self.simulator.right_hand_transform_rot[0, 0, :]
        print(f"[CALIB] left_hand_transform_rot (handle rel palm) wxyz = {[round(x,3) for x in lh.tolist()]}")
        print(f"[CALIB] right_hand_transform_rot wxyz = {[round(x,3) for x in rh.tolist()]}")
        print("===========================================================\n")
        sys.stdout.flush()
        sys.exit(0)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN])
    def _reward_standing_still(self):
        norm = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1)
        return self._tracking_reward_util(norm, std=0.05, target=0.0, scale=1.0, offset=0.0)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN])
    def _reward_penalty_not_standing_still(self):
        norm = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1)
        return norm

    @StagedTaskBase.effective_in_stage(STAGE_SWING)
    def _reward_penalty_standing_still(self):
        norm = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1)
        return self._tracking_reward_util(norm, std=0.05, target=0.0, scale=1.0, offset=0.0)

    @StagedTaskBase.effective_in_stage(STAGE_PREGRASP)
    def _reward_pregrasp_target_distance(self):
        pre_grasp_target = self._compute_pre_grasp_target()

        left_hand_pos = self.simulator._rigid_body_pos[:, self.left_palm_idx, :]
        right_hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_idx, :]

        left_hand_pos_to_pre_grasp_target = pre_grasp_target - left_hand_pos
        right_hand_pos_to_pre_grasp_target = pre_grasp_target - right_hand_pos

        left_hand_pos_to_pre_grasp_target_norm = torch.norm(
            left_hand_pos_to_pre_grasp_target, dim=-1
        )
        right_hand_pos_to_pre_grasp_target_norm = torch.norm(
            right_hand_pos_to_pre_grasp_target, dim=-1
        )

        pos_reward = self._tracking_reward_util(
            torch.where(
                self.door_open_lr < 0,
                left_hand_pos_to_pre_grasp_target_norm,
                right_hand_pos_to_pre_grasp_target_norm,
            ),
            std=0.2,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

        left_current_direction = F.normalize(pre_grasp_target - left_hand_pos, dim=-1)
        right_current_direction = F.normalize(pre_grasp_target - right_hand_pos, dim=-1)

        left_palm_vel = self.simulator._rigid_body_vel[:, self.left_palm_idx, :]
        right_palm_vel = self.simulator._rigid_body_vel[:, self.right_palm_idx, :]

        pregrasp_target_vel = self.config.get("pregrasp_target_vel", 0.5)
        left_target_vel = pregrasp_target_vel * left_current_direction
        right_target_vel = pregrasp_target_vel * right_current_direction

        vel_reward = self._tracking_reward_util(
            torch.where(
                self.door_open_lr < 0,
                torch.linalg.norm(left_palm_vel - left_target_vel, dim=-1),
                torch.linalg.norm(right_palm_vel - right_target_vel, dim=-1),
            ),
            std=0.15,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )
        return (pos_reward + vel_reward).clamp(max=1.0)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp_finger_dof_pos_l1(self):
        return torch.zeros(self.num_envs, device=self.device)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp_target_distance(self):
        grasp_target = self._compute_grasp_target()

        left_hand_pos = self.simulator._rigid_body_pos[:, self.left_palm_idx, :]
        right_hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_idx, :]

        left_hand_pos_to_grasp_target = grasp_target - left_hand_pos
        right_hand_pos_to_grasp_target = grasp_target - right_hand_pos

        left_hand_pos_to_grasp_target_norm = torch.norm(left_hand_pos_to_grasp_target, dim=-1)
        right_hand_pos_to_grasp_target_norm = torch.norm(right_hand_pos_to_grasp_target, dim=-1)

        return self._tracking_reward_util(
            torch.where(
                self.door_open_lr < 0,
                left_hand_pos_to_grasp_target_norm,
                right_hand_pos_to_grasp_target_norm,
            ),
            std=0.1,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp(self):
        left_contact_forces = self.simulator.object_to_hand_contact_forces[
            :, 0, self.left_hand_indices_tgt_ct_sensor, :
        ][:, self.left_hand_indices_convert, :]
        left_contact_forces_flattened = left_contact_forces.reshape(-1, 3)
        left_hand_rot = self.simulator._rigid_body_rot[:, self.left_hand_indices, :][
            :, :, [3, 0, 1, 2]
        ]  # flip xyzw to wxyz
        left_hand_rot_flattened = left_hand_rot.reshape(-1, 4)
        left_palm_side_repeat = torch.tile(
            self.left_hand_palm_side_direction, (left_contact_forces.shape[0], 1)
        )
        # rotate contact forces first to hand body frames, and then to palm-facing frames
        left_contact_forces_hand_frame = quat_apply(
            quat_inv(left_hand_rot_flattened), left_contact_forces_flattened
        )
        left_contact_forces_palm_frame = quat_apply(
            quat_inv(left_palm_side_repeat), left_contact_forces_hand_frame
        )

        right_contact_forces = self.simulator.object_to_hand_contact_forces[
            :, 0, self.right_hand_indices_tgt_ct_sensor, :
        ][:, self.right_hand_indices_convert, :]
        right_contact_forces_flattened = right_contact_forces.reshape(-1, 3)
        right_hand_rot = self.simulator._rigid_body_rot[:, self.right_hand_indices, :][
            :, :, [3, 0, 1, 2]
        ]  # flip xyzw to wxyz
        right_hand_rot_flattened = right_hand_rot.reshape(-1, 4)
        right_palm_side_repeat = torch.tile(
            self.right_hand_palm_side_direction, (right_contact_forces.shape[0], 1)
        )
        # rotate contact forces first to hand body frames, and then to palm-facing frames
        right_contact_forces_hand_frame = quat_apply(
            quat_inv(right_hand_rot_flattened), right_contact_forces_flattened
        )
        right_contact_forces_palm_frame = quat_apply(
            quat_inv(right_palm_side_repeat), right_contact_forces_hand_frame
        )

        # reward forces acting out of the palm (x) direction. penalize forces on other directions.
        left_reward = (
            (
                -1.0 * torch.abs(left_contact_forces_palm_frame[:, 1:]).sum(dim=-1)
                + left_contact_forces_palm_frame[:, 0]
            )
            .clamp(min=-10, max=10)
            .reshape(self.num_envs, -1)
            .mean(dim=-1)
        )
        right_reward = (
            (
                -1.0 * torch.abs(right_contact_forces_palm_frame[:, 1:]).sum(dim=-1)
                + right_contact_forces_palm_frame[:, 0]
            )
            .clamp(min=-10, max=10)
            .reshape(self.num_envs, -1)
            .mean(dim=-1)
        )
        reward = left_reward + right_reward

        reward[self.stage_buf == DoorPregrasp.STAGE_PREGRASP] = -1.0 * torch.abs(
            reward[self.stage_buf == DoorPregrasp.STAGE_PREGRASP]
        )

        return reward

    @StagedTaskBase.effective_in_stage(STAGE_OPEN)
    def _reward_push_door_force(self):
        left_net_force = self.simulator.object_to_hand_contact_forces[
            :, 0, self.left_hand_indices_tgt_ct_sensor, :
        ].sum(dim=-2)
        right_net_force = self.simulator.object_to_hand_contact_forces[
            :, 0, self.right_hand_indices_tgt_ct_sensor, :
        ].sum(dim=-2)
        # reward -x direction force (pushing the door)
        return (
            torch.where(self.door_open_lr < 0, left_net_force[:, 0], right_net_force[:, 0])
        ).clamp(min=0.0, max=20.0)

    @StagedTaskBase.effective_in_stage(STAGE_OPEN)
    def _reward_push_door_handle(self):
        handle_vel_reward = self.simulator.scene.articulations["door"].data.joint_vel[:, 1]
        handle_pos_reward = (
            self.simulator.scene.articulations["door"]
            .data.joint_pos[:, 1]
            .clamp(min=0.0, max=0.785398)
            / 0.785398
        )
        return (handle_vel_reward + handle_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_SWING, STAGE_THROUGH])
    def _reward_dont_push_door_handle(self):
        handle_vel_reward = -1.0 * self.simulator.scene.articulations["door"].data.joint_vel[:, 1]
        handle_pos_reward = (
            0.785398 - self.simulator.scene.articulations["door"].data.joint_pos[:, 1]
        ).clamp(min=0.0, max=0.785398) / 0.785398
        return (handle_vel_reward + handle_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_WALK_TO_DOOR, STAGE_PREGRASP])
    def _reward_penalty_disturb_lever(self):
        return self.simulator.scene.articulations["door"].data.joint_pos[:, 1].abs()

    @StagedTaskBase.effective_in_stage([STAGE_OPEN, STAGE_SWING])
    def _reward_push_door_hinge(self):
        hinge_vel_reward = self.simulator.scene.articulations["door"].data.joint_vel[:, 0] * 10
        hinge_pos_reward = (
            self.simulator.scene.articulations["door"]
            .data.joint_pos[:, 0]
            .clamp(min=0.0, max=1.5708)
            / 1.5708
        )
        return (hinge_vel_reward + hinge_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_SWING, STAGE_THROUGH])
    def _reward_target_root_distance(self):
        target_direction = F.normalize(
            self.target_root_pos - (self.simulator.robot_root_states[:, :3] - self.env_origins),
            dim=-1,
        )
        root_vel = self.simulator._rigid_body_vel[:, self.root_idx, :]
        root_vel_along_target_direction = torch.sum(root_vel * target_direction, dim=-1)
        root_vel_target = self.config.get("target_root_vel", 0.3)
        root_vel_reward = self._tracking_reward_util(
            root_vel_along_target_direction, std=0.2, target=root_vel_target, scale=1.0, offset=0.0
        )

        root_pos_diff = torch.norm(
            self.simulator.robot_root_states[:, :3] - self.env_origins - self.target_root_pos,
            dim=-1,
        )
        root_pos_reward = self._tracking_reward_util(
            root_pos_diff, std=0.2, target=0.0, scale=1.0, offset=0.0
        )
        reward = (root_vel_reward + root_pos_reward).clamp(max=1.0)
        reward[self.stage_buf == DoorPregrasp.STAGE_SWING] *= 0.8
        return reward

    @StagedTaskBase.effective_in_stage([STAGE_SWING, STAGE_THROUGH])
    def _reward_penalty_through_lateral_deviation(self):
        # Keep the robot on the doorway center-line while swinging the door open and walking
        # through. target_root_pos[1] is the center (env-relative y = 0); the 3D target_root_distance
        # reward only weakly pulls laterally while the large x-gap dominates, so the right-push
        # dynamics let the robot veer to the side. Penalize lateral (y) drift directly so it walks
        # through the middle instead of off to one side.
        lateral_y = self.simulator.robot_root_states[:, 1] - self.env_origins[:, 1]
        return (lateral_y - self.target_root_pos[:, 1]).abs()

    @StagedTaskBase.effective_in_stage([STAGE_THROUGH])
    def _reward_face_forward(self):
        angle = wrap_to_pi(
            axis_angle_from_quat(xyzw_to_wxyz(self.relative_door_rot_buf)).norm(dim=-1)
        )
        return self._tracking_reward_util(angle, std=0.6, target=0.0, scale=1.0, offset=0.0)

    @StagedTaskBase.effective_in_stage([STAGE_THROUGH])
    def _reward_through_arm_default(self):
        # In the final (through) stage, hold BOTH arms in the resting "ready" pose the robot
        # starts in (the bent shoulder/elbow + rotated-wrist pose stored in resting_dof_pos) so it
        # walks through with the same tucked arms instead of flailing them. Targets resting_dof_pos
        # (the start pose the user wants), NOT the URDF-neutral default_dof_pos.
        arm_idx = torch.cat([self._left_arm_dof_idx, self._right_arm_dof_idx])
        dev = (
            self.simulator.dof_pos[:, arm_idx] - self.resting_dof_pos[:, arm_idx]
        ).abs().mean(dim=-1)
        return self._tracking_reward_util(dev, std=0.5, target=0.0, scale=1.0, offset=0.0)

    @override
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
        if self.use_reward_limits_dof_pos_curriculum:
            m = (
                self.simulator.hard_dof_pos_limits[:, 0] + self.simulator.hard_dof_pos_limits[:, 1]
            ) / 2
            r = self.simulator.hard_dof_pos_limits[:, 1] - self.simulator.hard_dof_pos_limits[:, 0]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
        else:
            lower_soft_limit = self.simulator.dof_pos_limits[:, 0]
            upper_soft_limit = self.simulator.dof_pos_limits[:, 1]
        out_of_limits = -(self.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
        out_of_limits += (self.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
        return torch.sum(out_of_limits[:, self._upper_non_finger_dof_idx], dim=1)

    def _reward_penalty_humanly_dof_limit(self):
        lower_limit_violations = -1.0 * (
            self.simulator.dof_pos - self.dof_pos_humanly_lower_limit
        ).clip(max=0.0).sum(dim=-1)
        upper_limit_violations = (
            (self.simulator.dof_pos - self.dof_pos_humanly_upper_limit).clip(min=0.0).sum(dim=-1)
        )
        return lower_limit_violations + upper_limit_violations

    def _reward_penalty_door_frame_contact(self):
        door_frame_unwanted_contact_forces = self.simulator.scene.sensors[
            "door_frame_unwanted_contact_sensor"
        ].data.net_forces_w
        return door_frame_unwanted_contact_forces.norm(dim=-1).sum(dim=-1)

    def _reward_penalty_door_panel_contact(self):
        door_panel_unwanted_contact_forces = self.simulator.scene.sensors[
            "door_panel_unwanted_contact_sensor"
        ].data.net_forces_w
        return door_panel_unwanted_contact_forces.norm(dim=-1).sum(dim=-1)

    def _reward_penalty_head_door_frame_contact(self):
        # filtered contact forces between the head and the door frame
        # force_matrix_w shape: (num_envs, num_sensor_bodies, num_filtered_bodies, 3)
        head_door_frame_contact_forces = self.simulator.scene.sensors[
            "head_door_frame_contact_sensor"
        ].data.force_matrix_w
        return head_door_frame_contact_forces.norm(dim=-1).sum(dim=(-1, -2))

    def _reward_penalty_upper_body_dof_vel(self):
        return torch.sum(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx] ** 2, dim=-1)

    @StagedTaskBase.effective_in_stage(
        [STAGE_WALK_TO_DOOR, STAGE_PREGRASP, STAGE_GRASP, STAGE_THROUGH]
    )
    def _reward_penalty_face_door(self):
        return wrap_to_pi(
            axis_angle_from_quat(xyzw_to_wxyz(self.relative_door_rot_buf)).norm(dim=-1)
        )

    def _reward_penalty_upright(self):
        upright_vec = torch.repeat_interleave(
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device), self.num_envs, dim=0
        )
        torso_quat_wxyz = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.torso_index])
        rotated_vec = quat_apply(torso_quat_wxyz, upright_vec)
        return torch.sum(torch.square(rotated_vec - upright_vec), dim=-1)

    @override
    def _reward_penalty_dof_acc(self):
        return torch.sum(
            torch.square(self.simulator.dof_acc[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    @override
    def _reward_penalty_dof_vel(self):
        return torch.sum(
            torch.square(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    @override
    def _reward_penalty_undesired_contact(self):
        undesired_contact = torch.sum(
            torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
            > 1,
            dim=1,
            dtype=torch.float,
        )
        return undesired_contact

    def _reward_penalty_dof_overspeed(self):
        return (
            torch.maximum(
                torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]) - 2.0,
                torch.zeros_like(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]),
            )
            ** 2
        ).sum(dim=-1)

    def _get_obs_relative_to_door(self):
        relative_door_rot_6d = quat_to_tan_norm(self.relative_door_rot_buf, w_last=True)
        return torch.cat([self.relative_door_pos_buf, relative_door_rot_6d], dim=-1)

    def _get_obs_hand_handle_transform(self):
        left_hand_pos = self.simulator.left_hand_transform_pos[:, 0, :]
        left_hand_rot_wxyz = self.simulator.left_hand_transform_rot[:, 0, :]
        left_hand_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(left_hand_rot_wxyz), w_last=True)
        right_hand_pos = self.simulator.right_hand_transform_pos[:, 0, :]
        right_hand_rot_wxyz = self.simulator.right_hand_transform_rot[:, 0, :]
        right_hand_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(right_hand_rot_wxyz), w_last=True)
        return torch.cat(
            [left_hand_pos, left_hand_rot_6d, right_hand_pos, right_hand_rot_6d], dim=-1
        )

    def _get_obs_hand_force(self):
        left_hand_force = self.simulator.contact_forces[:, self.left_hand_indices, :]
        right_hand_force = self.simulator.contact_forces[:, self.right_hand_indices, :]
        return torch.cat(
            [
                left_hand_force.reshape(left_hand_force.shape[0], -1),
                right_hand_force.reshape(right_hand_force.shape[0], -1),
            ],
            dim=-1,
        )

    def _get_obs_privileged_door_info(self):
        return torch.stack(
            [
                self.door_width,
                self.door_height,
                self.door_handle_height,
                self.door_handle_width,
                self.door_weight / 100.0,
                self.door_open_lr,
                1.0 - self.door_open_lr,
                self.door_open_io,
            ],
            dim=1,
        )

    def _get_obs_door_dof_pos(self):
        return self.simulator.get_task_dof_pos("door")[:, :2]

    def _get_obs_dof_pos_non_finger(self):
        return self.simulator.dof_pos[:, self.non_finger_dof_idx]

    def _get_obs_dof_vel_non_finger(self):
        return self.simulator.dof_vel[:, self.non_finger_dof_idx]

    def _get_obs_target_obj_pos(self):
        return (
            self.simulator.scene.sensors["head_target_frame_transformer"]
            .data.target_pos_source[:, 0, :]
            .clone()
        )

    def _compute_grasp_target(self):
        grasp_target_pos_w = (
            self.simulator.scene.sensors["right_hand_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )
        return grasp_target_pos_w

    def _compute_pre_grasp_target(self):
        grasp_target_pos_w = self._compute_grasp_target()
        grasp_target_pos_w[:, 2] += 0.3  # pre-grasp sits clearly ABOVE the lever (top-down approach)
        return grasp_target_pos_w

    @override
    def _reset_object_states_callback(self, env_ids):
        self._reset_door_states(env_ids)
        return super()._reset_object_states_callback(env_ids)

    @override
    def _reset_root_states(self, env_ids, target_root_states=None):
        self.target_robot_root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
        )  # [7:10]: lin vel, [10:13]: ang vel

        r, p, _ = euler_xyz_from_quat(self.target_robot_root_states[env_ids, 3:7])
        self.target_robot_root_states[env_ids, 0:1] = torch_rand_float(
            -1.5, -0.6, (len(env_ids), 1), device=str(self.device)
        )
        self.target_robot_root_states[env_ids, 1:2] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=str(self.device)
        )
        self.target_robot_root_states[env_ids, 0:2] += self.env_origins[env_ids, 0:2]
        random_yaw = torch_rand_float(
            -torch.pi / 4, torch.pi / 4, (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        self.target_robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(r, p, random_yaw)

    @override
    def _reset_dofs(self, env_ids, target_state=None):
        # randomize wrist in +- 80 deg
        xx, yy = torch.meshgrid(env_ids, self.wrist_dof_idx)
        self.target_robot_dof_state[xx, yy, 0] = torch_rand_float(
            -1.39626, 1.39626, (len(env_ids), len(self.wrist_dof_idx)), device=str(self.device)
        )

        # completely randomize finger dofs
        xx, yy = torch.meshgrid(env_ids, self.finger_dof_idx)
        upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_upper_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_lower_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        self.target_robot_dof_state[xx, yy, 0] = lower_limit + (
            upper_limit - lower_limit
        ) * torch_rand_float(
            0.0, 1.0, (len(env_ids), len(self.finger_dof_idx)), device=str(self.device)
        )

        # set velocities to 0
        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    def _reset_door_states(self, env_ids):
        randomize_door_init_state = self.config.get("randomize_door_init_state", False)
        self.door_dof_state_buf[:] = 0.0
        if randomize_door_init_state:
            # 33% of the environments to have a different initial state
            rand_env_ids = env_ids[torch.randperm(len(env_ids))[: len(env_ids) // 3]]
            self.door_dof_state_buf[rand_env_ids, 0] = torch_rand_float(
                0.261799, 1.74533, (len(rand_env_ids), 1), device=self.device
            ).squeeze(-1)
        door_dof_state_dict = {
            "door": (
                self.door_dof_state_buf,
                torch.zeros_like(self.door_dof_state_buf),
                torch.tensor([0, 1, 2], device=self.device, dtype=torch.long),
            )
        }
        self.simulator.set_task_dof_state_tensor(env_ids, door_dof_state_dict)

        door_dof_target = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        door_dof_target[:, 0] = 0.0
        door_dof_target[:, 1] = 15 * torch.pi / 180.0  # tension the door handle
        self.simulator.apply_torques_at_task_dof(env_ids, {"door": door_dof_target})

    @override
    def _check_termination(self):
        super()._check_termination()
        self.reset_buf |= self.relative_door_pos_buf.norm(dim=-1) > 4.0

        dof_overspeed = torch.any(
            torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx])
            > self.termination_level * 20.0,
            dim=-1,
        )
        not_just_resetted = self.episode_length_buf > 20

        self.reset_buf |= dof_overspeed & not_just_resetted

        # reset if the homie command is too large when grasping or opening the door
        # is_grasping_or_opening = (self.stage_buf == DoorPregrasp.STAGE_GRASP) | (self.stage_buf == DoorPregrasp.STAGE_OPEN)
        # homie_command_norm = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1)
        # self.reset_buf |= (homie_command_norm > self.termination_level) & is_grasping_or_opening

    def init_eval_metrics_tracking(self, device):
        # Flat per-episode goal-reached records collected over the whole eval run. (The base
        # implementation only kept zero-tensors that were never populated, so eval success rate
        # was always 0 / unserializable.) Keep "goal_reached_buffer" present for trainer compat.
        self.eval_metrics = {
            "episode_goal_reached": [],
            "episode_last_stage_goal_reached": [],
            "goal_reached_buffer": [],
        }

    def process_eval_episode_completions(
        self, completed_env_ids, cur_reward_sum, cur_episode_length
    ):
        # Called when envs finish an episode (after env.step has updated the completion buffers).
        # Record whether each just-finished episode reached the goal.
        ids = completed_env_ids.reshape(-1)
        self.eval_metrics["episode_goal_reached"].extend(
            self.last_completed_task_buf[ids].detach().cpu().tolist()
        )
        self.eval_metrics["episode_last_stage_goal_reached"].extend(
            self.last_last_stage_completed_task_buf[ids].detach().cpu().tolist()
        )
        self.eval_metrics["goal_reached_buffer"].append(
            self.last_completed_task_buf[ids].detach().cpu()
        )

    def get_eval_metrics_summary(self):
        gr = [bool(x) for x in self.eval_metrics.get("episode_goal_reached", [])]
        lsg = [bool(x) for x in self.eval_metrics.get("episode_last_stage_goal_reached", [])]
        n = len(gr)
        rate = (sum(gr) / n) if n else 0.0
        last_stage_rate = (sum(lsg) / len(lsg)) if lsg else 0.0
        summary = {
            "num_episodes": n,
            "goal_reached_count": int(sum(gr)),
            "goal_reached_rate": rate,
            "last_stage_goal_reached_rate": last_stage_rate,
            "episode_goal_reached": gr,
        }
        print(
            f"[EVAL SUMMARY] goal_reached_rate={rate:.4f} "
            f"({int(sum(gr))}/{n} episodes), last_stage_rate={last_stage_rate:.4f}"
        )
        return summary

    @property
    def ground_height(self):
        return 0.0

    def _stage_0_reward_condition(self):
        # walk to the door
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _stage_0_to_complete_condition(self):
        return self._stage_0_to_1_advance_condition()

    def _stage_0_to_1_advance_condition(self):
        # get close enough to the door
        grasp_target = self._compute_grasp_target()
        root_pos = self.simulator.robot_root_states[:, :3].clone()
        root_pos[:, 2] = grasp_target[:, 2]
        # 0.3: stop close enough to hover over the lever. The arm no longer needs to be far to avoid
        # the catch -- penalty_disturb_lever (a learnable signal, not a geometric trick) keeps the
        # arm from clipping the lever on the way up.
        cond = (root_pos - grasp_target).norm(dim=-1) < 0.3

        # keep hands down
        max_deviation = (
            torch.abs(
                self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
                - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
            )
            .max(dim=-1)
            .values
        )
        cond &= max_deviation < 0.25
        return cond

    def _stage_1_reward_condition(self):
        # small homie command
        cond = torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1
        # stay close to the door
        cond &= self._stage_0_to_1_advance_condition()
        return cond

    def _stage_1_to_complete_condition(self):
        return self._stage_1_to_2_advance_condition()

    def _stage_1_to_2_advance_condition(self):
        # raise hand to pre-grasp position
        pre_grasp_target = self._compute_pre_grasp_target()

        left_palm_body_pos = self.simulator._rigid_body_pos[:, self.left_palm_idx, :]
        left_hand_above_handle = left_palm_body_pos[:, 2] > self.door_handle_height + 0.05
        left_hand_close_to_pre_grasp_target = (left_palm_body_pos - pre_grasp_target).norm(
            dim=-1
        ) < 0.1
        left_hand_close_to_pre_grasp_dof_target = (
            torch.abs(self.simulator.dof_pos[:, self._left_hand_dof_idx] - self._left_p0).mean(
                dim=-1
            )
            < 0.174533
        )
        left_hand_cond = (
            left_hand_above_handle
            & left_hand_close_to_pre_grasp_target
            & left_hand_close_to_pre_grasp_dof_target
        )

        right_palm_body_pos = self.simulator._rigid_body_pos[:, self.right_palm_idx, :]
        right_hand_above_handle = right_palm_body_pos[:, 2] > self.door_handle_height + 0.05
        right_hand_close_to_pre_grasp_target = (right_palm_body_pos - pre_grasp_target).norm(
            dim=-1
        ) < 0.1
        right_hand_close_to_pre_grasp_dof_target = (
            torch.abs(self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p0).mean(
                dim=-1
            )
            < 0.174533
        )
        right_hand_cond = (
            right_hand_above_handle
            & right_hand_close_to_pre_grasp_target
            & right_hand_close_to_pre_grasp_dof_target
        )

        cond = torch.where(self.door_open_lr < 0, left_hand_cond, right_hand_cond)

        cond &= self._reward_hand_handle_orientation() > 0.2

        cond &= torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1

        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533

        return cond | door_opened

    def _stage_2_reward_condition(self):
        return torch.norm(self.get_physical_homie_commands()[:, :3], dim=1) <= 0.1

    def _stage_2_to_complete_condition(self):
        # TODO: check error
        # grasp the door handle
        left_hand_handle_contact_count = (
            self.simulator.object_to_hand_contact_forces[
                :, 0, self.left_hand_indices_tgt_ct_sensor, :
            ].norm(dim=-1)
            > 1
        ).sum(dim=-1)
        left_hand_grasped = left_hand_handle_contact_count >= 1

        right_hand_handle_contact_count = (
            self.simulator.object_to_hand_contact_forces[
                :, 0, self.right_hand_indices_tgt_ct_sensor, :
            ].norm(dim=-1)
            > 1
        ).sum(dim=-1)
        right_hand_grasped = right_hand_handle_contact_count >= 1
        return torch.where(self.door_open_lr < 0, left_hand_grasped, right_hand_grasped)

    def _stage_2_to_3_advance_condition(self):
        # grasp the door handle
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533
        return self._stage_2_to_complete_condition() | door_opened

    def _stage_3_reward_condition(self):
        # keep grasping the door handle
        return self._stage_2_to_3_advance_condition() & self._stage_2_reward_condition()

    def _stage_3_to_4_advance_condition(self):
        # rotate the door handle and open the door
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533
        return door_opened

    def _stage_4_reward_condition(self):
        # keep grasping the door handle
        return self._stage_3_to_4_advance_condition()

    def _stage_4_to_5_advance_condition(self):
        # walk through the door and leave handle up
        walked_through_door = (
            self.simulator.robot_root_states[:, 0] - self.env_origins[:, 0]
        ) > 0.0
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 1.0472
        handle_up = self.simulator.scene.articulations["door"].data.joint_pos[:, 1] < 0.2
        return walked_through_door & handle_up & door_opened

    def _stage_5_reward_condition(self):
        # keep walking through the door
        return self._stage_4_to_5_advance_condition()

    def _stage_5_to_complete_condition(self):
        return (self.simulator.robot_root_states[:, 0] - self.env_origins[:, 0]) > 1.5

    def scene_creation_callback(self, simulator):
        door_frame_unwanted_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/root",
        )

        door_panel_unwanted_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/door_panel",
        )
        simulator.scene.sensors["door_frame_unwanted_contact_sensor"] = ContactSensor(
            door_frame_unwanted_contact_sensor_config
        )
        simulator.scene.sensors["door_panel_unwanted_contact_sensor"] = ContactSensor(
            door_panel_unwanted_contact_sensor_config
        )

        # contact between the robot's head and the door frame (filtered contact)
        head_door_frame_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/head_link",
            filter_prim_paths_expr=[
                f"/World/envs/env_.*/{simulator.task_config.target_obj}/root"
            ],
        )
        simulator.scene.sensors["head_door_frame_contact_sensor"] = ContactSensor(
            head_door_frame_contact_sensor_config
        )

        head_target_frame_transformer_config: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/head_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=simulator.scene.sensors["left_hand_frame_transformer"]
                    .cfg.target_frames[0]
                    .prim_path
                ),
            ],
        )
        simulator.scene.sensors["head_target_frame_transformer"] = FrameTransformer(
            head_target_frame_transformer_config
        )

    def _parse_palm_side_direction(self, palm_side_direction: list[str]) -> torch.Tensor:
        """
        Convert the palm side direction to a quaternion that rotates anything
        expressed in the finger frame to point into the palm.
        """
        output = torch.zeros(len(palm_side_direction), 4, device=self.device)  # wxyz
        for i, direction in enumerate(palm_side_direction):
            if direction == "+x":
                output[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            elif direction == "-x":
                output[i] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            elif direction == "+y":
                output[i] = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device)
            elif direction == "-y":
                output[i] = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=self.device)
            elif direction == "+z":
                output[i] = torch.tensor([0.7071068, 0.0, -0.7071068, 0.0], device=self.device)
            elif direction == "-z":
                output[i] = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=self.device)
            else:
                raise ValueError(f"Invalid palm side direction: {direction}")
        return output
