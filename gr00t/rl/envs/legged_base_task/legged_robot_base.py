# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import os
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict
from termcolor import colored
from torchvision.transforms import v2

from gr00t.rl.envs.base_task.base_task import BaseTask
from gr00t.rl.envs.env_utils.history_handler import HistoryHandler
from gr00t.rl.envs.env_utils.visualization import Point
from gr00t.rl.envs.legged_base_task.live_reward_plotter import RewardPlotter
from gr00t.rl.utils.torch_utils import *  # noqa: F403
from gr00t.rl.isaac_utils.rotations import (  # noqa: E402
    get_euler_xyz_in_tensor,
    quat_from_angle_axis,
    quat_mul_norm,
    wrap_to_pi,
)
from gr00t.rl.utils.helpers import parse_observation
from gr00t.rl.utils.torch_utils import to_torch

# from isaacgym import gymtorch, gymapi, gymutil






class LeggedRobotBase(BaseTask):
    # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._domain_rand_config()
        self._prepare_reward_function()
        self.history_handler = HistoryHandler(
            self.num_envs, config.obs.obs_auxiliary, config.obs.obs_dims, device, config
        )
        self.is_evaluating = False

        # History save interval - save observations every n steps instead of every step
        self.history_save_interval = config.obs.get("history_save_interval", 1)

        self.init_done = True
        if self.config.get("use_z", False):
            self._load_z_model()

        if not self.headless:
            if self.simulator.simulator_config.name == "isaacsim":
                self.simulator.add_keyboard_callback("R", self.reset_all)
                self.simulator.add_keyboard_callback("P", self.push_all_robots)
                self.simulator.add_keyboard_callback("O", self.push_all_robots_by_force)
                self.simulator.add_keyboard_callback("P", self.push_all_robots)
                self.simulator.add_keyboard_callback("O", self.push_all_robots_by_force)
            elif self.simulator.simulator_config.name == "isaacgym":
                from isaacgym import gymapi

                self.simulator.add_keyboard_callback(gymapi.KEY_P, self.push_all_robots, "Push")
                self.simulator.add_keyboard_callback(gymapi.KEY_R, self.reset_all, "Reset")
        self.target_task_root_states = copy.deepcopy(self.simulator.all_root_states)

        if self.config.get("live_reward_analysis", False):
            import logging

            import viser

            logging.getLogger("websockets.server").setLevel(logging.WARNING)
            self.reward_plotter = RewardPlotter()
            self.viser_server = viser.ViserServer()
            self.reward_plot = self.viser_server.gui.add_plotly(
                self.reward_plotter.show(), aspect=2
            )

    def _init_buffers(self, reinit_sim=False):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # initialize some data used later on
        self._init_counters()
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions_after_delay = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(self.simulator.robot_root_states[:, 7:13])
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 10:13]
        )

        self.target_robot_root_states = torch.zeros(
            self.num_envs, 13, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_robot_dof_state = torch.zeros(
            self.num_envs,
            self.num_dof,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        if hasattr(self, "end_effector_index"):
            self.end_effector_vel = self.simulator._rigid_body_vel[:, self.end_effector_index, :]
            self.end_effector_ang_vel = self.simulator._rigid_body_ang_vel[
                :, self.end_effector_index, :
            ]
            self.end_effector_pos = self.simulator._rigid_body_pos[:, self.end_effector_index]
            self.end_effector_rot = self.simulator._rigid_body_rot[:, self.end_effector_index, :]
            self.end_effector_rot_gravity = quat_rotate_inverse(
                self.end_effector_rot, self.gravity_vec
            )
            self.pre_end_effector_vel = torch.zeros_like(self.end_effector_vel)
            self.pre_end_effector_ang_vel = torch.zeros_like(self.end_effector_ang_vel)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.push_robot_recovery_counter = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(
                        f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}"
                    )
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
                    raise ValueError(
                        f"PD gain of joint {name} were not defined. Should be defined in the yaml file."
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.dof_scales = self.dof_pos_limits.abs().max(dim=-1).values + self.default_dof_pos.abs()

        if not reinit_sim:
            self._init_domain_rand_buffers()
            self._meta_kp_scale = torch.ones(
                self.num_envs,
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self._meta_kd_scale = torch.ones(
                self.num_envs,
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

            # for reward penalty curriculum
            self.average_episode_length = (
                0.0  # num_compute_average_epl last termination episode length
            )
            self.last_episode_length_buf = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self.num_compute_average_epl = to_torch(
                self.config.rewards.num_compute_average_epl, device=self.device
            )

            self.need_to_refresh_envs = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
            )

            self.add_noise_currculum = self.config.obs.add_noise_currculum
            self.current_noise_curriculum_value = to_torch(
                self.config.obs.noise_initial_value, device=self.device
            )

            # Initialize past observation buffer
            self.past_obs_buf_dict = {}
            self.use_past_obs = self.config.obs.get("use_past_obs", False)
            self.past_length = self.config.obs.get("past_length", 1)

        self.visualize_env_id = 0

    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            self.push_interval_s = torch.randint(
                self.config.domain_rand.push_interval_s[0],
                self.config.domain_rand.push_interval_s[1],
                (self.num_envs,),
                device=self.device,
            )

    def _init_counters(self):
        self.common_step_counter = 0
        self.push_robot_counter = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )
        self.push_robot_plot_counter = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )
        self.command_counter = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )

    def _update_counters_each_step(self):
        self.common_step_counter += 1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs,
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                self.dim_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.action_delay_idx = torch.randint(
                self.config.domain_rand.ctrl_delay_step_range[0],
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                (self.num_envs,),
                device=self.device,
                requires_grad=False,
            )

        # self._link_mass_scale = torch.ones(self.num_envs, len(self.config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._kd_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._rfi_lim_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.last_contacts_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_max_height = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Force based push related
        self.push_force = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.push_force_pos = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        if self.config.robot.get("fixed_kp_scale", None) is not None:
            self._kp_scale[:] = self.config.robot.fixed_kp_scale
        if self.config.robot.get("fixed_kd_scale", None) is not None:
            self._kd_scale[:] = self.config.robot.fixed_kd_scale

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(
            colored(
                f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}",
                "green",
            )
        )

        self.reward_scales = self.config.rewards.reward_scales
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.use_reward_penalty_curriculum = to_torch(
            self.config.rewards.reward_penalty_curriculum, device=self.device
        )
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = to_torch(
                self.config.rewards.reward_initial_penalty_scale, device=self.device
            )

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(
                f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}"
            )

        self.use_reward_limits_dof_pos_curriculum = to_torch(
            self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum,
            device=self.device,
        )
        self.use_reward_limits_dof_vel_curriculum = to_torch(
            self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum,
            device=self.device,
        )
        self.use_reward_limits_torque_curriculum = to_torch(
            self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum,
            device=self.device,
        )

        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(
                f"Use Reward Limits DOF Curriculum: {self.use_reward_limits_dof_pos_curriculum}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}"
            )
            self.soft_dof_pos_curriculum_value = to_torch(
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit,
                device=self.device,
            )

        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(
                f"Use Reward Limits DOF Vel Curriculum: {self.use_reward_limits_dof_vel_curriculum}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}"
            )
            self.soft_dof_vel_curriculum_value = to_torch(
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit,
                device=self.device,
            )

        if self.use_reward_limits_torque_curriculum:
            logger.info(
                f"Use Reward Limits Torque Curriculum: {self.use_reward_limits_torque_curriculum}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}"
            )
            self.soft_torque_curriculum_value = to_torch(
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit,
                device=self.device,
            )

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
            # reward episode sums
            self.episode_sums = {
                name: torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
                )
                for name in self.reward_scales.keys()
            }

    def set_is_evaluating(self, is_evaluating=True, log_info=True, **kwargs):
        if log_info:
            logger.info(f"Setting Env is evaluating to {is_evaluating}")
        self.is_evaluating = is_evaluating

    def step(self, actor_state):
        """Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions = actor_state["actions"]
        # actions *= 0.0
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value

        if self.config.get("use_z", False):
            self.actions[:] = self.get_z_action(actions)
        else:
            self.actions[:] = actions

        self.actions[:] = torch.clip(self.actions, -clip_action_limit, clip_action_limit).to(
            self.device
        )

        self.log_dict["action_clip_frac"] = (
            self.actions.abs() == clip_action_limit
        ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[
                torch.arange(self.num_envs), self.action_delay_idx
            ].clone()
        else:
            self.actions_after_delay = self.actions.clone()

        if hasattr(self, "end_effector_index"):
            self.pre_end_effector_vel[:] = self.simulator._rigid_body_vel[
                :, self.end_effector_index, :
            ]
            self.pre_end_effector_ang_vel[:] = self.simulator._rigid_body_ang_vel[
                :, self.end_effector_index, :
            ]

    def render_results(self):
        if (
            self.config.simulator.config.get("render_results", False)
            and self.config.simulator.config.name == "isaacsim"
        ):
            if self.debug_viz:
                self._draw_debug_vis()

            rgb_viewer = self.simulator.eval_camera.data.output["rgb"].clone()

            if "writers_rendering_results" not in self.__dict__:

                os.makedirs(self.config.save_rendering_dir, exist_ok=True)
                self.writers_rendering_results = []
                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for i in range(self.num_envs):
                    file_name = f"{self.config.save_rendering_dir}/{time_str}_viewer_{i}.mp4"
                    print(f"Saving rendering to {file_name}")
                    writer = imageio.get_writer(file_name, fps=int(1 / self.dt))
                    writer.append_data(rgb_viewer[i].cpu().numpy())
                    self.writers_rendering_results.append(writer)
            else:
                for i in range(self.num_envs):
                    self.writers_rendering_results[i].append_data(rgb_viewer[i].cpu().numpy())

    def end_render_results(self):
        if (
            self.config.simulator.config.get("render_results", False)
            and self.config.simulator.config.name == "isaacsim"
        ):
            for writer in self.writers_rendering_results:
                writer.close()

    def end_render_results(self):
        if (
            self.config.simulator.config.get("render_results", False)
            and self.config.simulator.config.name == "isaacsim"
        ):
            for writer in self.writers_rendering_results:
                writer.close()

    def _update_meta_pd_scale(self, sim_sub_t):
        use_meta_pd = self.config.robot.get("use_meta_pd", False)
        meta_pd_freq = self.config.simulator.config.sim.control_decimation // self.config.robot.get(
            "meta_pd_steps", 10
        )
        meta_kp_temp = self.config.robot.get("meta_kp_temp", 1.0)
        meta_kd_temp = self.config.robot.get("meta_kd_temp", 1.0)
        actions = self.actions_after_delay
        if use_meta_pd and sim_sub_t % meta_pd_freq == 0:
            pd_actions = actions[:, self.num_dof :]
            pd_actions = pd_actions.view(
                self.num_envs,
                self.config.robot.meta_pd_steps,
                2,
                self.config.robot.meta_pd_dim_size,
            )[:, sim_sub_t]
            meta_kp_scale = torch.sigmoid(pd_actions[:, 0] * meta_kp_temp)
            meta_kd_scale = torch.sigmoid(pd_actions[:, 1] * meta_kd_temp)
            self._meta_kp_scale[:] = (
                self.config.robot.meta_kp_scale_limit[0]
                + (
                    self.config.robot.meta_kp_scale_limit[1]
                    - self.config.robot.meta_kp_scale_limit[0]
                )
                * meta_kp_scale
            )
            self._meta_kd_scale[:] = (
                self.config.robot.meta_kd_scale_limit[0]
                + (
                    self.config.robot.meta_kd_scale_limit[1]
                    - self.config.robot.meta_kd_scale_limit[0]
                )
                * meta_kd_scale
            )

    def _physics_step(self):
        self.render()
        for sim_sub_t in range(self.config.simulator.config.sim.control_decimation):
            self._update_meta_pd_scale(sim_sub_t)
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _compute_perturbation_forces(self):
        """
        Should be implemented in the child class if needed.
        """
        pass

    def _set_perturbation_forces(self):
        """
        Should be implemented in the child class if needed.
        Update force applied pos every substep
        """
        pass

    def _apply_force_in_physics_step(self):
        if self.config.simulator.config.name == "isaacgym":
            self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
            self.simulator.apply_torques_at_dof(self.torques)
        elif self.config.simulator.config.name == "isaacsim":
            self.torques = self._compute_torques(self.actions_after_delay).view(
                self.torques.shape
            )  # When using implicit PD
            actions_scaled = self.actions_after_delay * self.config.robot.control.action_scale
            jpos_target = actions_scaled + self.default_dof_pos
            # jpos_target *= 0.
            self.simulator._robot.set_joint_position_target(
                jpos_target, joint_ids=self.simulator.dof_ids
            )
        else:
            raise NotImplementedError(
                f"Simulator {self.config.simulator.config.name} not implemented"
            )
        self._set_perturbation_forces()
        self.simulator.apply_rigid_body_force_at_pos_tensor(self.push_force, self.push_force_pos)

    def _action_backmap(self):
        return (
            self.simulator.dof_pos - self.default_dof_pos
        ) / self.config.robot.control.action_scale

    def reset_all(self):
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(
            torch.arange(self.num_envs, device=self.device), self.target_robot_root_states
        )
        self.simulator.set_dof_state_tensor(
            torch.arange(self.num_envs, device=self.device), self.target_robot_dof_state
        )
        # self.simulator.set_task_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.target_task_root_states)
        # self.simulator.set_task_visual_state_tensor(torch.arange(self.num_envs, device=self.device))
        self._refresh_sim_tensors()

        self._pre_compute_observations_callback()
        self._compute_observations()
        self._post_compute_observations_callback()

        if self.use_past_obs:
            self._reset_past_obs_callback()
            self._update_obs_buf_dict_with_past_obs()
        return self.obs_buf_dict

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            # ZL: this is a terrible way to set the sim values.
            self.simulator.set_actor_root_state_tensor(
                refresh_env_ids, self.target_robot_root_states
            )
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.target_robot_dof_state)
            # self.simulator.set_task_root_state_tensor(refresh_env_ids, self.target_task_root_states)
            # self.simulator.set_task_visual_state_tensor(refresh_env_ids)
            self.need_to_refresh_envs[refresh_env_ids] = False

        # [Hardcoded] set world pose for the ego camera
        base_pos = self.simulator.robot_root_states[:, 0:3]
        base_quat = self.simulator.base_quat

        # Lower the camera position
        base_pos[:, 2] -= 0.11

        # Extract yaw from base quaternion
        euler_angles = get_euler_xyz_in_tensor(base_quat)
        yaw = euler_angles[:, 2]  # Get just the yaw component

        # print("yaw", yaw)
        # Create fixed pitch rotation (looking down)
        pitch_angle = torch.tensor(np.pi / 2.0, device=self.device).repeat(self.num_envs)
        pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        pitch_quat = quat_from_angle_axis(pitch_angle, pitch_axis, w_last=True)

        # Create yaw rotation from base
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        yaw_quat = quat_from_angle_axis(yaw, yaw_axis, w_last=True)

        # Combine rotations: first apply pitch, then yaw
        camera_quat = quat_mul_norm(yaw_quat, pitch_quat, w_last=True)

        # Set the ego camera pose with the adjusted orientation
        # if hasattr(self.simulator.simulator_config, "cameras") and self.simulator.simulator_config.cameras.enable_cameras:
        #     self.simulator.set_ego_camera_pose(base_pos, camera_quat)

        if len(env_ids) > 0:
            self._pre_compute_observations_callback(env_ids=env_ids)
        self._compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        if self.config.get("use_z", False):
            self._compute_z_prior()

        self._post_compute_observations_callback()

        if self.use_past_obs:
            self._update_obs_buf_dict_with_past_obs()

        # Only save history observations every n steps based on history_save_interval
        # Since all environments step together, they all save at the same time
        should_save_history = (self.common_step_counter % self.history_save_interval) == 0

        if should_save_history:
            for key in self.history_handler.history.keys():
                self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
                # self._draw_end_effector_rotate_and_acc()

    def _update_obs_buf_dict_with_past_obs(self):
        for obs_key, obs_val in self.obs_buf_dict.items():
            # Shift buffer and add new observation
            self.past_obs_buf_dict[obs_key][:, :-1] = self.past_obs_buf_dict[obs_key][:, 1:].clone()
            self.past_obs_buf_dict[obs_key][:, -1] = obs_val
            self.obs_buf_dict[obs_key] = self.past_obs_buf_dict[obs_key].reshape(
                self.past_obs_buf_dict[obs_key].shape[0], -1
            )

    def _reset_past_obs_callback(self, env_ids=None):
        for obs_key, obs_val in self.obs_buf_dict.items():
            if obs_key not in self.past_obs_buf_dict:
                # Initialize buffer with zeros
                self.past_obs_buf_dict[obs_key] = torch.zeros(
                    obs_val.shape[:-1] + (self.past_length, obs_val.shape[-1]),
                    dtype=obs_val.dtype,
                    device=self.device,
                )
            if env_ids is not None:
                self.past_obs_buf_dict[obs_key][env_ids] = 0
            else:
                self.past_obs_buf_dict[obs_key][:] = 0

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    def _pre_compute_observations_callback(self, env_ids=None):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 10:13]
        )
        # print("self.base_ang_vel", self.base_ang_vel)
        if hasattr(self, "end_effector_index"):
            self.end_effector_vel[:] = self.simulator._rigid_body_vel[:, self.end_effector_index, :]
            self.end_effector_ang_vel[:] = self.simulator._rigid_body_ang_vel[
                :, self.end_effector_index, :
            ]
            self.end_effector_pos = self.simulator._rigid_body_pos[:, self.end_effector_index]
            self.end_effector_rot[:] = self.simulator._rigid_body_rot[:, self.end_effector_index, :]
            self.end_effector_rot_gravity[:] = quat_rotate_inverse(
                self.end_effector_rot, self.gravity_vec
            )
            # import ipdb; ipdb.set_trace()
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _update_tasks_callback(self):

        # if (self.episode_length_buf[0] - 1) % 100 == 0:
        #     self._push_robots(torch.arange(self.num_envs, device=self.device))

        if (
            self.config.domain_rand.push_robots and not self.is_evaluating
        ):  # don't push robots when evaluating
            self.push_robot_recovery_counter[self.push_robot_recovery_counter > 0] -= 1
            push_robot_env_ids = self.push_robot_counter == (self.push_interval_s / self.dt).int()
            push_at_episode_start = self.config.domain_rand.get("push_at_episode_start", False)
            if push_at_episode_start:
                push_robot_env_ids = push_robot_env_ids | (self.episode_length_buf == 1)
            push_robot_env_ids = push_robot_env_ids.nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(
                self.config.domain_rand.push_interval_s[0],
                self.config.domain_rand.push_interval_s[1],
                (len(push_robot_env_ids),),
                device=self.device,
                requires_grad=False,
            )
            self._push_robots(push_robot_env_ids)
            self.push_robot_recovery_counter[push_robot_env_ids] = int(
                self.config.domain_rand.get("push_robot_recovery_time", 2) / self.dt
            )  # 2 seconds recovery time

    def _post_compute_observations_callback(self):
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.simulator.dof_pos[:]
        self.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

    def _check_termination(self):
        """Check if environments need to be reset"""
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

        self.reset_buf |= self.time_out_buf

    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            termination_contact_force = self.config.termination_scales.get(
                "termination_contact_force", 1.0
            )
            self.reset_buf |= torch.any(
                torch.norm(
                    self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1
                )
                > termination_contact_force,
                dim=1,
            )

        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            self.reset_buf |= torch.any(
                torch.abs(self.projected_gravity[:, 0:1])
                > self.config.termination_scales.termination_gravity_x,
                dim=1,
            )
            self.reset_buf |= torch.any(
                torch.abs(self.projected_gravity[:, 1:2])
                > self.config.termination_scales.termination_gravity_y,
                dim=1,
            )
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            robot_height = self.simulator.robot_root_states[:, 2:3] - self.ground_height
            self.reset_buf |= torch.any(
                robot_height < self.config.termination_scales.termination_min_base_height, dim=1
            )
        if self.config.termination.get("terminate_by_hand_object_contact", False):
            termination_contact_force = self.config.termination_scales.get(
                "termination_contact_force", 1.0
            )
            self.reset_buf |= torch.any(
                torch.norm(self.simulator.object_to_hand_contact_forces[:, :, :].squeeze(1), dim=-1)
                > termination_contact_force,
                dim=1,
            )

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(
                self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]
            ).clip(
                max=0.0
            )  # lower limit
            out_of_dof_pos_limits += (
                self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]
            ).clip(min=0.0)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_dof_pos_limit
            ):
                self.reset_buf |= out_of_dof_pos_limits > 0.0

        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits
                    * self.config.termination_scales.termination_close_to_dof_vel_limit
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_dof_vel_limit
            ):
                self.reset_buf |= out_of_dof_vel_limits > 0.0

        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum(
                (
                    torch.abs(self.torques)
                    - self.torque_limits
                    * self.config.termination_scales.termination_close_to_torque_limit
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_torque_limit
            ):
                self.reset_buf |= out_of_torque_limits > 0.0

    def _update_timeout_buf(self):
        self.time_out_buf |= (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)  # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)
        self._reset_object_states_callback(env_ids)
        self._reset_past_obs_callback(env_ids)
        self._post_reset_callback(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf

        if self.config.domain_rand.push_robots:
            self.push_robot_recovery_counter[env_ids] = 0  # recovery counter reset.

    def _post_reset_callback(self, env_ids):
        pass

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_object_states_callback(self, env_ids, target_states=None):
        pass

    def _reset_tasks_callback(self, env_ids):
        self._episodic_domain_randomization(env_ids)
        # if self.config.simulator.config.name == 'isaacsim':
        #     self.simulator.trigger_reset_events()

        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if (
            self.use_reward_limits_dof_pos_curriculum
            or self.use_reward_limits_dof_vel_curriculum
            or self.use_reward_limits_torque_curriculum
        ):
            self._update_reward_limits_curriculum()
        if self.add_noise_currculum:
            self._update_obs_noise_curriculum()

    def _update_obs_noise_curriculum(self):
        if (
            self.average_episode_length
            < self.config.obs.soft_dof_pos_curriculum_level_down_threshold
        ):
            self.current_noise_curriculum_value *= (
                1 - self.config.obs.soft_dof_pos_curriculum_degree
            )
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.current_noise_curriculum_value *= (
                1 + self.config.obs.soft_dof_pos_curriculum_degree
            )

        self.current_noise_curriculum_value = torch.clip(
            self.current_noise_curriculum_value,
            self.config.obs.noise_value_min,
            self.config.obs.noise_value_max,
        )

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if target_buf is not None:
            raise NotImplementedError(
                "Target buf is no longer supported for legged robot base task"
            )
        else:
            self.actions[env_ids] = 0.0
            self.last_actions[env_ids] = 0.0
            if hasattr(self, "end_effector_index"):
                self.end_effector_vel[env_ids] = 0.0
                self.end_effector_ang_vel[env_ids] = 0.0
                self.pre_end_effector_vel[env_ids] = 0.0
                self.pre_end_effector_ang_vel[env_ids] = 0.0
                self.end_effector_pos[env_ids] = 0.0
                self.end_effector_rot[env_ids] = 0.0
                self.end_effector_rot_gravity[env_ids] = 0.0
            self.actions_after_delay[env_ids] = 0.0
            self.last_dof_pos[env_ids] = 0.0
            self.last_dof_vel[env_ids] = 0.0
            self.feet_air_time[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            # self.reset_buf[env_ids] = 0
            # self.time_out_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._update_average_episode_length(env_ids)

            self.history_handler.reset(env_ids)

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        reward_dict = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            try:
                rew = self.reward_functions[i]() * self.reward_scales[name]
            except:
                import ipdb

                ipdb.set_trace()

            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb

                ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
            reward_dict[name] = rew.mean().item() / self.dt

        if self.config.get("live_reward_analysis", False):
            self.reward_plotter.add_step(reward_dict)
            self.reward_plot.figure = self.reward_plotter.show()

        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        self.log_dict["average_episode_length"] = self.average_episode_length
        if self.use_reward_penalty_curriculum:
            self.log_dict["reward_penalty_scale"] = to_torch(
                self.reward_penalty_scale, dtype=torch.float, device=self.device
            )

        if self.use_reward_limits_dof_pos_curriculum:
            self.log_dict["soft_dof_pos_curriculum_value"] = to_torch(
                self.soft_dof_pos_curriculum_value, dtype=torch.float, device=self.device
            )
        if self.use_reward_limits_dof_vel_curriculum:
            self.log_dict["soft_dof_vel_curriculum_value"] = to_torch(
                self.soft_dof_vel_curriculum_value, dtype=torch.float, device=self.device
            )
        if self.use_reward_limits_torque_curriculum:
            self.log_dict["soft_torque_curriculum_value"] = to_torch(
                self.soft_torque_curriculum_value, dtype=torch.float, device=self.device
            )

        if self.add_noise_currculum:
            self.log_dict["current_noise_curriculum_value"] = to_torch(
                self.current_noise_curriculum_value, dtype=torch.float, device=self.device
            )

    def _compute_observations(self):
        """Computes observations"""
        # super().compute_observations()
        self.obs_buf_dict_raw = TensorDict()
        self.hist_obs_dict = TensorDict()

        if self.add_noise_currculum:
            noise_extra_scale = self.current_noise_curriculum_value
        else:
            noise_extra_scale = 1.0
        # print("noise_extra_scale", noise_extra_scale)
        # compute Algo observations
        no_noise_obs_keys = self.config.obs.get("no_noise_obs_keys", ["critic_obs"])
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = TensorDict()
            parse_observation(
                self,
                obs_config,
                self.obs_buf_dict_raw[obs_key],
                self.config.obs.obs_scales,
                self.config.obs.noise_scales,
                noise_extra_scale,
                use_noise=obs_key not in no_noise_obs_keys,
            )

        # Compute history observations
        for k in self.history_handler.history.keys():
            if k in self.obs_buf_dict_raw["actor_obs"]:
                self.hist_obs_dict[k] = self.obs_buf_dict_raw["actor_obs"][
                    k
                ]  # ZL: This assumes that history obs are always in actor_obs.
            elif k in self.obs_buf_dict_raw["critic_obs"]:
                self.hist_obs_dict[k] = self.obs_buf_dict_raw["critic_obs"][k]
            elif k in self.obs_buf_dict_raw["homie_obs"]:
                self.hist_obs_dict[k] = self.obs_buf_dict_raw["homie_obs"][k]
            elif k in self.obs_buf_dict_raw["teacher_obs"]:
                self.hist_obs_dict[k] = self.obs_buf_dict_raw["teacher_obs"][k]
            elif "vision_obs" in self.obs_buf_dict_raw:
                # Handle vision observations
                if k in self.obs_buf_dict_raw["vision_obs"]:
                    self.hist_obs_dict[k] = self.obs_buf_dict_raw["vision_obs"][k]
                else:
                    # If vision observation is not found, try to compute it directly
                    if k == "rgb_image":
                        # Try to get the rgb_image observation directly (single frame)
                        if hasattr(self, "_get_obs_rgb_image"):
                            self.hist_obs_dict[k] = self._get_obs_rgb_image()
                        else:
                            raise ValueError(
                                f"History obs {k} not found in vision_obs and _get_obs_rgb_image method not available"
                            )
                    else:
                        raise ValueError(f"History obs {k} not found in vision_obs")
            else:
                raise ValueError(
                    f"History obs {k} not found in actor_obs, critic_obs, or vision_obs"
                )

        self._post_config_observation_callback()

    def get_env_state_dict(self):
        state_dict = super().get_env_state_dict()
        state_dict.update({k: v for k, v in self.log_dict.items()})
        return state_dict

    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()

        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config)
            obs_keys = [k[:-4] if k.endswith("_raw") else k for k in obs_keys]
            self.obs_buf_dict[obs_key] = torch.cat(
                [self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1
            )

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions of shape (num_envs, num_actions_per_env)
                                  First num_dof dimensions are position targets
                                  Next num_dof dimensions are kp_scale (optional)
                                  Last num_dof dimensions are kd_scale (optional)

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        pos_targets = actions[:, : self.num_dof]
        if self.config.robot.control.get("rescale_action_by_dof_limit", False):
            pos_targets = pos_targets * self.dof_scales
        else:
            pos_targets = pos_targets * self.config.robot.control.action_scale

        control_type = self.config.robot.control.control_type

        if control_type == "P":
            torques = (
                self._kp_scale
                * self._meta_kp_scale
                * self.p_gains
                * (pos_targets + self.default_dof_pos - self.simulator.dof_pos)
                - self._kd_scale * self._meta_kd_scale * self.d_gains * self.simulator.dof_vel
            )
        elif control_type == "V":
            torques = (
                self._kp_scale
                * self._meta_kp_scale
                * self.p_gains
                * (pos_targets - self.simulator.dof_vel)
                - self._kd_scale
                * self._meta_kd_scale
                * self.d_gains
                * (self.simulator.dof_vel - self.last_dof_vel)
                / self.sim_dt
            )
        elif control_type == "T":
            torques = pos_targets
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.config.domain_rand.randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0)
                * self.config.domain_rand.rfi_lim
                * self._rfi_lim_scale
                * self.torque_limits
            )

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques

    def _create_terrain(self):
        super()._create_terrain()

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        draw_env_ids = (self.push_robot_plot_counter < 10).nonzero(as_tuple=False).flatten()
        not_draw_env_ids = (self.push_robot_plot_counter >= 10).nonzero(as_tuple=False).flatten()
        self.record_push_robot_vel_buf[not_draw_env_ids] *= 0
        self.push_robot_plot_counter[not_draw_env_ids] = 0

        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat([push_vel, torch.zeros(1, device=self.device)])
            push_pos = self.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for push_vel, push_pos, push_mag, push_color, push_line_width in zip(
                push_vel_list, push_pos_list, push_mag_list, push_color_schems, push_line_widths
            ):
                for _ in range(200):
                    self.simulator.draw_line(
                        Point(push_pos + torch.rand(3, device=self.device) * push_line_width),
                        Point(push_pos + push_vel * push_mag),
                        Point(push_color),
                        env_id,
                    )

        self.simulator.clear_lines()
        self._refresh_sim_tensors()

    ################ Curriculum #################

    def _update_average_episode_length(self, env_ids):
        if not self.is_evaluating:
            num = len(env_ids)
            current_average_episode_length = torch.mean(
                self.last_episode_length_buf[env_ids], dtype=torch.float
            )

            self.average_episode_length = self.average_episode_length * (
                1 - num / self.num_compute_average_epl
            ) + current_average_episode_length * (num / self.num_compute_average_epl)

    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= 1 - self.config.rewards.reward_penalty_degree
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= 1 + self.config.rewards.reward_penalty_degree

        self.reward_penalty_scale = torch.clip(
            self.reward_penalty_scale,
            self.config.rewards.reward_min_penalty_scale,
            self.config.rewards.reward_max_penalty_scale,
        )

    def _update_reward_limits_curriculum(self):
        """
        Update the reward limits curriculum based on the average episode length.
        """
        if self.use_reward_limits_dof_pos_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold
            ):
                self.soft_dof_pos_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold
            ):
                self.soft_dof_pos_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree
                )

            self.soft_dof_pos_curriculum_value = torch.clip(
                self.soft_dof_pos_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit,
            )

        if self.use_reward_limits_dof_vel_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold
            ):
                self.soft_dof_vel_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold
            ):
                self.soft_dof_vel_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree
                )

            self.soft_dof_vel_curriculum_value = torch.clip(
                self.soft_dof_vel_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit,
            )

        if self.use_reward_limits_torque_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold
            ):
                self.soft_torque_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold
            ):
                self.soft_torque_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree
                )

            self.soft_torque_curriculum_value = torch.clip(
                self.soft_torque_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit,
            )

    # ------------ reward functions----------------
    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_undesired_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        undesired_contact = torch.any(
            torch.abs(self.simulator.contact_forces[:, self.penalised_contact_indices, :]) > 1,
            dim=(1, 2),
        )
        # Penalize undesired contact
        # import ipdb; ipdb.set_trace()
        res[undesired_contact] = 1.0
        # print(res)
        return res

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_penalty_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)

    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1
        )

    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(self.last_actions[:, : self.num_dof] - self.actions[:, : self.num_dof]),
            dim=1,
        )

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    ######################## LIMITS REWARDS #########################

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
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if self.use_reward_limits_dof_vel_curriculum:
            return torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits * self.soft_dof_vel_curriculum_value
                ).clip(min=0.0, max=1.0),
                dim=1,
            )
        else:
            return torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

    def _reward_limits_torque(self):
        # penalize torques too close to the limit

        # if torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit).clip(min=0.), dim=1) > 0:
        #     print("torque limits violation", torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit).clip(min=0.), dim=1))

        if self.use_reward_limits_torque_curriculum:
            return torch.sum(
                (
                    torch.abs(self.torques) - self.torque_limits * self.soft_torque_curriculum_value
                ).clip(min=0.0, max=1.0),
                dim=1,
            )
        else:
            return torch.sum(
                (
                    torch.abs(self.torques)
                    - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit
                ).clip(min=0.0),
                dim=1,
            )

    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_horizontal_vel = self.simulator._rigid_body_vel[:, self.feet_indices, :2]
        return torch.sum(
            torch.norm(foot_horizontal_vel, dim=-1)
            * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0),
            dim=1,
        )

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Jiawei: Key ingredient.
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt
        self.feet_air_max_height = torch.max(
            self.feet_air_max_height, self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        )

        rew_feet_max_height = torch.sum(
            (
                torch.clamp_min(
                    self.config.rewards.desired_feet_max_height_for_this_air
                    - self.feet_air_max_height,
                    0,
                )
            )
            * from_air_to_contact,
            dim=1,
        )  # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])

        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))

        return heading_diff_left + heading_diff_right

    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return (
            torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
            + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
        )

    def _episodic_domain_randomization(self, env_ids):
        """Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.kp_range[0],
                self.config.domain_rand.kp_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )
            self._kd_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.kd_range[0],
                self.config.domain_rand.kd_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )

        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.rfi_lim_range[0],
                self.config.domain_rand.rfi_lim_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )

        if self.config.domain_rand.randomize_ctrl_delay:
            # self.action_queue[env_ids] = 0.delay:
            self.action_queue[env_ids] *= 0.0
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(
                self.config.domain_rand.ctrl_delay_step_range[0],
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )

    def _push_robots(self, env_ids=None):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        if self.config.simulator.config.name == "isaacsim":
            self.simulator.trigger_on_push_by_setting_velocity_events(env_ids)
        else:
            self.need_to_refresh_envs[env_ids] = (
                True  # This function asks the simulator to refresh the envs that has been pushed.
            )
            max_vel = self.config.domain_rand.max_push_vel_xy
            self.push_robot_vel_buf[env_ids] = torch_rand_float(
                -max_vel, max_vel, (len(env_ids), 2), device=str(self.device)
            )  # lin vel x/y
            self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
            self.simulator.robot_root_states[env_ids, 7:9] += self.push_robot_vel_buf[env_ids]
            if self.config.domain_rand.get("max_push_ang_vel", 0.0) > 0.0:
                max_ang_vel = self.config.domain_rand.max_push_ang_vel
                push_ang_vel = torch_rand_float(
                    -max_ang_vel, max_ang_vel, (len(env_ids), 3), device=str(self.device)
                )
                self.simulator.robot_root_states[env_ids, 10:13] += push_ang_vel

    def push_all_robots(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        if self.config.simulator.config.name == "isaacsim":
            self.simulator.trigger_on_push_by_setting_velocity_events(env_ids)
        else:
            max_vel = self.config.domain_rand.max_push_vel_xy
            self.push_robot_vel_buf[env_ids] = torch_rand_float(
                -max_vel, max_vel, (len(env_ids), 2), device=str(self.device)
            )  # lin vel x/y
            self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
            self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
            self.simulator.set_actor_root_state_tensor(
                torch.arange(self.num_envs, device=self.device), self.simulator.robot_root_states
            )

    def push_all_robots_by_force(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        if self.config.simulator.config.name == "isaacsim":
            self.simulator.trigger_on_push_by_force_torque_events(env_ids)
        else:
            raise NotImplementedError

    ############ TERRAIN AND COMMANDS

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            raise NotImplementedError(
                "Target state is no longer supported for legged robot base task"
            )
        else:
            self.target_robot_dof_state[env_ids, :, 0] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device)
            )
            self.target_robot_dof_state[env_ids, :, 1] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.target_robot_root_states[env_ids] = target_root_states
            self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            # base position
            if self.custom_origins:
                self.target_robot_root_states[env_ids] = self.base_init_state
                self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]
                self.target_robot_root_states[env_ids, :2] += torch_rand_float(
                    -1.0, 1.0, (len(env_ids), 2), device=str(self.device)
                )  # xy position within 1m of the center
            else:
                self.target_robot_root_states[env_ids] = self.base_init_state
                self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities

            self.target_robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel

    def _plot_domain_rand_params(self):
        raise NotImplementedError

    ######################### Observations #########################
    def _get_obs_base_pos_z(
        self,
    ):
        return self.simulator.robot_root_states[:, 2:3]

    def _get_obs_feet_contact_force(
        self,
    ):
        return self.simulator.contact_forces[:, self.feet_indices, :].view(self.num_envs, -1)

    def _get_obs_base_lin_vel(
        self,
    ):
        return self.base_lin_vel

    def _get_obs_base_ang_vel(
        self,
    ):
        return self.base_ang_vel

    def _get_obs_projected_gravity(
        self,
    ):
        return self.projected_gravity

    def _get_obs_dof_pos(
        self,
    ):
        return self.simulator.dof_pos - self.default_dof_pos

    def _get_obs_dof_vel(
        self,
    ):
        return self.simulator.dof_vel

    def _get_obs_history(
        self,
    ):
        assert "history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_short_history(
        self,
    ):
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["short_history"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_long_history(
        self,
    ):
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["long_history"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    ######################### Observations #########################
    def _get_obs_history_actor(
        self,
    ):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_actor"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_teacher_actor(
        self,
    ):
        assert "history_teacher_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_teacher_actor"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_critic(
        self,
    ):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_critic"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    ###############################################################

    def _get_obs_actions(
        self,
    ):
        return self.actions

    def _get_obs_rgb_image(self):
        """Get RGB image from the ego camera.
        Returns:
            torch.Tensor: Flattened RGB image tensor of shape [batch_size, width*height*3]
        """
        if hasattr(self.simulator, "ego_camera") and self.simulator.ego_camera is not None:
            # Get RGB image from simulator - shape is typically [batch_size, height, width, 3]
            rgb_image = self.simulator.get_rgb_image()
            # ----------- Image Augmentation (see domain_rand.image_augmentation) -----------

            rgb_image = self._image_augmentation(rgb_image)

            # ----------------------- Uncomment to show rgb image -----------------------
            if self.config.get("show_rgb_image", False):
                # only show the first env's rgb image
                # import ipdb; ipdb.set_trace()
                rgb_image_np = rgb_image.cpu().numpy()  # RGB to BGR for OpenCV
                image_mean = np.array(self.config.simulator.config.cameras.image_mean)
                image_std = np.array(self.config.simulator.config.cameras.image_std)
                rgb_image_np = (rgb_image_np * image_std) + image_mean
                rgb_image_np = (rgb_image_np * 255).astype(
                    np.uint8
                )  # Ensure the image is in uint8 format
                rgb_image_np = rgb_image_np[self.visualize_env_id, :, :, ::-1]
                import cv2

                cv2.imshow(f"rgb_image_in_env_{int(self.visualize_env_id)}", rgb_image_np)

                # if self.episode_length_buf[0] % 100 == 0:
                #     # save the image by the name and episode_length_buf
                #     filename = f"configuration_2_frame_{self.episode_length_buf[0]:06d}_v2.png"
                #     from pathlib import Path
                #     filepath = Path(self.config.save_rendering_dir) / filename
                #     os.makedirs(self.config.save_rendering_dir, exist_ok=True)
                #     cv2.imwrite(str(filepath), rgb_image_np)
                #     import ipdb; ipdb.set_trace()
                cv2.waitKey(1)

                # Save the image to experiment folder if ckpt_dir is available
                if self.config.get("save_rgb_image", False):
                    # Create frames directory in the experiment folder (only once)
                    if not self._frames_dir_created:
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%m%d_%H%M%S")
                        ckpt_dir_path = Path(self.config.ckpt_dir)
                        self._frames_dir = ckpt_dir_path / f"frames_{timestamp}"
                        self._frames_dir.mkdir(exist_ok=True, parents=True)
                        self._frames_dir_created = True

                    # Generate filename with step counter
                    filename = f"frame_{self._frame_counter:06d}.png"
                    filepath = self._frames_dir / filename

                    # Convert back to RGB for saving (OpenCV uses BGR)
                    rgb_image_save = rgb_image_np
                    cv2.imwrite(str(filepath), rgb_image_save)

                    self._frame_counter += 1
            # ----------------------- Uncomment to show rgb image -----------------------

            # rbg_image_after_augmentation = self.

            return rgb_image
        else:
            # Return zero tensor if camera is not enabled
            camera_resolution = self.config.simulator.config.cameras.camera_resolutions
            # Shape needs to be [batch_size, width*height*3]
            zero_image = torch.zeros(
                (self.num_envs, camera_resolution[0] * camera_resolution[1] * 3),
                device=self.device,
                dtype=torch.float,
            )
            return zero_image

    def _get_obs_rgb_image_history(self):
        """Get RGB image history from the history handler.
        Returns:
            torch.Tensor: RGB image history tensor of shape [batch_size, history_length, height, width, 3]
        """
        assert "rgb_image_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["rgb_image_history"]
        history_tensors = []

        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            # Query history from history_handler - shape: [batch_size, history_length, height, width, 3]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensors.append(history_tensor)

        # Concatenate along the history dimension if multiple keys
        if len(history_tensors) == 1:
            return history_tensors[0]
        else:
            return torch.cat(history_tensors, dim=1)

    def _get_obs_state_history(
        self,
    ):
        assert "state_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["state_history"]
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]  # Shape: [4096, history_length, obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=-1)

    def _image_augmentation(self, rgb_image):
        if self.config.domain_rand.image_augmentation.enabled:
            # import ipdb; ipdb.set_trace()
            image_mean = torch.tensor(
                self.config.simulator.config.cameras.image_mean, device=self.device
            )
            image_std = torch.tensor(
                self.config.simulator.config.cameras.image_std, device=self.device
            )
            # denormalize the image
            rgb_image = rgb_image * image_std + image_mean
            # rgb is (1, 60, 60, 3)
            # perturb it to be (1, 3, 60, 60)
            rgb_image = rgb_image.permute(0, 3, 1, 2)

            if self.config.domain_rand.image_augmentation.brightness.enabled:
                if (
                    torch.rand(1)
                    < self.config.domain_rand.image_augmentation.brightness.probability
                ):
                    # logger.info(f"----- Image Augmentation: Brightness -----")
                    brightness_factor_float = torch_rand_float(
                        self.config.domain_rand.image_augmentation.brightness.range[0],
                        self.config.domain_rand.image_augmentation.brightness.range[1],
                        (1, 1),
                        device=self.device,
                    )
                    rgb_image = v2.functional.adjust_brightness(
                        rgb_image, brightness_factor=brightness_factor_float
                    )

            # saturation
            if self.config.domain_rand.image_augmentation.saturation.enabled:
                if (
                    torch.rand(1)
                    < self.config.domain_rand.image_augmentation.saturation.probability
                ):
                    # logger.info(f"----- Image Augmentation: Saturation -----")
                    saturation_factor_float = torch_rand_float(
                        self.config.domain_rand.image_augmentation.saturation.range[0],
                        self.config.domain_rand.image_augmentation.saturation.range[1],
                        (1, 1),
                        device=self.device,
                    )
                    rgb_image = v2.functional.adjust_saturation(
                        rgb_image, saturation_factor=saturation_factor_float
                    )

            # hue
            if self.config.domain_rand.image_augmentation.hue.enabled:
                if torch.rand(1) < self.config.domain_rand.image_augmentation.hue.probability:
                    # logger.info(f"----- Image Augmentation: Hue -----")
                    hue_factor_float = torch_rand_float(
                        self.config.domain_rand.image_augmentation.hue.range[0],
                        self.config.domain_rand.image_augmentation.hue.range[1],
                        (1, 1),
                        device=self.device,
                    )
                    rgb_image = v2.functional.adjust_hue(rgb_image, hue_factor=hue_factor_float)

            # contrast
            if self.config.domain_rand.image_augmentation.contrast.enabled:
                if torch.rand(1) < self.config.domain_rand.image_augmentation.contrast.probability:
                    # logger.info(f"----- Image Augmentation: Contrast -----")
                    contrast_factor_float = torch_rand_float(
                        self.config.domain_rand.image_augmentation.contrast.range[0],
                        self.config.domain_rand.image_augmentation.contrast.range[1],
                        (1, 1),
                        device=self.device,
                    )
                    rgb_image = v2.functional.adjust_contrast(
                        rgb_image, contrast_factor=contrast_factor_float
                    )

            # gaussian noise
            if self.config.domain_rand.image_augmentation.gaussian_noise.enabled:
                if (
                    torch.rand(1)
                    < self.config.domain_rand.image_augmentation.gaussian_noise.probability
                ):
                    # logger.info(f"----- Image Augmentation: Gaussian Noise -----")
                    noise_std_float = torch_rand_float(
                        self.config.domain_rand.image_augmentation.gaussian_noise.std_range[0],
                        self.config.domain_rand.image_augmentation.gaussian_noise.std_range[1],
                        (1, 1),
                        device=self.device,
                    )
                    rgb_image = v2.functional.gaussian_noise(
                        rgb_image, sigma=noise_std_float, clip=True
                    )

            # gaussian blur
            if self.config.domain_rand.image_augmentation.gaussian_blur.enabled:
                if (
                    torch.rand(1)
                    < self.config.domain_rand.image_augmentation.gaussian_blur.probability
                ):
                    # logger.info(f"----- Image Augmentation: Gaussian Blur -----")
                    rgb_image = v2.functional.gaussian_blur(
                        rgb_image,
                        kernel_size=(
                            self.config.domain_rand.image_augmentation.gaussian_blur.kernel_size_range[
                                0
                            ],
                            self.config.domain_rand.image_augmentation.gaussian_blur.kernel_size_range[
                                1
                            ],
                        ),
                        sigma=(
                            self.config.domain_rand.image_augmentation.gaussian_blur.sigma_range[0],
                            self.config.domain_rand.image_augmentation.gaussian_blur.sigma_range[1],
                        ),
                    )

            # perturb it back to be (1, 60, 60, 3)
            rgb_image = rgb_image.permute(0, 2, 3, 1)

            # normalize the image
            rgb_image = (rgb_image - image_mean) / image_std

        return rgb_image

    def _depth_augmentation(self, depth_image):
        """
        Apply depth augmentation techniques to simulate various depth sensor conditions.

        Args:
            depth_image (torch.Tensor): Input depth image tensor of shape [batch_size, height, width, channels]
                                       channels can be 1 or 3 (when stacked)

        Returns:
            torch.Tensor: Augmented depth image tensor
        """
        if not self.config.domain_rand.depth_augmentation.enabled:
            return depth_image

        # Clone the input to avoid modifying the original
        augmented_depth = depth_image.clone()

        # Handle both single channel and stacked (3-channel) depth images
        # If it's stacked (3 channels), we only augment the first channel and replicate to others
        batch_size, h, w, c = augmented_depth.shape
        is_stacked = c == 3

        if is_stacked:
            # Extract first channel for augmentation
            depth_single = augmented_depth[:, :, :, 0:1]  # Keep dimension
        else:
            depth_single = augmented_depth

        # 1. Depth value scaling/normalization
        if (
            self.config.domain_rand.depth_augmentation.depth_scaling.enabled
            and torch.rand(1) < self.config.domain_rand.depth_augmentation.depth_scaling.probability
        ):

            scale_factor = (
                torch_rand_float(
                    self.config.domain_rand.depth_augmentation.depth_scaling.scale_range[0],
                    self.config.domain_rand.depth_augmentation.depth_scaling.scale_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # Reshape from (num_envs, 1) to (num_envs, 1, 1, 1)
            depth_single = depth_single * scale_factor

        # 2. Depth noise injection
        if (
            self.config.domain_rand.depth_augmentation.depth_noise.enabled
            and torch.rand(1) < self.config.domain_rand.depth_augmentation.depth_noise.probability
        ):

            # Gaussian noise
            if self.config.domain_rand.depth_augmentation.depth_noise.gaussian_noise.enabled:
                noise_std = torch_rand_float(
                    self.config.domain_rand.depth_augmentation.depth_noise.gaussian_noise.std_range[
                        0
                    ],
                    self.config.domain_rand.depth_augmentation.depth_noise.gaussian_noise.std_range[
                        1
                    ],
                    (1, 1),
                    device=self.device,
                )
                gaussian_noise = torch.randn_like(depth_single) * noise_std
                depth_single = depth_single + gaussian_noise

            # Perlin noise (simplified implementation using scaled random noise)
            if self.config.domain_rand.depth_augmentation.depth_noise.perlin_noise.enabled:
                amplitude = torch_rand_float(
                    self.config.domain_rand.depth_augmentation.depth_noise.perlin_noise.amplitude_range[
                        0
                    ],
                    self.config.domain_rand.depth_augmentation.depth_noise.perlin_noise.amplitude_range[
                        1
                    ],
                    (1, 1),
                    device=self.device,
                )
                # Create structured noise by downsampling and upsampling random noise
                low_res_noise = torch.randn(self.num_envs, h // 4, w // 4, 1, device=self.device)
                # Upsample using nearest neighbor interpolation
                perlin_noise = F.interpolate(
                    low_res_noise.permute(0, 3, 1, 2), size=(h, w), mode="nearest"
                ).permute(0, 2, 3, 1)
                depth_single = depth_single + perlin_noise * amplitude

        # 3. Dropout/masking (simulate missing data)
        if (
            self.config.domain_rand.depth_augmentation.depth_dropout.enabled
            and torch.rand(1) < self.config.domain_rand.depth_augmentation.depth_dropout.probability
        ):

            # Random patches dropout
            if self.config.domain_rand.depth_augmentation.depth_dropout.patch_dropout.enabled:
                for env_idx in range(self.num_envs):
                    num_patches = torch.randint(
                        self.config.domain_rand.depth_augmentation.depth_dropout.patch_dropout.num_patches_range[
                            0
                        ],
                        self.config.domain_rand.depth_augmentation.depth_dropout.patch_dropout.num_patches_range[
                            1
                        ]
                        + 1,
                        (1,),
                    ).item()

                    for _ in range(num_patches):
                        patch_size = torch.randint(
                            self.config.domain_rand.depth_augmentation.depth_dropout.patch_dropout.patch_size_range[
                                0
                            ],
                            self.config.domain_rand.depth_augmentation.depth_dropout.patch_dropout.patch_size_range[
                                1
                            ]
                            + 1,
                            (1,),
                        ).item()

                        # Random patch location
                        start_y = torch.randint(0, max(1, h - patch_size), (1,)).item()
                        start_x = torch.randint(0, max(1, w - patch_size), (1,)).item()
                        end_y = min(start_y + patch_size, h)
                        end_x = min(start_x + patch_size, w)

                        # Set patch to zero (missing data)
                        depth_single[env_idx, start_y:end_y, start_x:end_x, :] = 0.0

            # Random pixel dropout
            if self.config.domain_rand.depth_augmentation.depth_dropout.pixel_dropout.enabled:
                dropout_rate = torch_rand_float(
                    self.config.domain_rand.depth_augmentation.depth_dropout.pixel_dropout.dropout_rate_range[
                        0
                    ],
                    self.config.domain_rand.depth_augmentation.depth_dropout.pixel_dropout.dropout_rate_range[
                        1
                    ],
                    (1, 1),
                    device=self.device,
                )
                dropout_mask = torch.rand_like(depth_single) < dropout_rate
                depth_single = torch.where(
                    dropout_mask, torch.zeros_like(depth_single), depth_single
                )

        # 4. Quantization (simulate lower resolution depth sensors)
        if (
            self.config.domain_rand.depth_augmentation.depth_quantization.enabled
            and torch.rand(1)
            < self.config.domain_rand.depth_augmentation.depth_quantization.probability
        ):

            num_levels = torch.randint(
                self.config.domain_rand.depth_augmentation.depth_quantization.levels_range[0],
                self.config.domain_rand.depth_augmentation.depth_quantization.levels_range[1] + 1,
                (1,),
            ).item()

            # Find min and max depth values (excluding zeros which represent missing data)
            valid_depth_mask = depth_single > 0
            if valid_depth_mask.any():
                min_depth = depth_single[valid_depth_mask].min()
                max_depth = depth_single[valid_depth_mask].max()

                if max_depth > min_depth:
                    # Quantize depth values
                    depth_range = max_depth - min_depth
                    quantized_depth = (
                        torch.round((depth_single - min_depth) / depth_range * (num_levels - 1))
                        / (num_levels - 1)
                        * depth_range
                        + min_depth
                    )
                    # Keep zeros as zeros (missing data)
                    depth_single = torch.where(valid_depth_mask, quantized_depth, depth_single)

        # Ensure non-negative depth values (depth cannot be negative)
        depth_single = torch.clamp(depth_single, min=0.0)

        # If original was stacked (3 channels), replicate the augmented depth to all channels
        if is_stacked:
            augmented_depth = torch.cat([depth_single, depth_single, depth_single], dim=-1)
        else:
            augmented_depth = depth_single

        return augmented_depth

    def _get_obs_depth_image(self):
        depth_image = self.simulator.get_depth_image()

        # Apply depth augmentation
        depth_image = self._depth_augmentation(depth_image)

        if self.config.get("show_rgb_image", False):
            # only show the first env's depth image
            depth_image_np = depth_image.cpu().numpy()
            depth_image_np = depth_image_np[self.visualize_env_id]

            # Since depth_image is (height, width, 3), extract the first channel for depth processing
            if len(depth_image_np.shape) == 3:
                depth_image_np = depth_image_np[:, :, 0]  # Use first channel as depth

            # Normalize depth for visualization (0-1)
            depth_min = depth_image_np.min()
            depth_max = depth_image_np.max()
            if depth_max > depth_min:
                depth_normalized = (depth_image_np - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth_image_np)

            # Ensure depth_normalized is float64 for matplotlib
            depth_normalized = depth_normalized.astype(np.float64)

            # Apply colormap to create colored depth visualization
            import matplotlib.pyplot as plt

            depth_colored = plt.cm.viridis(depth_normalized)  # Use viridis colormap

            # Convert to BGR format for OpenCV and ensure uint8 type
            depth_image_np = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            depth_image_np = depth_image_np[:, :, ::-1]

            # Ensure the array is contiguous in memory
            depth_image_np = np.ascontiguousarray(depth_image_np)

            import cv2

            cv2.imshow(f"depth_image_in_env_{int(self.visualize_env_id)}", depth_image_np)
            cv2.waitKey(1)

        return depth_image

    def _load_z_model(
        self,
    ):
        from pathlib import Path

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from gr00t.rl.utils.helpers import pre_process_config

        check_points_path = self.config.z_model_config.check_points
        config_path = Path(check_points_path).parent / "config.yaml"
        checkpoint = torch.load(check_points_path, map_location=self.device, weights_only=False)
        with open(config_path) as file:
            self.z_config = OmegaConf.load(file)
            pre_process_config(self.z_config)

        if getattr(self.z_config.algo.config, "use_new_actor_critic", False):
            policy = instantiate(
                self.z_config.algo.config.actor,
                env_config=self.z_config.env.config,
                algo_config=self.z_config.algo.config,
                module_dim_dict={},
                _recursive_=False,
            ).to(self.device)
            policy.eval()
            self.z_actor = policy
            self.z_actor.load_state_dict(checkpoint["policy_state_dict"])
        else:
            raise NotImplementedError

    def _compute_z_prior(self):
        teacher_obs_dict_raw = TensorDict()
        self.z_obs_dict = TensorDict()
        noise_curriculum_value = 0
        for obs_key, obs_config in self.z_config.obs.obs_dict.items():
            if obs_key == "proprioception_obs":
                teacher_obs_dict_raw[obs_key] = dict()
                parse_observation(
                    self,
                    obs_config,
                    teacher_obs_dict_raw[obs_key],
                    self.z_config.obs.obs_scales,
                    self.z_config.obs.noise_scales,
                    noise_curriculum_value,
                )

        for key in teacher_obs_dict_raw[
            "proprioception_obs"
        ].keys():  # prefer to use the same noised obs for z_prior as buf_dict_raw
            if key in self.obs_buf_dict_raw["actor_obs"]:
                teacher_obs_dict_raw["proprioception_obs"][key] = self.obs_buf_dict_raw[
                    "actor_obs"
                ][key]

        for obs_key, obs_config in self.z_config.obs.obs_dict.items():
            if obs_key == "proprioception_obs":
                obs_keys = sorted(obs_config)
                # print("obs_keys", obs_keys)
                self.z_obs_dict[obs_key] = torch.cat(
                    [teacher_obs_dict_raw[obs_key][key] for key in obs_keys], dim=-1
                )

        with torch.no_grad():
            self.prior_mu, self.prior_sigma = self.z_actor.eval_prior(obs_dict=self.z_obs_dict)

    def get_z_action(self, actions_z):
        # ZL: Due to the history_actor's update schedule, will have to first compute the prior in compute_observations then update it. Be very careful.
        if "prior_mu" not in self.__dict__:
            # ZL: First step will have to compute the prior
            self._compute_z_prior()

        with torch.no_grad():

            ############################## Debugging #################################
            # print(teacher_obs_dict['proprioception_obs'].max(), teacher_obs_dict['proprioception_obs'].min())
            # std = torch.exp(0.5 * self.prior_sigma)
            # actions_z = self.prior_mu + std * torch.randn_like(self.prior_mu); # print(prior_mu.max(), prior_mu.min())
            # actions_z = self.prior_mu
            ############################## Debugging #################################

            actions_z = actions_z + self.prior_mu

            actions_mean = self.z_actor.eval_motion_decoder(actions_z, obs_dict=self.z_obs_dict)
        return actions_mean

    def init_eval_metrics_tracking(self, device):
        self.eval_metrics = {
            "episode_goal_reached": torch.zeros(self.num_envs, dtype=torch.bool, device=device),
            "goal_reached_buffer": [torch.zeros(self.num_envs, dtype=torch.bool, device=device)],
        }

    def update_eval_metrics_per_step(self, infos):
        pass

    def process_eval_episode_completions(
        self, completed_env_ids, cur_reward_sum, cur_episode_length
    ):
        pass

    def reset_eval_episode_tracking(self, completed_env_ids):
        pass

    def get_eval_metrics_summary(self):
        return self.eval_metrics

    @staticmethod
    def _tracking_reward_util(
        reward: torch.Tensor, std: float, target: float, scale: float = 1.0, offset: float = 0.0
    ):
        return scale * torch.exp(-((reward - target) ** 2) / (2 * std**2)) + offset
