# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Optional

import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class StagedTaskBase(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.max_stage_time = torch.tensor(
            self.config.max_stage_time, dtype=torch.long, device=self.device
        )  # max time allowed in each stage
        self.total_stage_time = torch.sum(self.max_stage_time).item()
        self.stage_reward_scale = torch.tensor(
            self.config.stage_reward_scale, dtype=torch.float, device=self.device
        )
        self.num_stages = len(self.max_stage_time)
        assert torch.all(self.max_stage_time > 0), "max_stage_time must be greater than 0"
        assert (
            len(self.stage_reward_scale) == self.num_stages
        ), "stage_reward_scale must have the same length as max_stage_time"

        self.stage_reward_cond_funcs: list[Callable] = []
        self.stage_advance_cond_funcs: list[Callable] = []
        self.stage_advance_callback_funcs: list[Optional[Callable]] = []
        self.stage_complete_cond_func: Callable | None = None

        for i in range(self.num_stages):
            func_name = f"_stage_{i}_reward_condition"
            if hasattr(self, func_name):
                self.stage_reward_cond_funcs.append(getattr(self, func_name))
            else:
                raise ValueError(f"Stage {i} reward condition function {func_name} not found")

        for i in range(self.num_stages - 1):
            func_name = f"_stage_{i}_to_{i+1}_advance_condition"
            if hasattr(self, func_name):
                self.stage_advance_cond_funcs.append(getattr(self, func_name))
            else:
                raise ValueError(
                    f"Stage {i} to {i+1} advance condition function {func_name} not found"
                )
            callback_func_name = f"_stage_{i}_to_{i+1}_advance_callback"
            if hasattr(self, callback_func_name):
                self.stage_advance_callback_funcs.append(getattr(self, callback_func_name))
            else:
                self.stage_advance_callback_funcs.append(None)

        func_name = f"_stage_{self.num_stages - 1}_to_complete_condition"
        if hasattr(self, func_name):
            self.stage_complete_cond_func = getattr(self, func_name)
        else:
            raise ValueError(
                f"Stage {self.num_stages - 1} to complete condition function {func_name} not found"
            )

        self.log_dict["average_stage_reached"] = 0.0
        self.log_dict["average_goal_reached"] = 0.0
        self.log_dict["average_last_stage_goal_reached"] = 0.0

        if self.config.get("termination_curriculum", None) is not None:
            self.termination_level = torch.tensor(
                self.config.termination_curriculum.initial_level, device=self.device
            )
            self.log_dict["termination_level"] = self.termination_level
        else:
            self.termination_level = torch.tensor(1.0, device=self.device)

        # staged reset
        self.staged_reset_buf = {}
        self.staged_reset_num_samples = None
        self.enable_staged_reset = self.config.get("enable_staged_reset", False)
        self.staged_reset_ratios = self.config.get("staged_reset_ratios", None)
        if self.staged_reset_ratios is not None:
            self.staged_reset_ratios = torch.tensor(self.staged_reset_ratios, device=self.device)
        self.staged_reset_max_samples_per_stage = self.config.get(
            "staged_reset_max_samples_per_stage", None
        )

        if self.enable_staged_reset:
            assert (
                self.staged_reset_ratios is not None
            ), "staged_reset_ratios must be set if staged_reset is enabled"
            assert (
                len(self.staged_reset_ratios) == self.num_stages
            ), "staged_reset_ratios must have the same length as max_stage_time"
            assert torch.all(
                self.staged_reset_ratios >= 0
            ), "staged_reset_ratios must be non-negative"
            assert self.staged_reset_ratios[0] > 0, "staged_reset_ratios[0] must be greater than 0"
            assert (
                self.staged_reset_max_samples_per_stage is not None
            ), "staged_reset_max_samples_per_stage must be set if staged_reset is enabled"
            assert (
                self.staged_reset_max_samples_per_stage > 0
            ), "staged_reset_max_samples_per_stage must be greater than 0"
            self.staged_reset_num_samples = torch.zeros(
                self.num_stages, self.num_envs, device=self.device, dtype=torch.long
            )
            # stage 0 can always be resetted to, and is provided by the user.
            self.staged_reset_num_samples[0, :] = self.staged_reset_max_samples_per_stage
        # track robot state by default
        if self.enable_staged_reset:
            self._register_task_state_to_track(self.simulator._robot, "robot")
            self._register_buffer_to_track(
                "total_time_buf",
                (self.num_envs,),
                self.total_time_store_callback,
                self.total_time_load_callback,
                dtype=torch.long,
            )

    def _init_buffers(self):
        super()._init_buffers()
        self.stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # current stage index
        self.total_time_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # total time in the task
        self.just_resetted_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env just resetted
        self.just_completed_task_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env just completed the task
        self.time_in_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # time in current stage
        self.actual_time_in_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # actual time in current stage
        self.time_since_completion_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # time since the task was last completed in the current episode
        self.last_completed_task_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env has completed the task in the past episode
        self.last_max_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # max stage index of the past episode
        self.current_completed_task_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env has completed the task in the current episode
        self.current_max_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # max stage index of the current episode
        self.last_last_stage_completed_task_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env has completed the last stage in the past episode
        self.current_last_stage_completed_task_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether the env has completed the last stage in the current episode

    def _post_compute_observations_callback(self):
        # advance stage
        advance_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_stages - 1):
            advance_cond = self.stage_advance_cond_funcs[i]()
            assert advance_cond.shape == (self.num_envs,)
            assert advance_cond.dtype == torch.bool
            advance_mask |= advance_cond & self._make_mask(i)
        advance_mask &= ~self.just_resetted_buf  # avoid advancing on reset
        self.just_resetted_buf[:] = False  # reset the flag
        if self.config.get("award_remaining_time_on_advance", False):
            self.time_in_stage_buf[advance_mask] -= self.max_stage_time[
                self.stage_buf[advance_mask]
            ]
        else:
            self.time_in_stage_buf[advance_mask] = 0
        self.actual_time_in_stage_buf[advance_mask] = 0
        self.stage_buf[advance_mask] += 1
        self.time_in_stage_buf[~advance_mask] += 1
        self.actual_time_in_stage_buf[~advance_mask] += 1
        self.total_time_buf[:] += 1
        self.current_max_stage_buf[advance_mask] = torch.maximum(
            self.current_max_stage_buf[advance_mask], self.stage_buf[advance_mask]
        )

        # trigger stage advance callbacks
        for i in range(self.num_stages - 1):
            if self.stage_advance_callback_funcs[i] is not None:
                env_ids = torch.where(advance_mask & (self.stage_buf == (i + 1)))[0]
                if len(env_ids) > 0:
                    self.stage_advance_callback_funcs[i](env_ids)

        # reset just_completed_task_buf
        self.just_completed_task_buf[:] = False

        # take a snapshot of the buffered states
        if self.enable_staged_reset:
            self._take_snapshot_of_buffered_states(advance_mask)

        return super()._post_compute_observations_callback()

    @override
    def _check_termination(self):
        super()._check_termination()
        is_overtime = self.time_in_stage_buf >= self.max_stage_time[self.stage_buf]
        is_last_stage = self._make_mask(self.num_stages - 1)
        is_complete = self.stage_complete_cond_func() & (self.episode_length_buf >= 2)
        assert is_complete.shape == (self.num_envs,)
        assert is_complete.dtype == torch.bool

        is_overtime &= ~(is_last_stage & is_complete)

        if self.config.get("reset_on_overtime", True):
            self.reset_buf |= is_overtime
        self.current_completed_task_buf |= is_complete

        self.current_last_stage_completed_task_buf |= is_complete & is_last_stage

        if self.config.get("reset_on_complete", False):
            delay_time = self.config.get("reset_on_complete_delay", 0)
            if delay_time > 0:
                is_complete_and_delayed = is_complete & (
                    self.time_since_completion_buf >= delay_time
                )
                self.reset_buf |= is_complete_and_delayed
                self.time_out_buf |= is_complete_and_delayed
            else:
                self.reset_buf |= is_complete
                self.time_out_buf |= is_complete

        self.just_completed_task_buf |= is_complete

        self.time_since_completion_buf[self.time_since_completion_buf > 0] += 1
        first_complete = is_complete & (self.time_since_completion_buf == 0)
        self.time_since_completion_buf[first_complete] = 1

    @override
    def _reset_tasks_callback(self, env_ids: torch.Tensor):
        super()._reset_tasks_callback(env_ids)
        self.last_max_stage_buf[env_ids] = self.current_max_stage_buf[env_ids]
        self.current_max_stage_buf[env_ids] = 0
        self.last_completed_task_buf[env_ids] = self.current_completed_task_buf[env_ids]
        self.current_completed_task_buf[env_ids] = False
        self.last_last_stage_completed_task_buf[env_ids] = (
            self.current_last_stage_completed_task_buf[env_ids]
        )
        self.current_last_stage_completed_task_buf[env_ids] = False
        self.log_dict["average_stage_reached"] = self.last_max_stage_buf.float().mean()
        self.log_dict["average_goal_reached"] = self.last_completed_task_buf.float().mean()
        self.log_dict["average_last_stage_goal_reached"] = (
            self.last_last_stage_completed_task_buf.float().mean()
        )
        # Per-stage failure breakdown: of episodes that did NOT complete the task, which stage did
        # they get stuck at (their furthest stage reached). fail_at_stage_s summed over all s equals
        # the overall failure rate (== 1 - average_goal_reached); this shows WHERE failures
        # concentrate so we know which stage to fix to push the success rate up.
        failed = ~self.last_completed_task_buf
        for s in range(self.num_stages):
            self.log_dict[f"fail_at_stage_{s}"] = (
                (self.last_max_stage_buf == s) & failed
            ).float().mean()
        self._update_termination_curriculum()

    def _reward_stage(self):
        reward = torch.zeros(self.num_envs, self.num_stages, device=self.device, dtype=torch.bool)
        for i in range(self.num_stages):
            reward_cond = self.stage_reward_cond_funcs[i]()
            assert reward_cond.shape == (self.num_envs,)
            assert reward_cond.dtype == torch.bool
            this_stage_reward = reward_cond & (self.stage_buf == i)
            if self.config.get("no_reward_for_awarded_time", False):
                this_stage_reward &= self.time_in_stage_buf >= 0
            reward[:, i] = this_stage_reward
            if self.config.get("accumulate_stage_reward", True):
                reward[:, i] |= self.stage_buf > i
            elif self.config.get("no_reward_for_overtime", False):
                reward[:, i] &= self.time_in_stage_buf < self.max_stage_time[i]

        return (reward.float() * self.stage_reward_scale[None, :]).sum(dim=-1)

    def _reward_penalty_overtime(self):
        is_overtime = self.time_in_stage_buf >= self.max_stage_time[self.stage_buf]
        is_last_stage = self._make_mask(self.num_stages - 1)
        is_complete = self.stage_complete_cond_func() & (self.episode_length_buf >= 2)
        is_overtime &= ~(is_last_stage & is_complete)
        return is_overtime.float()

    def _reward_time_saving(self):
        return self.time_in_stage_buf < 0

    def _reward_transition(self):
        return (self.actual_time_in_stage_buf == 0) & (self.stage_buf > 0)

    def _reward_complete(self):
        return (self.time_since_completion_buf > 0).float()

    def _reward_success_save_time(self):
        reset_and_timeout = self.reset_buf & self.time_out_buf
        saved_time = self.total_stage_time - self.total_time_buf
        return (reset_and_timeout * saved_time).float()

    def _get_obs_stage(self):
        return F.one_hot(self.stage_buf, self.num_stages).float()

    def _get_obs_time_in_stage(self):
        return (
            self.time_in_stage_buf[:, None].float() / self.max_stage_time[self.stage_buf][:, None]
        )

    def _get_obs_total_time(self):
        return self.total_time_buf[:, None].float() / self.total_stage_time

    def _get_obs_actual_time_in_stage(self):
        return (
            self.actual_time_in_stage_buf[:, None].float()
            / self.max_stage_time[self.stage_buf][:, None]
        )

    def _get_obs_transition(self):
        return (self.actual_time_in_stage_buf == 0)[:, None].float()

    def _get_obs_complete(self):
        return (self.time_since_completion_buf > 0)[:, None].float()

    @classmethod
    def effective_in_stage(cls, stages: int | list[int]) -> Callable:
        """
        Wraps a reward function to only be effective in any of the given stages, and zero otherwise.
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(self, *args, **kwargs):
                reward = func(self, *args, **kwargs)
                assert isinstance(reward, torch.Tensor)
                assert reward.shape == (self.num_envs,)
                mask = self._make_mask(stages)
                return torch.where(mask, reward, torch.zeros_like(reward))

            return wrapper

        return decorator

    def _make_mask(self, stages: int | list[int] | torch.Tensor) -> torch.Tensor:
        """
        Make a mask of shape (num_envs,) that is True for envs in any of the stages.
        """
        if isinstance(stages, int):
            return self.stage_buf == stages
        elif isinstance(stages, list):
            return torch.any(
                self.stage_buf[:, None] == torch.tensor(stages, device=self.device)[None, :], dim=-1
            )
        elif isinstance(stages, torch.Tensor):
            assert (
                len(stages.shape) == 1
            ), "stages must be an int, a list of ints, or a tensor of shape (num_stages,)"
            return torch.any(self.stage_buf[:, None] == stages[None, :], dim=-1)
        else:
            raise ValueError(f"Invalid type for stages: {type(stages)}")

    def set_to_stage(self, env_ids: torch.Tensor, stage: torch.Tensor):
        assert len(env_ids) == len(stage) and len(env_ids.shape) == 1

        self.stage_buf[env_ids] = stage
        self.time_in_stage_buf[env_ids] = 0
        self.actual_time_in_stage_buf[env_ids] = 0
        self.just_resetted_buf[env_ids] = True
        self.time_since_completion_buf[env_ids] = 0
        self.current_max_stage_buf[env_ids] = stage
        self.total_time_buf[env_ids] = 0

    @override
    def _update_reward_penalty_curriculum(self):
        reward_penalty_level_down_ave_stage = self.config.rewards.get(
            "reward_penalty_level_down_ave_stage", None
        )
        reward_penalty_level_up_ave_stage = self.config.rewards.get(
            "reward_penalty_level_up_ave_stage", None
        )

        reward_penalty_level_down_ave_goal_reached_rate = self.config.rewards.get(
            "reward_penalty_level_down_ave_goal_reached_rate", None
        )
        reward_penalty_level_up_ave_goal_reached_rate = self.config.rewards.get(
            "reward_penalty_level_up_ave_goal_reached_rate", None
        )

        if (
            reward_penalty_level_down_ave_stage is not None
            and reward_penalty_level_down_ave_goal_reached_rate is not None
        ):
            raise ValueError(
                "reward_penalty_level_down_ave_stage and reward_penalty_level_down_ave_goal_reached_rate cannot be set at the same time"
            )

        if reward_penalty_level_down_ave_stage is not None:
            if self.log_dict["average_stage_reached"] < reward_penalty_level_down_ave_stage:
                self.reward_penalty_scale *= 1 - self.config.rewards.reward_penalty_degree
            elif self.log_dict["average_stage_reached"] > reward_penalty_level_up_ave_stage:
                self.reward_penalty_scale *= 1 + self.config.rewards.reward_penalty_degree
            self.reward_penalty_scale = torch.clip(
                self.reward_penalty_scale,
                self.config.rewards.reward_min_penalty_scale,
                self.config.rewards.reward_max_penalty_scale,
            )
        elif reward_penalty_level_down_ave_goal_reached_rate is not None:
            if (
                self.log_dict["average_goal_reached"]
                < reward_penalty_level_down_ave_goal_reached_rate
            ):
                self.reward_penalty_scale *= 1 - self.config.rewards.reward_penalty_degree
            elif (
                self.log_dict["average_goal_reached"]
                > reward_penalty_level_up_ave_goal_reached_rate
            ):
                self.reward_penalty_scale *= 1 + self.config.rewards.reward_penalty_degree
            self.reward_penalty_scale = torch.clip(
                self.reward_penalty_scale,
                self.config.rewards.reward_min_penalty_scale,
                self.config.rewards.reward_max_penalty_scale,
            )
        else:
            super()._update_reward_penalty_curriculum()

    def _update_termination_curriculum(self):
        # termination curriculum
        if self.config.get("termination_curriculum", None) is not None:
            if (
                self.log_dict["average_goal_reached"]
                < self.config.termination_curriculum.level_down_ave_goal_reached_rate
            ):
                # logger.warning(f"termination level down to {self.termination_level * (1 - self.config.termination_curriculum.degree)}")
                self.termination_level *= 1 - self.config.termination_curriculum.degree
            elif (
                self.log_dict["average_goal_reached"]
                > self.config.termination_curriculum.level_up_ave_goal_reached_rate
            ):
                # logger.warning(f"termination level up to {self.termination_level * (1 + self.config.termination_curriculum.degree)}")
                self.termination_level *= 1 + self.config.termination_curriculum.degree
            self.termination_level = torch.clip(
                self.termination_level,
                self.config.termination_curriculum.min_level,
                self.config.termination_curriculum.max_level,
            )

            self.log_dict["termination_level"] = self.termination_level

    def _register_task_state_to_track(self, obj: Articulation | RigidObject, name: str):
        """
        If staged_reset is enabled, the state of these task objects will be tracked and resetted together with the robot.
        """
        if not self.enable_staged_reset:
            return

        if self.config.simulator.config.name != "isaacsim":
            raise NotImplementedError("Staged reset is only supported in IsaacSim for now")

        if name in self.staged_reset_buf:
            raise ValueError(f"Duplicate task state name: {name}")

        assert isinstance(
            obj, (Articulation, RigidObject)
        ), "Only Articulation and RigidObject are supported for staged reset"

        self.staged_reset_buf[name] = {
            "obj": obj,
            "type": "articulation" if isinstance(obj, Articulation) else "rigid_object",
            "root_state": torch.zeros(
                self.num_stages,
                self.staged_reset_max_samples_per_stage,
                self.num_envs,
                13,
                device=self.device,
                dtype=torch.float,
                requires_grad=False,
            ),
        }

        if isinstance(obj, Articulation):
            self.staged_reset_buf[name]["dof_state"] = torch.zeros(
                self.num_stages,
                self.staged_reset_max_samples_per_stage,
                self.num_envs,
                obj.num_joints,
                2,
                device=self.device,
                dtype=torch.float,
                requires_grad=False,
            )

    def _register_buffer_to_track(
        self,
        name: str,
        shape: tuple[int, ...],
        store_callback: Callable,
        load_callback: Callable,
        dtype: torch.dtype = torch.float,
    ):
        """
        Register a buffer to track.
        When recording a sample, store_callback will be called to get a snapshot of the tensor to be tracked.
        When resetting the sample, load_callback will be called to load the snapshot.
        """
        self.staged_reset_buf[name] = {
            "type": "buffer",
            "store_callback": store_callback,
            "load_callback": load_callback,
            "data": torch.zeros(
                self.num_stages,
                self.staged_reset_max_samples_per_stage,
                *shape,
                device=self.device,
                dtype=dtype,
                requires_grad=False,
            ),
        }

    def _take_snapshot_of_buffered_states(self, advance_mask: torch.Tensor):
        """
        Take a snapshot of the buffered states.
        """
        advance_env_ids = torch.where(advance_mask)[0]
        advanced_stages = self.stage_buf[advance_mask]
        sample_idx_to_overwrite = (
            self.staged_reset_num_samples[advanced_stages, advance_env_ids]
            % self.staged_reset_max_samples_per_stage
        )
        for name, state_case in self.staged_reset_buf.items():
            if state_case["type"] == "buffer":
                state_case["data"][advanced_stages, sample_idx_to_overwrite, advance_env_ids] = (
                    state_case["store_callback"](advance_env_ids).clone()
                )
            elif state_case["type"] == "rigid_object" or state_case["type"] == "articulation":
                state_case["root_state"][
                    advanced_stages, sample_idx_to_overwrite, advance_env_ids
                ] = (state_case["obj"].data.root_state_w[advance_env_ids].clone())
                if state_case["type"] == "articulation":
                    if name == "robot":
                        state_case["dof_state"][
                            advanced_stages, sample_idx_to_overwrite, advance_env_ids, :, 0
                        ] = self.simulator.dof_pos[advance_env_ids].clone()
                        state_case["dof_state"][
                            advanced_stages, sample_idx_to_overwrite, advance_env_ids, :, 1
                        ] = self.simulator.dof_vel[advance_env_ids].clone()
                    else:
                        state_case["dof_state"][
                            advanced_stages, sample_idx_to_overwrite, advance_env_ids, :, 0
                        ] = (state_case["obj"].data.joint_pos[advance_env_ids].clone())
                        state_case["dof_state"][
                            advanced_stages, sample_idx_to_overwrite, advance_env_ids, :, 1
                        ] = (state_case["obj"].data.joint_vel[advance_env_ids].clone())
        self.staged_reset_num_samples[advanced_stages, advance_env_ids] += 1

    @override
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """
        Non-cooperative override of reset_envs_idx when staged_reset is enabled.
        StagedTaskBase is fully in charge of resetting the task states.
        """
        if not self.enable_staged_reset:
            super().reset_envs_idx(env_ids, target_states, target_buf)
            self.set_to_stage(env_ids, torch.zeros_like(env_ids))
            return

        if len(env_ids) == 0:
            return

        # sample stages to reset to
        selected_stages = self._sample_reset_stages(env_ids)

        # within the selected stages, uniformly sample a sample to reset to
        # TODO(haoru): importance sampling based on return
        selected_sample_indices = self._sample_reset_sample_indices(env_ids, selected_stages)

        # exclude stage 0 - those are handled by the user class
        stage_0_mask = selected_stages == 0
        user_env_ids = env_ids[stage_0_mask]
        selected_env_ids = env_ids[~stage_0_mask]
        selected_stages = selected_stages[~stage_0_mask]
        selected_sample_indices = selected_sample_indices[~stage_0_mask]

        # reset
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)  # if target_states is not None, reset to target states
        self._reset_past_obs_callback(env_ids)

        # handle reset robot and object states in stage 0
        # TODO(haoru): likely error will be thrown here because target_buf's length is env_ids
        if len(user_env_ids) > 0:
            self.set_to_stage(user_env_ids, torch.zeros_like(user_env_ids))
            self._reset_robot_states_callback(user_env_ids, target_states)
            self._reset_object_states_callback(user_env_ids)
            self._post_reset_callback(user_env_ids)

        # handle staged reset
        if len(selected_env_ids) > 0:
            self.set_to_stage(selected_env_ids, selected_stages)
            root_states = {}
            dof_states = {}
            for name, state_case in self.staged_reset_buf.items():
                if name == "robot":
                    self.target_robot_root_states[selected_env_ids] = state_case["root_state"][
                        selected_stages, selected_sample_indices, selected_env_ids
                    ].clone()
                    self.target_robot_dof_state[selected_env_ids] = state_case["dof_state"][
                        selected_stages, selected_sample_indices, selected_env_ids
                    ].clone()
                elif state_case["type"] == "rigid_object" or state_case["type"] == "articulation":
                    root_states[name] = torch.zeros(
                        self.num_envs,
                        13,
                        device=self.device,
                        dtype=torch.float,
                        requires_grad=False,
                    )
                    root_states[name][selected_env_ids] = state_case["root_state"][
                        selected_stages, selected_sample_indices, selected_env_ids
                    ].clone()
                    if state_case["type"] == "articulation":
                        obj: Articulation = state_case["obj"]
                        dof_pos = torch.zeros(
                            self.num_envs,
                            obj.num_joints,
                            device=self.device,
                            dtype=torch.float,
                            requires_grad=False,
                        )  # dof_pos
                        dof_vel = torch.zeros(
                            self.num_envs,
                            obj.num_joints,
                            device=self.device,
                            dtype=torch.float,
                            requires_grad=False,
                        )  # dof_vel
                        dof_pos[selected_env_ids] = state_case["dof_state"][
                            selected_stages, selected_sample_indices, selected_env_ids, :, 0
                        ].clone()
                        dof_vel[selected_env_ids] = state_case["dof_state"][
                            selected_stages, selected_sample_indices, selected_env_ids, :, 1
                        ].clone()
                        dof_states[name] = (
                            dof_pos,
                            dof_vel,
                            torch.arange(obj.num_joints, device=self.device, dtype=torch.long),
                        )
                elif state_case["type"] == "buffer":
                    state_case["load_callback"](
                        selected_env_ids,
                        state_case["data"][
                            selected_stages, selected_sample_indices, selected_env_ids
                        ].clone(),
                    )

            if root_states:
                self.simulator.set_task_root_state_tensor(selected_env_ids, root_states)
                self.simulator.set_task_dof_state_tensor(selected_env_ids, dof_states)

        # fill extras
        # haoru: this part of the code must be periodically synced with LeggedRobotBase
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf

        if self.config.domain_rand.push_robots:
            self.push_robot_recovery_counter[env_ids] = 0  # recovery counter reset.

    def _sample_reset_stages(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Sample stages to reset for the given envs.
        """
        # Create weight matrix: (len(env_ids), num_stages)
        # Start with base ratios for all environments
        weights = self.staged_reset_ratios.unsqueeze(0).repeat(len(env_ids), 1)

        # Get validity mask for the environments we're resetting
        # Shape: (num_stages, len(env_ids))
        valid_mask = self.staged_reset_num_samples[:, env_ids] > 0

        # Set invalid stages to zero weight
        # Transpose to match weights shape: (len(env_ids), num_stages)
        invalid_mask = ~valid_mask.T
        weights[invalid_mask] = 0.0

        # Sample stages for all environments simultaneously
        selected_stages = torch.multinomial(weights, 1).squeeze(1)

        return selected_stages

    def _sample_reset_sample_indices(
        self, env_ids: torch.Tensor, selected_stages: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample random sample indices within the selected stages for each environment.

        Args:
            env_ids: Environment IDs to reset (shape: [n])
            selected_stages: Selected stage for each environment (shape: [n])

        Returns:
            selected_sample_indices: Sample index for each environment (shape: [n])
        """
        # Gather number of available samples per (stage, env) pair; shape: [n]
        num_samples_per_env = self.staged_reset_num_samples[selected_stages, env_ids].clamp(
            max=self.staged_reset_max_samples_per_stage
        )

        # Sample uniformly in [0, num_samples) for each env in a vectorized way
        # rand in [0,1), scale by counts, floor to integers
        random_uniform = torch.rand(len(env_ids), device=self.device)
        selected_sample_indices = torch.floor(random_uniform * num_samples_per_env.float()).to(
            torch.long
        )

        # Safety clamp in case of any numerical edge cases
        selected_sample_indices = torch.minimum(selected_sample_indices, num_samples_per_env - 1)

        return selected_sample_indices

    def total_time_store_callback(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.total_time_buf[env_ids]

    def total_time_load_callback(self, env_ids: torch.Tensor, data: torch.Tensor):
        self.total_time_buf[env_ids] = data
