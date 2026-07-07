# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.distributions import Normal

from gr00t.rl.trl.modules.memory import Memory
from gr00t.rl.trl.modules.vision_actor_critic_modules import VisionActor, VisionCritic
from gr00t.rl.utils.running_mean_std import RunningMeanStd


class VisionRecurrentActor(VisionActor):
    """Vision Recurrent Actor that adds LSTM/GRU memory to the VisionActor class."""

    is_recurrent = True

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        max_rollout_history=1,
        input_key="actor_obs",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    ):
        # Initialize the base VisionActor without calling super().__init__() to avoid conflicts
        nn.Module.__init__(self)

        self.input_key = input_key
        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        init_noise_std = algo_config.init_noise_std
        self.max_rollout_history = max_rollout_history

        # Vision and MLP modules
        self.vision_module = instantiate(
            backbone.vision_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.vision_module_config_dict = backbone.vision_module.module_config_dict

        # Calculate concatenated dimension (actor_obs + vision features)
        vision_latent_dim = self.vision_module.output_dim
        concat_dim = obs_dim_dict[self.input_key] + vision_latent_dim

        # Add memory module
        self.memory = Memory(
            input_size=concat_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Replace the MLP module to take rnn_hidden_dim as input instead of concat_dim
        module_dim_dict_recurrent = module_dim_dict.copy()
        module_dim_dict_recurrent[self.input_key] = rnn_hidden_dim
        self.mlp_module = instantiate(
            backbone.mlp_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict={self.input_key: rnn_hidden_dim},
            module_dim_dict=module_dim_dict_recurrent,
            _recursive_=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd(
                (obs_dim_dict[self.input_key],), per_channel=True
            )

        # Action noise
        self.num_actions = self.mlp_module.output_dim
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        if algo_config.get("freeze_noise_std", False):
            self.std.requires_grad = False

        self.clamp_noise_std = algo_config.get("clamp_noise_std", False)
        if self.clamp_noise_std:
            self.max_noise_std = algo_config.get("max_noise_std", 1.0)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # Initialize observation buffer for rollout
        self.obs_dict_buffer = {}
        self.dones_buffer = None
        self.steps = 0
        self.is_eval_mode = False

        print(f"VisionRecurrentActor RNN: {self.memory}")
        print(f"VisionRecurrentActor MLP: {self.mlp_module}")

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
        """Override parent method to handle cur_dones parameter."""
        update_episode_attnmask = False
        for key in obs_dict.keys():
            if key not in self.obs_dict_buffer:
                self.obs_dict_buffer[key] = obs_dict[key].unsqueeze(1)
            else:
                self.obs_dict_buffer[key] = torch.cat(
                    [self.obs_dict_buffer[key], obs_dict[key].unsqueeze(1)], dim=1
                )
            if self.obs_dict_buffer[key].shape[1] > self.max_rollout_history:
                update_episode_attnmask = True
                self.obs_dict_buffer[key] = self.obs_dict_buffer[key][
                    :, -self.max_rollout_history :
                ]

        if episode_attnmask is not None and update_episode_attnmask:
            episode_attnmask = episode_attnmask[
                :, -self.max_rollout_history :, -self.max_rollout_history :
            ]

        return episode_attnmask

    def reset(self, dones=None):
        """Reset memory hidden states for done environments."""
        self.memory.reset(dones)

    def forward(self, obs_dict, masks=None, hidden_states=None, episode_attnmask=None, **kwargs):
        if self.running_mean_std is not None:
            obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        elif len(original_shape) == 6:  # Batch training mode: [batch_size, seq_len, H, W, C]
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()
            actor_obs = obs_dict[self.input_key]  # Should have shape [batch_size, seq_len, obs_dim]
        else:
            actor_obs = obs_dict[self.input_key]  # Should have shape [batch_size, obs_dim]

        # Concatenate observations and vision features
        concated_obs = torch.cat([actor_obs, latent], dim=-1)

        # Pass through memory module
        if len(concated_obs.shape) == 2:  # Rollout mode: [batch_size, concat_dim]
            memory_out = self.memory(concated_obs)
        else:  # Batch training mode: [batch_size, seq_len, concat_dim]
            batch_size, seq_len, concat_dim = concated_obs.shape
            # Reshape for RNN: [seq_len, batch_size, concat_dim]
            concated_obs = concated_obs.transpose(0, 1)
            memory_out = self.memory(concated_obs, masks=masks, hidden_states=hidden_states)
            # Reshape back: [batch_size, seq_len, hidden_dim]
            memory_out = memory_out.transpose(0, 1)

        result = self.mlp_module(memory_out)
        return result

    def get_hidden_states(self):
        """Get current hidden states from memory."""
        return self.memory.hidden_states

    def act(self, obs_dict, episode_attnmask=None, masks=None, hidden_states=None, **kwargs):
        """Forward pass for sampling actions during training."""
        try:
            self.update_distribution(
                obs_dict,
                episode_attnmask=episode_attnmask,
                last_step_only=False,
                masks=masks,
                hidden_states=hidden_states,
                **kwargs,
            )
        except Exception as e:
            raise e
        actions = self.distribution.sample()
        return {
            "actions": actions,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def update_distribution(
        self,
        obs_dict,
        episode_attnmask=None,
        last_step_only=False,
        masks=None,
        hidden_states=None,
        **kwargs,
    ):
        mean = self.forward(
            obs_dict,
            masks=masks,
            hidden_states=hidden_states,
            episode_attnmask=episode_attnmask,
            **kwargs,
        )
        # The obj-pred vision actor's forward returns {"actions", "obj_pred"}; the
        # action-distribution (RL/GRPO) path only needs the action mean.
        if isinstance(mean, dict):
            mean = mean["actions"]
        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Rollout method for recurrent actor during environment interaction."""
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict = obs_dict.copy()
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # For recurrent models, we use the memory hidden states rather than obs buffer
        # Pass current obs through memory and get output
        # Handle vision processing
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        # Process vision
        original_shape = image_input.shape
        batch_size = original_shape[0]
        image_reshaped = image_input
        effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        # Concatenate and pass through memory
        concated_obs = torch.cat([obs_dict[self.input_key], latent], dim=-1)
        memory_out = self.memory(concated_obs)

        # Remove sequence dimension added by memory module during inference
        if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
            memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]

        # Update distribution using memory output
        mean = self.mlp_module(memory_out)
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

        self.steps += 1
        return {
            "actions": self.distribution.sample(),
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Inference method for recurrent actor."""
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        # For recurrent models, pass through memory
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict = obs_dict.copy()
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # Handle vision processing
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        # Process vision
        original_shape = image_input.shape
        batch_size = original_shape[0]
        image_reshaped = image_input
        effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        # Concatenate and pass through memory
        concated_obs = torch.cat([obs_dict[self.input_key], latent], dim=-1)
        memory_out = self.memory(concated_obs)

        # Remove sequence dimension added by memory module during inference
        if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
            memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]

        actions_mean = self.mlp_module(memory_out)

        self.steps += 1
        return actions_mean


class VisionRecurrentCritic(VisionCritic):
    """Vision Recurrent Critic that adds LSTM/GRU memory to the VisionCritic class."""

    is_recurrent = True

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    ):
        # Initialize the base VisionCritic without calling super().__init__() to avoid conflicts
        nn.Module.__init__(self)

        obs_dim_dict = env_config.robot.algo_obs_dim_dict

        # Vision module
        self.vision_module = instantiate(
            backbone.vision_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )
        self.vision_module_config_dict = backbone.vision_module.module_config_dict

        # Calculate concatenated dimension (critic_obs + vision features)
        vision_latent_dim = self.vision_module.output_dim
        concat_dim = obs_dim_dict["critic_obs"] + vision_latent_dim

        # Add memory module
        self.memory = Memory(
            input_size=concat_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Replace the MLP module to take rnn_hidden_dim as input instead of concat_dim
        module_dim_dict_recurrent = module_dim_dict.copy()
        module_dim_dict_recurrent["critic_obs"] = rnn_hidden_dim
        self.mlp_module = instantiate(
            backbone.mlp_module,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict={"critic_obs": rnn_hidden_dim},
            module_dim_dict=module_dim_dict_recurrent,
            _recursive_=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)

        print(f"VisionRecurrentCritic RNN: {self.memory}")
        print(f"VisionRecurrentCritic MLP: {self.mlp_module}")

    def reset(self, dones=None):
        """Reset memory hidden states for done environments."""
        self.memory.reset(dones)

    def get_hidden_states(self):
        """Get current hidden states from memory."""
        return self.memory.hidden_states

    def evaluate(self, obs_dict, masks=None, hidden_states=None, episode_attnmask=None, **kwargs):
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])

        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        elif len(original_shape) == 6:  # Batch training mode: [batch_size, seq_len, H, W, C]
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()
            critic_obs = obs_dict["critic_obs"]  # Should have shape [batch_size, seq_len, obs_dim]
        else:
            critic_obs = obs_dict["critic_obs"]  # Should have shape [batch_size, obs_dim]

        # Concatenate observations and vision features
        concated_obs = torch.cat([critic_obs, latent], dim=-1)

        # Pass through memory module
        if len(concated_obs.shape) == 2:  # Rollout mode: [batch_size, concat_dim]
            memory_out = self.memory(concated_obs)
            # Remove sequence dimension added by memory module during inference
            if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
                memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]
        else:  # Batch training mode: [batch_size, seq_len, concat_dim]
            batch_size, seq_len, concat_dim = concated_obs.shape
            # Reshape for RNN: [seq_len, batch_size, concat_dim]
            concated_obs = concated_obs.transpose(0, 1)
            memory_out = self.memory(concated_obs, masks=masks, hidden_states=hidden_states)
            # Reshape back: [batch_size, seq_len, hidden_dim]
            memory_out = memory_out.transpose(0, 1)

        value = self.mlp_module(memory_out)
        return value
