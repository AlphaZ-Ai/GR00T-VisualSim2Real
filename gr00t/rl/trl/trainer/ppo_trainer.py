# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import deque
from copy import deepcopy
from typing import Dict, Optional

import pandas as pd
import torch
import torchvision
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from tqdm import tqdm
from transformers.trainer import *
from trl.trainer.ppo_trainer import *

from gr00t.rl.agents.modules.data_utils import RolloutStorage
from gr00t.rl.trl.callbacks.hv_callback_handler import HVCallbackHandler
from gr00t.rl.trl.utils.common import wandb_run_exists
from gr00t.rl.trl.utils.rl import compute_episode_attnmask
from gr00t.rl.trl.utils.scheduler import update_scheduled_params
from gr00t.rl.utils.average_meters import TensorAverageMeterDict

console = Console()
import time

import numpy as np


class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model, ref_model=None) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.ref_model = ref_model

    def set_mode(self, mode):
        if hasattr(self.policy, "mode"):
            self.policy.mode = mode

    def transform_train(self):
        if hasattr(self.policy, "transform_train"):
            self.policy.transform_train()

    def transform_eval(self):
        if hasattr(self.policy, "transform_eval"):
            self.policy.transform_eval()

    def forward(self, modes, input_kwargs):
        results = {}
        for mode in modes:
            results[mode] = self.forward_component(mode, **input_kwargs[mode])
        return results

    def forward_component(self, mode, actions=None, **kwargs):
        if mode == "policy":
            self.policy.act(**kwargs)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "logprobs": log_probs,
                "action_mean": self.policy.action_mean,
                "action_std": self.policy.action_std,
                "entropy": self.policy.entropy,
            }
        elif mode == "policy_distill":
            results = self.policy.act(**kwargs)
        elif mode == "policy_distill_ppo":
            policy_state_dict = self.policy.act(**kwargs)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_std": policy_state_dict["action_sigma"],
                "entropy": self.policy.entropy,
            }
            if "normalized_actions" in policy_state_dict:
                results["normalized_actions"] = policy_state_dict["normalized_actions"]
        elif mode == "policy_w_and_wo_imgaug":
            # The first forward is without image augmentation
            self.policy.transform_eval()
            policy_state_dict = self.policy.act(**kwargs)
            # Use the distribution without image augmentation to get the log_probs
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_std": policy_state_dict["action_sigma"],
                "entropy": self.policy.entropy,
            }

            # The second forward is with image augmentation
            self.policy.transform_train()
            # The second time doesn't need deepcopy
            policy_state_dict_w_imgaug = self.policy.act(**kwargs)
            results["action_mean_w_imgaug"] = policy_state_dict_w_imgaug["action_mean"]
            results["actions_w_imgaug"] = policy_state_dict_w_imgaug["actions"]
            if "normalized_actions" in policy_state_dict_w_imgaug:
                results["normalized_actions_w_imgaug"] = policy_state_dict_w_imgaug[
                    "normalized_actions"
                ]
        elif mode == "policy_deterministic":
            self.policy.act(**kwargs)
            results = {
                "action_mean": self.policy.action_mean,
            }
        elif mode == "vae_policy_deterministic":
            self.policy.act(**kwargs)
            prior_mu, prior_log_var = self.policy.eval_prior(**kwargs)
            results = {
                "action_mean": self.policy.action_mean,
                "vae_mu": self.policy.z_mu,
                "vae_log_var": self.policy.z_log_sigma,
                "prior_mu": prior_mu,
                "prior_log_var": prior_log_var,
            }
        elif mode == "value":
            results = self.value_model.evaluate(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return results


class PrinterHVCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            width = 80
            pad = 35
            print_str = f" \033[1m Learning iteration {state.global_step}  \033[0m "

            log_string = (
                f"""{print_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {logs['fps']:.0f} steps/s (Collection: {logs['collection_time']:.3f}s, Learning {logs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {logs['Policy/mean_noise_std']:.2f}\n"""
            )

            for k, v in logs.items():
                if k.startswith("objective/"):
                    # Keep the original logic
                    if k.startswith("objective/kin_"):
                        log_string += f"""{f'{k}:':>{pad}} {v:.5f}\n"""
                    else:
                        new_key = k.replace("objective/", "")
                        log_string += f"""{f'Mean {new_key}:':>{pad}} {v:.5f}\n"""
                elif k.startswith("loss/"):
                    # Add support for loss/ prefix metrics
                    new_key = k.replace("loss/", "")
                    log_string += f"""{f'{new_key}:':>{pad}} {v:.6f}\n"""

            env_log_string = ""
            ep_string = ""
            for k, v in logs.items():
                if k.startswith("Env/"):
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                    env_log_string += f"{entry}\n"
                if k.startswith("Episode/"):
                    new_key = k.replace("Episode/", "")
                    ep_string += f"""{f'Mean episode {new_key}:':>{pad}} {v:.4f}\n"""

            log_string += env_log_string
            log_string += ep_string
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total episodes:':>{pad}} {logs['episode']}\n"""
                f"""{'Total timesteps:':>{pad}} {logs['tot_timesteps']}\n"""
                f"""{'Iteration time:':>{pad}} {logs['collection_time'] + logs['learn_time']:.2f}s\n"""
                f"""{'Total time:':>{pad}} {logs['tot_time']:.2f}s\n"""
                f"""{'ETA:':>{pad}} {logs['tot_time'] / logs['batch_idx'] * (logs['num_total_batches'] - logs['batch_idx']):.1f}s\n"""
            )

            log_string += f"Logging Directory: {logs['experiment_save_dir']}"
            with Live(
                Panel(log_string, title="Training Log"), refresh_per_second=4, console=console
            ):
                # Your training loop or other operations
                pass


def process_ep_infos(ep_infos, device):
    infos = {}
    for key in ep_infos[0]:
        infotensor = torch.tensor([], device=device)
        for ep_info in ep_infos:
            # handle scalar and zero dimensional tensor infos
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key].to(device)))
        value = torch.mean(infotensor)
        infos[key] = value
    return infos


class TRLPPOTrainer(PPOTrainer):
    """
    Custom PPO Trainer that adapts TRL's PPOTrainer to work with Humanoid environments.
    """

    _tag_names = ["trl", "humanoid_ppo"]

    def __init__(
        self,
        args,
        config,
        env,
        model,
        ref_model=None,
        reward_model=None,
        processing_class=None,
        value_model=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        log_dir=None,
        # less commonly used
        optimizers=(None, None),
        callbacks=None,
        peft_config=None,
        use_ref_model=False,
        checkpoint=None,
        local_seed=None,
        schedule_dict=None,
        accelerator=None,
    ) -> None:
        self.accelerator = accelerator
        self._init_trl(
            args,
            config,
            env,
            processing_class,
            model,
            ref_model,
            reward_model,
            train_dataset,
            value_model,
            data_collator,
            eval_dataset,
            optimizers,
            callbacks,
            peft_config,
            use_ref_model,
            local_seed,
            log_dir,
            schedule_dict=schedule_dict,
        )
        self._init_config()
        self._setup_storage()

        # Initialize trajectory counter for recurrent policy training
        self._current_first_traj = 0

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def _init_trl(
        self,
        args,
        config,
        env,
        processing_class,
        model,
        ref_model,
        reward_model,
        train_dataset,
        value_model,
        data_collator,
        eval_dataset,
        optimizers,
        callbacks,
        peft_config,
        use_ref_model,
        local_seed,
        log_dir,
        schedule_dict=None,
    ):
        self.args = args
        self.config = config
        self.env = env
        self.processing_class = processing_class
        self.policy_model = model
        self.learn_normalized_actions = model.has_normalized_actions
        self.episode_env_tensors = TensorAverageMeterDict()
        self.ep_infos = []
        self.eval_callbacks = []
        self.log_dir = log_dir
        self.schedule_dict = schedule_dict
        self.scheduled_params_dict = {}

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if use_ref_model:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model:
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(self.policy_model)
        else:
            self.ref_model = None

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = (
            len(train_dataset) if train_dataset is not None else self.env.config.num_envs
        )
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset

        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########

        accelerator = self.accelerator

        self.device = accelerator.device
        args.global_rank = accelerator.process_index
        args.world_size = accelerator.num_processes
        args.is_main_process = accelerator.is_main_process
        args.local_batch_size = self.env.config.num_envs
        args.batch_size = int(args.local_batch_size * args.world_size)
        try:
            args.mini_batch_size = exact_div(
                args.batch_size,
                args.num_mini_batches,
                "`batch_size` must be a multiple of `num_mini_batches`",
            )
            args.local_mini_batch_size = exact_div(
                args.local_batch_size,
                args.num_mini_batches,
                "`local_batch_size` must be a multiple of `num_mini_batches`",
            )
        except Exception as e:
            print(f"Error: {e}")
            args.mini_batch_size = 1
            args.local_mini_batch_size = 1

        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = (
                args.local_mini_batch_size
            )  # same as mini-batch size, which implies no micro-batching (num_micro_batches = 1)
        args.num_micro_batches = args.local_mini_batch_size // args.per_device_train_batch_size
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        if args.total_episodes is None:
            assert args.num_total_batches is not None
            args.total_episodes = args.num_total_batches * args.batch_size
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = local_seed
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        if self.config.get("disable_dropout", True):
            for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
                if module is not None:
                    disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        # self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = HVCallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
            self.env,
            self.accelerator,
        )
        self.add_callback(
            PrinterHVCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        if self.train_dataset is not None:
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.local_dataloader_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=False,  # needed; otherwise the last batch will be of ragged shape
            )
        else:
            self.dataloader = None
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)

        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.unwrapped_model = unwrap_model(self.model)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
            )  # no need to shuffle eval dataset
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        else:
            self.eval_dataloader = None

        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                # if not self.is_peft_model:
                #     raise ValueError("No reference model and model is not a Peft model.")
                pass
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
        self.use_apex = False

        self.train_with_evaluating_env = self.config.get("train_with_evaluating_env", False)

        # Camera resolution
        if "vision_obs" in self.env.config.obs.obs_dict:
            if (
                self.env.config.obs.obs_dict.vision_obs[0] in ["depth_image", "height_map"]
                and not self.env.config.simulator.config.cameras.stack_depth_image
            ):
                num_channels = 1
            else:
                num_channels = 3

            if self.env.config.obs.obs_dict.vision_obs[0] == "height_map":
                heightmap_resolution = self.env.config.simulator.config.heightmap.resolution
                self.camera_resolution = [heightmap_resolution, heightmap_resolution] + [
                    num_channels
                ]
            else:
                self.camera_resolution = (
                    self.env.config.simulator.config.cameras.camera_resolutions + [num_channels]
                )
        else:
            self.camera_resolution = None

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.policy_model.num_actions

        self.num_steps_per_env = self.config.num_steps_per_env
        self.use_padding_mask = self.config.get("use_padding_mask", False)
        self.ppo_shuffle_every_epoch = self.config.get("ppo_shuffle_every_epoch", True)
        self.empty_cache_every_n_ppo_epoch = self.config.get("empty_cache_every_n_ppo_epoch", -1)

        self.entropy_coef = self.config.entropy_coef
        self.desired_kl = self.config.desired_kl
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.sync_advantage_normalization = self.config.get("sync_advantage_normalization", True)

        self.compute_imgaug_bc_loss = self.config.get("compute_imgaug_bc_loss", False)
        self.imgaug_bc_loss_coef = self.config.get("imgaug_bc_loss_coef", 1.0)
        self.imgaug_bc_loss_fn = torch.nn.MSELoss()

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.env.num_envs, self.num_steps_per_env, device=self.accelerator.device
        )

        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            if obs_key == "vision_obs":
                # Handle vision observations - check if it's history-based
                if (
                    hasattr(self.env.config.obs, "obs_auxiliary")
                    and "rgb_image_history" in self.env.config.obs.obs_auxiliary
                ):
                    # The environment's HistoryHandler provides [num_envs, history_length, H, W, C]
                    # We need to store this as [num_steps, num_envs, history_length, H, W, C] in storage
                    history_config = self.env.config.obs.obs_auxiliary.rgb_image_history
                    history_length = history_config.rgb_image

                    # Register with full 5D shape: [history_length, H, W, C]
                    vision_shape = tuple([history_length] + self.camera_resolution)
                    self.storage.register_key(obs_key, shape=vision_shape, dtype=torch.float)
                    self.vision_history_length = history_length
                    self.vision_obs_key = obs_key
                else:
                    # Regular single-frame vision observation: [H, W, C]
                    assert obs_dim == np.prod(
                        self.camera_resolution
                    ), f"{obs_dim=}, {self.camera_resolution=}"
                    self.storage.register_key(
                        obs_key, shape=tuple(self.camera_resolution), dtype=torch.float
                    )
                    self.vision_history_length = None
                    self.vision_obs_key = None
            elif obs_key == "state_history_obs":
                # The environment's HistoryHandler provides [num_envs, history_length, obs_dim]
                # We need to store this as [num_steps, num_envs, history_length, obs_dim] in storage
                history_config = self.env.config.obs.obs_auxiliary.state_history
                # get the max history length from history_config values
                history_length = max(history_config.values())
                concat_obs_dim = 0
                for obs_name in history_config.keys():
                    concat_obs_dim += self.env.config.obs.obs_dims[obs_name]

                self.storage.register_key(
                    obs_key, shape=(history_length, concat_obs_dim), dtype=torch.float
                )
            else:
                self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)

        # Set vision history flags
        if not hasattr(self, "vision_history_length"):
            self.vision_history_length = None
            self.vision_obs_key = None

        ## Register others
        self.storage.register_key("actions", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("rewards", shape=(1,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)
        self.storage.register_key("time_outs", shape=(1,), dtype=torch.bool)
        self.storage.register_key("values", shape=(1,), dtype=torch.float)
        self.storage.register_key("returns", shape=(1,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(1,), dtype=torch.float)
        self.storage.register_key("actions_log_prob", shape=(1,), dtype=torch.float)
        self.storage.register_key("action_mean", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("action_sigma", shape=(self.num_act,), dtype=torch.float)

        if self.learn_normalized_actions:
            self.storage.register_key(
                "normalized_actions", shape=(self.num_act,), dtype=torch.float
            )

        self.state.rewbuffer = deque(maxlen=100)
        self.state.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.accelerator.device
        )

        self.state.cur_reward_sum = self.cur_reward_sum
        self.state.cur_episode_length = self.cur_episode_length
        self.ep_infos = []
        self.state.tot_timesteps = 0
        self.state.tot_time = 0
        self.state.eval_step = 0
        self.state.eval_render_step = 0

    def policy_step(self, policy_model, obs_dict, cur_dones=None, store_hidden_states=True):
        actor_obs_dict = deepcopy(obs_dict)

        if cur_dones is None:
            dones = (
                self.storage.query_key("dones")
                .to(self.accelerator.device)[: self.storage.step + 1]
                .squeeze(-1)
                .transpose(0, 1)
            )
            episode_attnmask = compute_episode_attnmask(dones)
        else:
            episode_attnmask = None

        # Store hidden states BEFORE rollout for recurrent policies
        actor_hidden_states = None
        if (
            store_hidden_states
            and hasattr(policy_model, "is_recurrent")
            and policy_model.is_recurrent
        ):
            actor_hidden_states = policy_model.get_hidden_states()

        policy_state_dict = policy_model.rollout(
            obs_dict=actor_obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones
        )
        actions = policy_state_dict["actions"]
        actions_log_prob = policy_model.get_actions_log_prob(actions).unsqueeze(1)
        policy_state_dict["actions_log_prob"] = actions_log_prob

        # Add hidden states to return dict if they were captured
        if store_hidden_states and actor_hidden_states is not None:
            policy_state_dict["hidden_states"] = (
                actor_hidden_states,
                None,
            )  # (actor, critic) - critic is computed separately

        return policy_state_dict

    def _chunked_value_evaluate(self, value_model, obs_dict, episode_attnmask, chunk_size=1024):
        batch_size = list(obs_dict.values())[0].shape[0]
        if batch_size <= chunk_size:
            return value_model.evaluate(obs_dict=obs_dict, episode_attnmask=episode_attnmask)

        obs_chunks = {}
        for key, value in obs_dict.items():
            obs_chunks[key] = torch.split(value, chunk_size, dim=0)
        if episode_attnmask is not None:
            attnmask_chunks = torch.split(episode_attnmask, chunk_size, dim=0)
        else:
            attnmask_chunks = [None] * len(obs_chunks[list(obs_chunks.keys())[0]])

        value_chunks = []
        for i in range(len(attnmask_chunks)):
            chunk_obs_dict = {key: obs_chunks[key][i] for key in obs_chunks}
            chunk_values = value_model.evaluate(
                obs_dict=chunk_obs_dict, episode_attnmask=attnmask_chunks[i]
            )
            value_chunks.append(chunk_values)
        return torch.cat(value_chunks, dim=0)

    def _rollout_step(self, model, obs_dict):
        self._train_rollout_mode()
        device = self.accelerator.device
        policy_model = model.policy
        value_model = model.value_model
        policy_model.init_rollout()
        self.storage.clear()

        dones = torch.zeros(self.env.num_envs, device=device)
        with torch.no_grad():
            values = []
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                policy_state_dict = self.policy_step(policy_model, obs_dict, cur_dones=dones)

                # For recurrent critics, compute values step-by-step and capture critic hidden states
                if self.value_model is not None:
                    value_is_recurrent = (
                        hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
                    )
                    if value_is_recurrent:
                        critic_hidden_states = (
                            self.value_model.get_hidden_states()
                            if hasattr(self.value_model, "get_hidden_states")
                            else None
                        )
                        critic_obs_dict = {k: v for k, v in obs_dict.items() if k != "actor_obs"}
                        step_values = self.value_model.evaluate(obs_dict=critic_obs_dict)
                        values.append(step_values)
                        # Store both actor and critic hidden states
                        combined_hidden_states = (
                            policy_state_dict.get("hidden_states", (None, None))[0],
                            critic_hidden_states,
                        )
                        policy_state_dict["hidden_states"] = combined_hidden_states

                # Append states to storage
                for key, value in obs_dict.items():
                    self.storage.update_key(key, value)

                for key, value in policy_state_dict.items():
                    # Skip hidden_states as they're stored separately
                    if key != "hidden_states":
                        self.storage.update_key(key, value)

                # Store hidden states separately for recurrent policies
                if "hidden_states" in policy_state_dict:
                    self.storage._save_hidden_states(policy_state_dict["hidden_states"])  # type: ignore[attr-defined]

                # Step the environment
                actor_state = {"actions": policy_state_dict["actions"]}
                # Pass gt_actions to environment if available (for distillation mode)
                if "gt_actions" in policy_state_dict:
                    actor_state["gt_actions"] = policy_state_dict["gt_actions"]
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)
                rewards, dones = rewards.to(device), dones.to(device)
                rewards_stored = rewards.clone().unsqueeze(1)
                assert len(rewards_stored.shape) == 2

                self.ep_infos.append(infos["episode"])
                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.update_key("time_outs", infos["time_outs"].unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                self.cur_reward_sum += rewards
                self.cur_episode_length += 1

                new_ids = (dones > 0).nonzero(as_tuple=False)
                self.state.rewbuffer.extend(
                    self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                self.state.lenbuffer.extend(
                    self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                )

                self.cur_reward_sum[new_ids] = 0
                self.cur_episode_length[new_ids] = 0

            if values:
                values = torch.stack(values, dim=0)

            if self.value_model is not None:
                # Check if critic is recurrent
                value_is_recurrent = (
                    hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
                )

                if value_is_recurrent:
                    # For recurrent critics: values were already computed step-by-step during rollout
                    # We only need to compute the final bootstrapping value
                    critic_obs_dict = {k: v for k, v in obs_dict.items() if k != "actor_obs"}
                    last_values = self.value_model.evaluate(
                        obs_dict=critic_obs_dict
                    )  # Shape: [num_envs, 1]
                else:
                    # For non-recurrent critics: compute all values at once
                    dones = self.storage.query_key("dones").to(device).squeeze(-1).transpose(0, 1)
                    dones = torch.cat([dones, torch.zeros_like(dones[:, :1])], dim=1)
                    episode_attnmask = compute_episode_attnmask(dones)
                    all_obs_dict = {}
                    for key in obs_dict.keys():
                        if key not in ["actor_obs"]:  # actor_obs not required by value model
                            obs_value = self.storage.query_key(key).to(device)
                            obs_value = torch.cat([obs_value, obs_dict[key].unsqueeze(0)], dim=0)
                            all_obs_dict[key] = obs_value.transpose(0, 1)
                    all_values = self._chunked_value_evaluate(
                        value_model, all_obs_dict, episode_attnmask
                    ).transpose(0, 1)
                    values, last_values = all_values[:-1], all_values[-1]

                rewards = self.storage.query_key("rewards")
                new_rewards = (
                    rewards.to(device)
                    + self.gamma * self.storage.query_key("time_outs").to(device) * values
                )
                self.storage.batch_update_data("rewards", new_rewards)

                returns, advantages = self._compute_returns(
                    values=values,
                    last_values=last_values,
                    policy_state_dict={
                        "dones": self.storage.query_key("dones"),
                        "rewards": self.storage.query_key("rewards"),
                    },
                )
                self.storage.batch_update_data("values", values)
                self.storage.batch_update_data("returns", returns)
                self.storage.batch_update_data("advantages", advantages)

        policy_model.clear_rollout()

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.policy_model.reset(dones)
        if self.value_model is not None:
            self.value_model.reset(dones)
        self.episode_env_tensors.add(infos["to_log"])

    def _register_stats_buffer(self):
        args = self.args
        device = self.accelerator.device

        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        weighted_ppo_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)
            weighted_imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)

        self.approxkl_stats = approxkl_stats
        self.pg_clipfrac_stats = pg_clipfrac_stats
        self.pg_loss_stats = pg_loss_stats
        self.vf_loss_stats = vf_loss_stats
        self.entropy_stats = entropy_stats
        self.weighted_ppo_loss_stats = weighted_ppo_loss_stats
        self.vf_clipfrac_stats = vf_clipfrac_stats
        self.ratio_stats = ratio_stats
        if self.compute_imgaug_bc_loss:
            self.imgaug_bc_loss_stats = imgaug_bc_loss_stats
            self.weighted_imgaug_bc_loss_stats = weighted_imgaug_bc_loss_stats

    def _get_rollout_data(self, obs_keys):
        device = self.accelerator.device

        # Handle all observations - vision history needs special handling for tensor dimensions
        all_obs_dict = {}
        for key in obs_keys:
            obs_data = self.storage.query_key(key).transpose(0, 1).to(device)
            all_obs_dict[key] = obs_data

        actions = self.storage.actions.transpose(0, 1).to(device)
        logprobs = self.storage.actions_log_prob.transpose(0, 1).squeeze(-1).to(device)
        values = self.storage.values.transpose(0, 1).squeeze(-1).to(device)
        rewards = self.storage.rewards.transpose(0, 1).squeeze(-1).to(device)
        dones = self.storage.dones.transpose(0, 1).squeeze(-1).to(device)
        old_mu_batch = self.storage.action_mean.transpose(0, 1).to(device)
        old_sigma_batch = self.storage.action_sigma.transpose(0, 1).to(device)
        returns = self.storage.returns.transpose(0, 1).squeeze(-1).to(device)
        advantages = self.storage.advantages.transpose(0, 1).squeeze(-1).to(device)

        if self.use_padding_mask:
            padding_mask = dones.clone()
            padding_mask_p1 = padding_mask.clone()
            for i in range(padding_mask.shape[0]):
                true_indices = torch.where(padding_mask[i])[0]
                if len(true_indices) > 0:
                    padding_mask[i, true_indices[0]] = False
                    padding_mask_p1[
                        i, true_indices[0] : min(true_indices[0] + 2, padding_mask_p1.shape[1])
                    ] = False
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask_p1, 0)
        else:
            padding_mask = torch.zeros_like(dones)
            padding_mask_p1 = torch.zeros_like(dones)

        # For recurrent policies/critics, pre-split trajectories ONCE globally
        padded_obs_dict = None
        trajectory_masks = None
        if (hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent) or (
            hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
        ):
            from gr00t.rl.trl.utils.rl import split_and_pad_trajectories

            padded_obs_dict = {}
            for key in obs_keys:
                # all_obs_dict[key]: [num_envs, num_steps, ...]
                # Transpose to [num_steps, num_envs, ...] for split_and_pad_trajectories
                obs_transposed = all_obs_dict[key].transpose(0, 1)
                dones_transposed = dones.transpose(0, 1)
                padded_obs, traj_masks = split_and_pad_trajectories(
                    obs_transposed, dones_transposed
                )
                # padded_obs: [max_traj_len, num_trajectories, ...]
                # Transpose to [num_trajectories, max_traj_len, ...]
                padded_obs_dict[key] = padded_obs.transpose(0, 1)
                if trajectory_masks is None:
                    # traj_masks: [max_traj_len, num_trajectories]
                    # Transpose to [num_trajectories, max_traj_len]
                    trajectory_masks = traj_masks.transpose(0, 1)

        return dict(
            all_obs_dict=all_obs_dict,
            actions=actions,
            logprobs=logprobs,
            values=values,
            rewards=rewards,
            dones=dones,
            old_mu_batch=old_mu_batch,
            old_sigma_batch=old_sigma_batch,
            returns=returns,
            advantages=advantages,
            padding_mask=padding_mask,
            padding_mask_p1=padding_mask_p1,
            padded_obs_dict=padded_obs_dict,
            trajectory_masks=trajectory_masks,
        )

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        mb_advantage = rollout_data["advantages"][micro_batch_inds]
        mb_logprobs = rollout_data["logprobs"][micro_batch_inds]
        mb_return = rollout_data["returns"][micro_batch_inds]
        mb_values = rollout_data["values"][micro_batch_inds]
        mb_dones = rollout_data["dones"][micro_batch_inds]
        mb_actions = rollout_data["actions"][micro_batch_inds]
        mb_old_mu = rollout_data["old_mu_batch"][micro_batch_inds]
        mb_old_sigma = rollout_data["old_sigma_batch"][micro_batch_inds]
        mb_padding_mask = rollout_data["padding_mask"][micro_batch_inds]
        mb_padding_mask_p1 = rollout_data["padding_mask_p1"][micro_batch_inds]

        episode_attnmask = compute_episode_attnmask(mb_dones)

        # For recurrent policies, SLICE pre-split trajectories instead of re-splitting
        mb_hidden_states = None
        if hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent:
            if (
                rollout_data.get("padded_obs_dict") is None
                or rollout_data.get("trajectory_masks") is None
            ):
                raise RuntimeError(
                    "Recurrent policy requires padded_obs_dict and trajectory_masks in rollout_data! This should have been created in _get_rollout_data."
                )
            # Calculate how many trajectories are in this mini-batch
            last_was_done = torch.zeros_like(mb_dones, dtype=torch.bool)
            last_was_done[:, 1:] = mb_dones[:, :-1].bool()
            last_was_done[:, 0] = True
            num_trajectories = torch.sum(last_was_done).item()
            # Slice from globally pre-split trajectories using trajectory counter
            first_traj = self._current_first_traj
            last_traj = first_traj + num_trajectories
            mb_obs_dict = {
                key: rollout_data["padded_obs_dict"][key][first_traj:last_traj]
                for key in rollout_data["padded_obs_dict"].keys()
            }
            mb_masks = rollout_data["trajectory_masks"][first_traj:last_traj]
            # Update trajectory counter for next mini-batch
            self._current_first_traj = last_traj

            # Extract hidden states at trajectory boundaries if available
            if self.storage.saved_hidden_states_a is not None or self.storage.saved_hidden_states_c is not None:  # type: ignore[attr-defined]
                # Determine which timesteps are right after dones (start of trajectories)
                dones_for_extraction = mb_dones.clone()
                last_was_done = torch.zeros_like(dones_for_extraction, dtype=torch.bool)
                last_was_done[:, 1:] = dones_for_extraction[:, :-1].bool()
                last_was_done[:, 0] = True
                # Permute to [num_steps, batch_size]
                last_was_done_transposed = last_was_done.transpose(0, 1)
                # Extract actor hidden states
                if self.storage.saved_hidden_states_a is not None:  # type: ignore[attr-defined]
                    hid_a_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_a:  # type: ignore[attr-defined]
                        saved_hid_mb = saved_hidden_states[:, :, micro_batch_inds, :]
                        saved_hid_mb = saved_hid_mb.permute(2, 0, 1, 3)
                        saved_hid_mb_flat = saved_hid_mb.reshape(
                            -1, saved_hid_mb.shape[2], saved_hid_mb.shape[3]
                        )
                        hid_at_traj_starts = saved_hid_mb_flat[last_was_done.reshape(-1)]
                        hid_at_traj_starts = hid_at_traj_starts.transpose(1, 0).contiguous()
                        hid_a_batch.append(hid_at_traj_starts)
                    hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else tuple(hid_a_batch)
                else:
                    hid_a_batch = None
                # Extract critic hidden states
                if self.storage.saved_hidden_states_c is not None:  # type: ignore[attr-defined]
                    hid_c_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_c:  # type: ignore[attr-defined]
                        saved_hid_mb = saved_hidden_states[:, :, micro_batch_inds, :]
                        saved_hid_mb = saved_hid_mb.permute(2, 0, 1, 3)
                        saved_hid_mb_flat = saved_hid_mb.reshape(
                            -1, saved_hid_mb.shape[2], saved_hid_mb.shape[3]
                        )
                        hid_at_traj_starts = saved_hid_mb_flat[last_was_done.reshape(-1)]
                        hid_at_traj_starts = hid_at_traj_starts.transpose(1, 0).contiguous()
                        hid_c_batch.append(hid_at_traj_starts)
                    hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else tuple(hid_c_batch)
                else:
                    hid_c_batch = None
                mb_hidden_states = (hid_a_batch, hid_c_batch)
        else:
            # Non-recurrent: standard [batch_size, num_steps, ...] format
            mb_obs_dict = {
                key: rollout_data["all_obs_dict"][key][micro_batch_inds]
                for key in rollout_data["all_obs_dict"].keys()
            }
            mb_masks = None

        mb_rollout_data = dict(
            micro_batch_inds=micro_batch_inds,
            mb_obs_dict=mb_obs_dict,
            mb_advantage=mb_advantage,
            mb_logprobs=mb_logprobs,
            mb_return=mb_return,
            mb_values=mb_values,
            mb_dones=mb_dones,
            mb_actions=mb_actions,
            mb_old_mu=mb_old_mu,
            mb_old_sigma=mb_old_sigma,
            mb_padding_mask=mb_padding_mask,
            mb_padding_mask_p1=mb_padding_mask_p1,
            episode_attnmask=episode_attnmask,
            mb_masks=mb_masks,
            mb_hidden_states=mb_hidden_states,
        )
        return mb_rollout_data

    def _forward_model(self, model, mb_rollout_data):
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        mb_actions = mb_rollout_data["mb_actions"]
        episode_attnmask = mb_rollout_data["episode_attnmask"]
        mb_masks = mb_rollout_data.get("mb_masks", None)
        mb_dones = mb_rollout_data["mb_dones"]
        mb_hidden_states = mb_rollout_data.get("mb_hidden_states", None)

        # Extract separate hidden states for actor and critic (like rsl_rl)
        actor_hidden_states = mb_hidden_states[0] if mb_hidden_states is not None else None
        critic_hidden_states = mb_hidden_states[1] if mb_hidden_states is not None else None

        # We should only do one forward pass for especially DDP model
        if self.compute_imgaug_bc_loss:
            results = model.forward(
                modes=["policy_w_and_wo_imgaug", "value"],
                input_kwargs=dict(
                    policy_w_and_wo_imgaug=dict(
                        obs_dict=mb_obs_dict,
                        actions=mb_actions,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=actor_hidden_states,
                        original_dones=mb_dones,
                    ),
                    value=dict(
                        obs_dict=mb_obs_dict,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=critic_hidden_states,
                        original_dones=mb_dones,
                    ),
                ),
            )
            policy_results = results["policy_w_and_wo_imgaug"]
        else:
            results = model.forward(
                modes=["policy", "value"],
                input_kwargs=dict(
                    policy=dict(
                        obs_dict=mb_obs_dict,
                        actions=mb_actions,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=actor_hidden_states,
                        original_dones=mb_dones,
                    ),
                    value=dict(
                        obs_dict=mb_obs_dict,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=critic_hidden_states,
                        original_dones=mb_dones,
                    ),
                ),
            )
            policy_results = results["policy"]

        return dict(
            policy_results=policy_results,
            value_results=results["value"],
        )

    def _compute_loss(self, forward_results, mb_rollout_data):
        ppo_loss_dict = self._compute_ppo_loss(forward_results, mb_rollout_data)

        loss = ppo_loss_dict["ppo_loss"] * self.config.get("ppo_loss_coef", 1.0)

        ret_dict = dict(
            ppo_loss_dict=ppo_loss_dict,
        )

        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_dict = self._compute_imgaug_bc_loss(forward_results, mb_rollout_data)
            loss += imgaug_bc_loss_dict["imgaug_bc_loss"] * self.config.imgaug_bc_loss_coef
            ret_dict["imgaug_bc_loss_dict"] = imgaug_bc_loss_dict

        ret_dict["loss"] = loss

        return ret_dict

    def _compute_ppo_loss(self, forward_results, mb_rollout_data):
        args = self.args
        optimizer = self.optimizer

        policy_results = forward_results["policy_results"]
        value_results = forward_results["value_results"]

        mb_old_mu = mb_rollout_data["mb_old_mu"]
        mb_old_sigma = mb_rollout_data["mb_old_sigma"]
        mb_values = mb_rollout_data["mb_values"]
        mb_return = mb_rollout_data["mb_return"]
        mb_logprobs = mb_rollout_data["mb_logprobs"]
        mb_advantage = mb_rollout_data["mb_advantage"]
        padding_mask = mb_rollout_data["mb_padding_mask"]
        padding_mask_p1 = mb_rollout_data["mb_padding_mask_p1"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]

        new_logprobs = policy_results["logprobs"]
        sigma_batch = policy_results["action_std"]
        mu_batch = policy_results["action_mean"]
        entropy_batch = policy_results["entropy"]
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma_batch / mb_old_sigma + 1.0e-5)
                + (torch.square(mb_old_sigma) + torch.square(mb_old_mu - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            local_kl_mean = torch.mean(kl)
            kl_mean = self.accelerator.gather(local_kl_mean).mean()
            self._adjust_learning_rate_based_on_kl(kl_mean, optimizer)

        # Forward a DDP model twice will cause the error: "one of the variables needed for gradient computation has been modified by an inplace operation"
        vpred = value_results.squeeze(-1)
        vpredclipped = torch.clamp(
            vpred,
            mb_values - args.cliprange_value,
            mb_values + args.cliprange_value,
        )
        vf_losses1 = torch.square(vpred - mb_return)
        vf_losses2 = torch.square(vpredclipped - mb_return)
        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        vf_loss = masked_mean(vf_loss_max, ~padding_mask_p1)
        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1)
        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = masked_mean(pg_loss_max, ~padding_mask)

        entropy_loss = -masked_mean(entropy_batch, ~padding_mask)

        loss = pg_loss + args.vf_coef * vf_loss + self.entropy_coef * entropy_loss

        return dict(
            ppo_loss=loss,
            # logging metrics
            local_kl_mean=local_kl_mean,
            pg_losses=pg_losses,
            pg_losses2=pg_losses2,
            pg_loss=pg_loss,
            vf_loss=vf_loss,
            entropy_loss=entropy_loss,
            ratio=ratio,
            vf_clipfrac=vf_clipfrac,
        )

    def _compute_imgaug_bc_loss(self, forward_results, mb_rollout_data):
        policy_results = forward_results["policy_results"]
        mu_batch = policy_results["action_mean"]

        action_mean_w_imgaug = policy_results["action_mean_w_imgaug"]
        imgaug_bc_loss = self.imgaug_bc_loss_fn(action_mean_w_imgaug, mu_batch.detach())

        return dict(
            imgaug_bc_loss=imgaug_bc_loss,
        )

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,
        mb_rollout_data,
    ):
        local_kl_mean = loss_dict["ppo_loss_dict"]["local_kl_mean"]
        pg_losses = loss_dict["ppo_loss_dict"]["pg_losses"]
        pg_losses2 = loss_dict["ppo_loss_dict"]["pg_losses2"]
        pg_loss = loss_dict["ppo_loss_dict"]["pg_loss"]
        vf_loss = loss_dict["ppo_loss_dict"]["vf_loss"]
        entropy_loss = loss_dict["ppo_loss_dict"]["entropy_loss"]
        weighted_ppo_loss = loss_dict["ppo_loss_dict"]["ppo_loss"] * self.config.get(
            "ppo_loss_coef", 1.0
        )
        ratio = loss_dict["ppo_loss_dict"]["ratio"]
        vf_clipfrac = loss_dict["ppo_loss_dict"]["vf_clipfrac"]

        padding_mask = mb_rollout_data["mb_padding_mask"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]

        self.approxkl_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = local_kl_mean
        pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask)
        self.pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_clipfrac
        self.pg_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_loss
        self.vf_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_loss
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss = loss_dict["imgaug_bc_loss_dict"]["imgaug_bc_loss"]
            self.imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = imgaug_bc_loss
            self.weighted_imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                self.config.imgaug_bc_loss_coef * imgaug_bc_loss
            )
        self.entropy_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = -entropy_loss
        self.weighted_ppo_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
            weighted_ppo_loss
        )
        self.vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_clipfrac
        self.ratio_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = ratio.mean()

    def _get_train_metrics(self):
        metrics = {}

        approxkl_avg = self.accelerator.gather_for_metrics(self.approxkl_stats).mean().item()

        metrics["policy/approxkl_avg"] = approxkl_avg
        metrics["policy/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_clipfrac_stats).mean().item()
        )
        metrics["loss/policy_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_loss_stats).mean().item()
        )
        if self.compute_imgaug_bc_loss:
            metrics["loss/imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.imgaug_bc_loss_stats).mean().item()
            )
            metrics["loss/weighted_imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.weighted_imgaug_bc_loss_stats)
                .mean()
                .item()
            )
        metrics["loss/value_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_loss_stats).mean().item()
        )
        metrics["loss/entropy_avg"] = (
            self.accelerator.gather_for_metrics(self.entropy_stats).mean().item()
        )
        metrics["loss/weighted_ppo_loss_avg"] = (
            self.accelerator.gather_for_metrics(self.weighted_ppo_loss_stats).mean().item()
        )
        metrics["val/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_clipfrac_stats).mean().item()
        )
        metrics["val/ratio"] = self.accelerator.gather_for_metrics(self.ratio_stats).mean().item()
        metrics["val/ratio_var"] = (
            self.accelerator.gather_for_metrics(self.ratio_stats).var().item()
        )
        metrics["objective/entropy"] = metrics["loss/entropy_avg"]

        return metrics

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                if dataloader is not None:
                    yield from dataloader
                else:
                    yield None

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        self._register_stats_buffer()
        model.train()

        # trainer state initialization
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # env
        obs_dict = self.env.reset_all()
        if self.config.get("init_at_random_ep_len", False):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase

            if isinstance(self.env, StagedTaskBase):
                self.env.time_in_stage_buf[:] = torch.randint_like(
                    self.env.time_in_stage_buf, high=int(self.env.max_stage_time[0])
                )
                self.env.actual_time_in_stage_buf[:] = self.env.time_in_stage_buf
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        for batch_idx in range(1, args.num_total_batches + 1):
            batch_start_time = time.time()
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            # update scheduled params
            if self.schedule_dict is not None:
                self.scheduled_params_dict = update_scheduled_params(
                    self, self.schedule_dict, self.state.global_step
                )

            reinit_sim_freq = self.env.config.get("reinit_sim_freq", 0)
            if reinit_sim_freq > 0 and (self.state.global_step + 1) % reinit_sim_freq == 0:
                self.env.reinit_sim()
            # DEBUG: Print model weights and learning rate (every 5 batches to reduce clutter)
            if batch_idx == 0:
                # Check if model is wrapped in DDP
                model_type = type(self.model).__name__
                if self.accelerator.process_index == 0:
                    print(f"\n{'='*80}")
                    print(f"[CRITICAL] Model type: {model_type}")
                    print(f"[CRITICAL] Num processes: {self.accelerator.num_processes}")
                    print(f"[CRITICAL] Distributed type: {self.accelerator.distributed_type}")
                    print(f"[CRITICAL] Use distributed: {self.accelerator.use_distributed}")
                    print("  >> Model should be DistributedDataParallel if multi-GPU!")
                    print(f"{'='*80}\n")

            with torch.no_grad():
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as model:
                    obs_dict = self._rollout_step(model, obs_dict)

                end_collection_time = time.time()
                collection_time = end_collection_time - batch_start_time

                rollout_data = self._get_rollout_data(obs_keys=obs_dict.keys())
                torch.cuda.empty_cache()
                gc.collect()

            model = self.model
            self._train_mode()
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                # Reset trajectory counter at start of each epoch for recurrent slicing
                self._current_first_traj = 0
                minibatch_idx = 0
                if self.ppo_shuffle_every_epoch or ppo_epoch_idx == 0:
                    # Disable shuffling for recurrent policies to keep contiguous env indices
                    policy_model = self.accelerator.unwrap_model(model).policy
                    if hasattr(policy_model, "is_recurrent") and policy_model.is_recurrent:
                        b_inds = torch.arange(args.local_batch_size, device=device)
                    else:
                        b_inds = torch.randperm(args.local_batch_size, device=device)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    microbatch_idx = 0
                    for micro_batch_start in range(
                        0, args.local_mini_batch_size, args.per_device_train_batch_size
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            mb_rollout_data = self._get_mb_rollout_data(
                                rollout_data, micro_batch_inds
                            )

                            forward_results = self._forward_model(model, mb_rollout_data)

                            loss_dict = self._compute_loss(forward_results, mb_rollout_data)

                            accelerator.backward(loss_dict["loss"])
                            self._gradient_clipping()
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                self._update_stats_buffer(
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    microbatch_idx,
                                    loss_dict,
                                    forward_results,
                                    mb_rollout_data,
                                )
                            del loss_dict, forward_results, mb_rollout_data
                            microbatch_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                if (
                    self.empty_cache_every_n_ppo_epoch > 0
                    and ppo_epoch_idx % self.empty_cache_every_n_ppo_epoch == 0
                ):
                    torch.cuda.empty_cache()
                    gc.collect()

            with torch.no_grad():
                learn_time = time.time() - end_collection_time
                eps = int(self.state.episode / (time.time() - start_time))

                metrics = {}
                train_metrics = self._get_train_metrics()
                metrics.update(train_metrics)
                metrics["eps"] = eps

                metrics["objective/rewards"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(
                            np.mean(self.state.rewbuffer) if len(self.state.rewbuffer) > 0 else 0,
                            dtype=torch.float32,
                        ).to(device)
                    )
                    .mean()
                    .item()
                )
                metrics["objective/length"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(
                            np.mean(self.state.lenbuffer) if len(self.state.lenbuffer) > 0 else 0,
                            dtype=torch.float32,
                        ).to(device)
                    )
                    .mean()
                    .item()
                )

                # Add goal reached rate from environment if available
                # The environment already calculates the average, so we can use it directly
                # if "recent_goal_reached_rate" in infos:
                #     metrics["objective/goal_reached_rate"] = self.accelerator.gather_for_metrics(torch.tensor(infos["recent_goal_reached_rate"]).to(device)).mean().item()
                # else:
                #     metrics["objective/goal_reached_rate"] = 0.0

                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                env_log_dict_local = self.episode_env_tensors.mean_and_clear()
                # Synchronize environment metrics across all GPUs to properly aggregate
                # metrics from all environments (e.g., 1024 envs/GPU * 8 GPUs = 8192 total envs)
                env_log_dict = {
                    k: (
                        self.accelerator.gather_for_metrics(
                            torch.tensor(v, dtype=torch.float32).to(device)
                        )
                        .mean()
                        .item()
                        if not isinstance(v, (int, float))
                        else self.accelerator.gather_for_metrics(
                            torch.tensor(v, dtype=torch.float32).to(device)
                        )
                        .mean()
                        .item()
                    )
                    for k, v in env_log_dict_local.items()
                }
                ep_infos = process_ep_infos(self.ep_infos, device)
                self.state.tot_timesteps += (
                    self.num_steps_per_env * self.env.num_envs * accelerator.num_processes
                )
                self.state.tot_time += collection_time + learn_time
                log_dict = {
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "tot_timesteps": self.state.tot_timesteps,
                    "tot_time": self.state.tot_time,
                    "it": self.state.global_step,
                    "fps": int(
                        self.num_steps_per_env
                        * self.env.num_envs
                        * accelerator.num_processes
                        / (collection_time + learn_time)
                    ),
                    "experiment_save_dir": self.args.output_dir,
                    "batch_idx": batch_idx,
                    "num_total_batches": args.num_total_batches,
                }

                for key, value in ep_infos.items():
                    log_dict[f"Episode/{key}"] = value

                # Add scheduled parameters to metrics
                for param_name, param_value in self.scheduled_params_dict.items():
                    log_dict[f"scheduled_params/{param_name}"] = param_value

                if hasattr(self.policy_model, "std"):
                    metrics["Policy/mean_noise_std"] = self.policy_model.std.mean().item()
                else:
                    metrics["Policy/mean_noise_std"] = 0.0

                metrics.update({f"Env/{k}": v for k, v in env_log_dict.items()})
                metrics.update(log_dict)

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1

                self.log(metrics)
                self.ep_infos.clear()

            self.lr_scheduler.step()

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            del (
                metrics,
                rollout_data,
            )
            torch.cuda.empty_cache()
            gc.collect()

            if self.control.should_training_stop:
                break

        if self.control.should_training_stop:
            return

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if wandb_run_exists():
            wandb.finish()

    def _eval_mode(self):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("eval")
        model.transform_eval()
        self.env.set_is_evaluating(is_evaluating=True, log_info=False)

    def _train_rollout_mode(self):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train_rollout")
        model.transform_eval()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(True, log_info=False)
        else:
            self.env.set_is_evaluating(False, log_info=False)

    def _train_mode(self):
        self.model.train()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train")
        model.transform_train()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(True, log_info=False)
        else:
            self.env.set_is_evaluating(False, log_info=False)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _gradient_clipping(self):
        # Gradient clipping
        args = self.args
        model = self.model
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            # deepspeed does its own clipping

            if is_sagemaker_mp_enabled() and args.fp16:
                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
            elif self.use_apex:
                # Revert to normal clipping otherwise, handling Apex or full precision
                _grad_norm = nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer),
                    args.max_grad_norm,
                )
            else:
                _grad_norm = self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

            if (
                is_accelerate_available()
                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                grad_norm = model.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
            else:
                grad_norm = _grad_norm

        return grad_norm

    def _compute_returns(self, values, last_values, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the
        environment based on the provided observations and policy state. It uses
        Generalized Advantage Estimation (GAE) to compute the advantages, which
        helps in reducing the variance of the policy gradient estimates.
        Args:
            values (torch.Tensor): The values for each step.
            last_values (torch.Tensor): The last values for the last step.
            policy_state_dict (dict): A dictionary containing the policy state
                          information, including 'values', 'dones',
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        device = self.accelerator.device
        advantage = 0

        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        dones = dones.to(device)
        rewards = rewards.to(device)

        returns = torch.zeros_like(values)

        num_steps = returns.shape[0]

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        if self.sync_advantage_normalization:
            # gather advantages from all processes before normalization
            advantages = self.accelerator.gather(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # ungather advantages
            advantages = advantages.reshape(
                self.accelerator.num_processes, -1, *advantages.shape[1:]
            )[self.accelerator.process_index].to(device)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _adjust_learning_rate_based_on_kl(self, kl_mean, optimizer):
        """Adjust the learning rate based on the KL divergence.

        This function implements a learning rate schedule that adjusts the learning rate
        based on the KL divergence between the current policy and the old policy.
        If the KL divergence is too high, the learning rate is decreased.
        If the KL divergence is too low, the learning rate is increased.

        Args:
            kl_mean (float): The mean KL divergence across all processes.
            optimizer (torch.optim.Optimizer): The optimizer to update.
        """
        if self.desired_kl is None:
            return

        if kl_mean > self.desired_kl * 2.0:
            new_lr = max(1e-5, self.args.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            new_lr = min(1e-2, self.args.learning_rate * 1.5)
        else:
            new_lr = self.args.learning_rate
        self.args.learning_rate = new_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.args.learning_rate

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to restore the state of model, optimizer, trainer etc.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.accelerator.device, weights_only=False
        )

        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        if "actor_model_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["actor_model_state_dict"])
        elif "policy_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint and model.value_model is not None:
            model.value_model.load_state_dict(checkpoint["value_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Update learning rate if available
            if "args" in checkpoint and hasattr(checkpoint["args"], "learning_rate"):
                self.args.learning_rate = checkpoint["args"].learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.args.learning_rate

        # Load learning rate scheduler state
        if (
            "lr_scheduler_state_dict" in checkpoint
            and checkpoint["lr_scheduler_state_dict"] is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if "env_state_dict" in checkpoint:
            self.env.load_env_state_dict(checkpoint["env_state_dict"])

        if "state" in checkpoint:
            for key, value in checkpoint["state"].__dict__.items():
                if key in ["cur_reward_sum", "cur_episode_length"]:
                    setattr(self, key, value)
                if key not in [
                    "stateful_callbacks",
                    "is_local_process_zero",
                    "is_world_process_zero",
                    "log_history",
                ]:
                    setattr(self.state, key, value)

        print(f"Loaded checkpoint from step {self.state.global_step}")
        return checkpoint

    def eval(self, save_dirpath: Optional[str] = None):
        if self.config.eval.save_videos or self.config.eval.save_trajectories:
            assert (
                save_dirpath is not None
            ), "save_dirpath must be provided if save_videos or save_trajectories is True"

        if self.config.eval.save_videos:
            if (
                self.env.config.obs.obs_dict.vision_obs[0] == "depth_image"
                and not self.env.config.simulator.config.cameras.stack_depth_image
            ):
                num_channels = 1
            else:
                num_channels = 3
            self.camera_resolution = self.env.config.simulator.config.cameras.camera_resolutions + [
                num_channels
            ]

        self._eval_mode()
        self.policy_model.eval_mode()
        self.policy_model.init_rollout()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

        # Initialize environment-based metrics tracking
        self.env.init_eval_metrics_tracking(self.accelerator.device)

        # Reuse the cur_reward_sum, cur_episode_length (will be reset at the end in case `train()` is called after `eval(_teacher)`)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.accelerator.device
        )

        # Keep old buffer attributes for backward compatibility with terminate_rollout()
        self.eval_rewbuffer = []
        self.eval_lenbuffer = []
        self.eval_goal_reached_buffer = []

        # Get evaluation mode configuration
        eval_num_envs_episodes = self.config.eval.get(
            "eval_num_envs_episodes", False
        )  # Default to one episode per env
        # Determine maximum number of episodes to collect
        if eval_num_envs_episodes:
            max_episodes = self.env.num_envs  # One episode per environment
        else:
            max_episodes = self.config.eval.get(
                "num_eval_episodes", self.env.num_envs
            )  # Use config or fallback to num_envs

        # Track which environments have completed their first episode (only used when eval_num_envs_episodes=True)
        if eval_num_envs_episodes:
            self.env_episode_completed = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.accelerator.device
            )

        if self.config.eval.save_videos:
            self.saved_episode_cnt = 0
            self.batch_frames = [[] for _ in range(self.env.num_envs)]

        if self.config.eval.save_trajectories:
            self.actor_obs_to_save = [[] for _ in range(self.env.num_envs)]
            self.student_obs_to_save = [[] for _ in range(self.env.num_envs)]
            self.actions_to_save = [[] for _ in range(self.env.num_envs)]
            self.reward_to_save = [[] for _ in range(self.env.num_envs)]
            self.done_to_save = [[] for _ in range(self.env.num_envs)]

        def terminate_rollout():
            if eval_num_envs_episodes:
                # Stop when all environments have completed their first episode
                return torch.all(self.env_episode_completed).item()
            else:
                # Original behavior: stop when we have collected max_episodes episodes total
                if self.config.eval.save_trajectories:
                    return self.saved_episode_cnt >= self.config.eval.num_save_episodes
                else:
                    return len(self.eval_goal_reached_buffer) >= self.config.eval.num_eval_episodes

        start_time = time.time()
        expected_total_len = max_episodes
        pbar = tqdm(total=expected_total_len)
        dones = torch.zeros(self.env.num_envs, device=self.accelerator.device)
        with torch.no_grad():
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as model:
                while not terminate_rollout():
                    action_state = self.policy_step(model.policy, obs_dict, cur_dones=dones)

                    last_obs_dict = deepcopy(obs_dict)
                    action_state["actions"] = action_state["action_mean"]
                    obs_dict, rewards, dones, infos = self.env.step(action_state)

                    if self.config.eval.save_videos:
                        self.batch_write_frame(last_obs_dict["vision_obs"])

                    if self.config.eval.save_trajectories:
                        self.batch_write_low_dim_obs(
                            actor_obs=last_obs_dict["actor_obs"],
                            student_obs=last_obs_dict["student_obs"],
                            actions=action_state["actions"],
                            rewards=rewards,
                            dones=dones,
                        )

                    for obs_key in obs_dict.keys():
                        obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

                    rewards, dones = rewards.to(self.accelerator.device), dones.to(
                        self.accelerator.device
                    )

                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1

                    # Update environment metrics per step
                    self.env.update_eval_metrics_per_step(infos)

                    new_ids = (dones > 0).nonzero(as_tuple=False)

                    if len(new_ids) > 0:
                        if eval_num_envs_episodes:
                            # Only collect data from environments that haven't completed their first episode yet
                            valid_new_ids = new_ids[~self.env_episode_completed[new_ids][:, 0]]
                        else:
                            # Original behavior: collect data from all completed episodes
                            valid_new_ids = new_ids

                        if len(valid_new_ids) > 0:
                            if self.config.eval.save_videos:
                                self.batch_reset_data_writer(valid_new_ids, save_dirpath)

                            # Process episode completions using environment callback
                            self.env.process_eval_episode_completions(
                                valid_new_ids, self.cur_reward_sum, self.cur_episode_length
                            )

                            # Also populate old buffer attributes for backward compatibility with terminate_rollout()
                            self.eval_rewbuffer.extend(
                                self.cur_reward_sum[valid_new_ids][:, 0].cpu().numpy().tolist()
                            )
                            self.eval_lenbuffer.extend(
                                self.cur_episode_length[valid_new_ids][:, 0].cpu().numpy().tolist()
                            )
                            self.eval_goal_reached_buffer.extend(
                                self.env.eval_metrics["episode_goal_reached"][valid_new_ids][:, 0]
                                .cpu()
                                .numpy()
                                .tolist()
                            )

                            if eval_num_envs_episodes:
                                # Mark these environments as having completed their first episode
                                self.env_episode_completed[valid_new_ids] = True
                                # Update progress bar with newly completed episodes
                                pbar.update(len(valid_new_ids))
                            else:
                                # Original behavior: update progress bar with collected episodes
                                if self.config.eval.save_trajectories:
                                    pbar.update(self.saved_episode_cnt - pbar.n)
                                else:
                                    pbar.update(
                                        len(self.env.eval_metrics["goal_reached_buffer"]) - pbar.n
                                    )

                        # Reset all environments that finished (regardless of whether we collected data)
                        self.cur_reward_sum[new_ids] = 0
                        self.cur_episode_length[new_ids] = 0

                        # Reset environment episode tracking
                        self.env.reset_eval_episode_tracking(new_ids)

                    # TODO: 0: make sure this is compatiable with all use cases
                    self.callback_handler.on_step_end(self.args, self.state, self.control)

                torch.cuda.empty_cache()
                gc.collect()

        pbar.close()
        end_time = time.time()

        # Get evaluation summary from environment (includes class-wise metrics)
        eval_dict = self.env.get_eval_metrics_summary()
        eval_dict["eval_time"] = end_time - start_time

        print(f"Eval time: {end_time - start_time} seconds")
        if self.config.eval.save_trajectories:
            print("-" * 30)
            print(f"Saved {self.saved_episode_cnt} episodes")
        if self.config.eval.save_videos:
            print(f"Please check {save_dirpath}")

        # Reset env (in the outer loop because we need the obs_dict), cur_reward_sum, cur_episode_length
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.accelerator.device
        )

        return eval_dict

    def batch_write_frame(self, vision_obs):
        """
        vision_obs: [B, H*W*C], [0, 1/255], float32 (The range is a bug)
        This function is for eval time, save all envs' frames,
            as opposed to write_frame, which is for training time, only save the frames for self.visualize_env_idx
        """
        flattened_rgb_images = vision_obs.clone()

        batch_size = flattened_rgb_images.shape[0]

        rgb_images = flattened_rgb_images.reshape(
            batch_size, *self.camera_resolution
        )  # [B, H, W, C]

        if not getattr(self, "_grpo_dbg_ranged", False):
            self._grpo_dbg_ranged = True
            print(
                f"[VISION_OBS RANGE] min={rgb_images.min().item():.3f} "
                f"max={rgb_images.max().item():.3f} mean={rgb_images.mean().item():.3f}"
            )

        # vision_obs is stored ImageNet-normalized ((x - mean) / std, values ~[-2.5, 2.5]);
        # a raw *255 wraps mod 256 into psychedelic bands. Un-normalize back to [0, 1]
        # first so the saved video shows the actual camera image the policy sees.
        try:
            cams = self.env.config.simulator.config.cameras
            mean = torch.tensor(cams.image_mean, device=rgb_images.device, dtype=rgb_images.dtype)
            std = torch.tensor(cams.image_std, device=rgb_images.device, dtype=rgb_images.dtype)
            rgb_images = rgb_images * std + mean
        except Exception:
            pass

        # To uint8
        rgb_images = (rgb_images.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # [B, H, W, C]

        # Append to frames list
        for i in range(self.env.num_envs):
            self.batch_frames[i].append(rgb_images[i])  # [H, W, C]

    def batch_write_low_dim_obs(
        self,
        actor_obs,
        student_obs,
        actions,
        rewards,
        dones,
    ):
        """
        actor_obs: [B, dim_actor_obs]
        student_obs: [B, dim_student_obs]
        actions: [B, dim_actions]
        rewards: [B]
        dones: [B]
        """
        for i in range(self.env.num_envs):
            self.actor_obs_to_save[i].append(actor_obs[i].cpu().numpy())
            self.student_obs_to_save[i].append(student_obs[i].cpu().numpy())
            self.actions_to_save[i].append(actions[i].cpu().numpy())
            self.reward_to_save[i].append(rewards[i].item())
            # Convert to bool
            self.done_to_save[i].append(bool(dones[i].item()))

    def batch_reset_data_writer(self, env_indices, save_dirpath):
        """
        :param env_indices: list of env indices that have finished the episode
        """
        for env_idx in env_indices:
            do_not_save = False

            cur_reward_sum = self.cur_reward_sum[env_idx].item()
            # Get goal_reached status from environment metrics (used for the
            # filename + optional success-only filtering). Be defensive: the
            # per-env lookup can be out of range depending on how eval_metrics is
            # populated, and we still want the video regardless.
            goal_reached = 0
            try:
                gr = None
                if getattr(self.env, "goal_reached_buf", None) is not None:
                    gr = self.env.goal_reached_buf[env_idx]
                elif (
                    hasattr(self.env, "eval_metrics")
                    and "episode_goal_reached" in self.env.eval_metrics
                ):
                    gr = self.env.eval_metrics["episode_goal_reached"][env_idx]
                goal_reached = int(gr.item()) if hasattr(gr, "item") else int(bool(gr))
            except (IndexError, KeyError, TypeError, RuntimeError, AttributeError):
                goal_reached = 0

            if self.config.eval.save_goal_reached_only:
                if goal_reached == 0:
                    do_not_save = True

            if np.random.rand() > self.config.eval.video_save_prob:
                do_not_save = True

            if do_not_save:
                # Simply empty the buffers
                self.batch_frames[env_idx] = []
                if self.config.eval.save_trajectories:
                    self.actor_obs_to_save[env_idx] = []
                    self.student_obs_to_save[env_idx] = []
                    self.actions_to_save[env_idx] = []
                    self.reward_to_save[env_idx] = []
                    self.done_to_save[env_idx] = []
                continue

            # Video saving
            video_save_dirpath = Path(save_dirpath) / "videos"

            save_filename = f"eps{self.saved_episode_cnt}_env{env_idx.item()}_len{len(self.batch_frames[env_idx])}_reward{cur_reward_sum:.2f}_goal{int(goal_reached)}_rank{self.accelerator.process_index}"
            video_path = video_save_dirpath / f"{save_filename}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)

            video_tensor = torch.stack(self.batch_frames[env_idx][1:])  # [T, H, W, C]

            fps = 20
            torchvision.io.write_video(str(video_path), video_tensor, fps=fps, video_codec="h264")

            # print(f"Video saved to {video_path}")

            self.batch_frames[env_idx] = []
            self.saved_episode_cnt += 1

            # Low dim data saving
            if self.config.eval.save_trajectories:
                low_dim_data_save_dirpath = Path(save_dirpath) / "data"
                low_dim_data_save_dirpath.mkdir(parents=True, exist_ok=True)

                save_data = {
                    "observation.state": self.student_obs_to_save[env_idx][1:],
                    "actor_obs": self.actor_obs_to_save[env_idx][1:],
                    "action": self.actions_to_save[env_idx][1:],
                    "reward": self.reward_to_save[env_idx][1:],
                    "done": self.done_to_save[env_idx][1:],
                }

                episode_length = len(video_tensor)
                for key, value in save_data.items():
                    assert len(value) == episode_length, f"{key}: {len(value)} != {episode_length}"

                save_data["timestamp"] = np.arange(episode_length) / fps

                data_save_path = low_dim_data_save_dirpath / f"{save_filename}.parquet"

                df = pd.DataFrame(save_data)
                df.to_parquet(data_save_path)

                # Empty the buffers
                self.actor_obs_to_save[env_idx] = []
                self.student_obs_to_save[env_idx] = []
                self.actions_to_save[env_idx] = []
                self.reward_to_save[env_idx] = []
                self.done_to_save[env_idx] = []

            if self.saved_episode_cnt >= self.config.eval.num_save_episodes:
                break

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @property
    def inference_model(self):
        return {"actor": self.model.policy, "critic": self.model.value_model}
