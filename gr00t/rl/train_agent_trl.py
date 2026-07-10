# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for RL agents using TRL (Transformer Reinforcement Learning).

Supports two training modes:
  - Teacher training: Standard PPO with state-based actor-critic
  - Student training: DAgger-based distillation from a teacher policy into a
    vision-based student policy

Usage:
    # Teacher training
    HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 groot/rl/train_agent_trl.py \\
        +exp=loco_manip/walk_stand_place_grasp_turn_homie num_envs=48

    # Student training (distillation with vision)
    HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 groot/rl/train_agent_trl.py \\
        +exp=loco_manip/wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay num_envs=8 headless=True
"""

import glob
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()


def seeding(seed=0, torch_deterministic=False):
    """Set random seeds for reproducibility across all libraries."""
    import torch

    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def resume_training(config):
    """Find the latest checkpoint and update config for training resumption."""
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
    else:
        experiment_dir_base = os.path.join(
            config.base_dir, config.project_name, config.experiment_name
        )
        last_existing_checkpoint = sorted(
            glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt"))
        )[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def auto_calculate_vision_feature_dim(config):
    """Auto-calculate vision_feature_dim based on history_length and temporal aggregation mode.

    For concatenation mode: vision_feature_dim = base_vision_feature_dim * history_length
    For attention mode: vision_feature_dim = base_vision_feature_dim
    """
    if not (hasattr(config, "history_length") and hasattr(config, "base_vision_feature_dim")):
        return

    use_attention = config.algo.config.get("actor", {}).get("use_temporal_attention", False)

    if not use_attention:
        calculated_dim = config.base_vision_feature_dim * config.history_length
    else:
        calculated_dim = config.base_vision_feature_dim

    if not hasattr(config, "vision_feature_dim") or config.vision_feature_dim == -1:
        config.vision_feature_dim = calculated_dim
        mode = "concatenation" if not use_attention else "attention"
        logger.info(
            f"Auto-calculated vision_feature_dim for {mode} mode: "
            f"{config.base_vision_feature_dim} * "
            f"{config.history_length if not use_attention else 1} = {calculated_dim}"
        )


def process_output_dim_in_config(config):
    """Process and adapt output dimensions for actor/teacher_actor backbones.

    When output_dim is set to -1, auto-calculates from homie command keys.
    """

    def calculate_homie_output_dim():
        output_dim = 0
        for key in config.obs["homie_command_keys"].keys():
            output_dim += len(config.obs["homie_command_default"][key])
        return output_dim

    def adapt_backbone_output_dim(backbone_config, config_name=""):
        try:
            if hasattr(backbone_config, "module_config_dict"):
                if backbone_config.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.module_config_dict.output_dim = [output_dim]
                    return True
            elif hasattr(backbone_config, "mlp_module") and hasattr(
                backbone_config.mlp_module, "module_config_dict"
            ):
                if backbone_config.mlp_module.module_config_dict.output_dim[0] == -1:
                    output_dim = calculate_homie_output_dim()
                    backbone_config.mlp_module.module_config_dict.output_dim = [output_dim]
                    return True
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not adapt {config_name} backbone output_dim: {e}")
        return False

    if (
        config.algo.config.get("use_new_actor_critic", False)
        and hasattr(config.algo.config, "actor")
        and hasattr(config.algo.config.actor, "backbone")
    ):
        adapt_backbone_output_dim(config.algo.config.actor.backbone, "actor")
        if (
            getattr(config.algo.config, "use_dagger", False)
            and hasattr(config.algo.config, "teacher_actor")
            and hasattr(config.algo.config.teacher_actor, "backbone")
        ):
            adapt_backbone_output_dim(config.algo.config.teacher_actor.backbone, "teacher_actor")


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # Auto-calculate vision_feature_dim for history-based vision models
    auto_calculate_vision_feature_dim(config)

    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

    # --- Distributed training setup ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    # Resume wandb run if meta.yaml exists from a previous run
    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path, "r"))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    # --- Isaac Sim setup ---
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2")
            from omni.isaac.lab.app import AppLauncher

        import argparse

        import isaaclab

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = (
            config.simulator.config.cameras.enable_cameras or config.simulator.config.render_results
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Copy headless rendering kit file if cameras are enabled in headless mode
        dest_path = Path(isaaclab.__file__).resolve().parent / "apps"
        current_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
        if args_cli.enable_cameras and args_cli.headless:
            source_file = current_file_dir_path / "apps/phc.isaaclab.python.headless.rendering.kit"
            shutil.copy(source_file, dest_path)
            args_cli.experience = dest_path / "phc.isaaclab.python.headless.rendering.kit"

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    # --- Imports that must come after Isaac Sim initialization ---
    import wandb

    from gr00t.rl.agents.base_algo.base_algo import BaseAlgo  # noqa: E402, F401
    from gr00t.rl.agents.modules.ppo_modules import (
        PPOCritic,
        PPOStateActor,
        PPOStateActorFixSigma,
        PPOVisionStateActorFixSigma,
        PPOVisionStateActorWithTransformFixSigma,
    )
    from gr00t.rl.envs.base_task.base_task import BaseTask  # noqa: E402, F401
    from gr00t.rl.trl.utils.common import custom_instantiate, wandb_run_exists
    from gr00t.rl.utils.helpers import pre_process_config
    from gr00t.rl.utils.logging import HydraLoggerBridge

    # --- Logging setup ---
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # --- Wandb setup ---
    # resolve=False preserves interpolations for inference-time overrides
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    pre_process_config(config)

    # --- Initialize environment ---
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    config.env.config.experiment_dir = str(Path(config.experiment_dir))
    env = custom_instantiate(config.env, device=device, _resolve=False)

    # --- Build policy and value models ---
    ref_model = None
    value_model = None
    process_output_dim_in_config(config)

    if config.algo.config.get("use_new_actor_critic", False):
        # New-style actor-critic: instantiated from config with backbone specification
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        ).to(device)
        if getattr(config.algo.config, "use_dagger", False):
            ref_model = instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
                input_key="teacher_obs",
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False) and hasattr(
            config.algo.config, "critic"
        ):
            value_model = instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            ).to(device)
    else:
        # Legacy actor-critic: manually constructed from module_dict
        algo_obs_dim_dict = env.config.robot.algo_obs_dim_dict
        actions_dim = env.config.robot.actions_dim

        if getattr(config.algo.config, "use_dagger", False):
            # DAgger student: vision-based policy + state-based teacher
            module_dim_dict = getattr(config.algo.config, "module_dim", {})
            if getattr(config.algo.config, "use_data_transform", False):
                config.max_state_dim = algo_obs_dim_dict["actor_obs"]
                config.max_action_dim = actions_dim
                policy = PPOVisionStateActorWithTransformFixSigma(
                    obs_dim_dict=algo_obs_dim_dict,
                    mlp_module_config_dict=config.algo.config.module_dict.actor,
                    vision_module_config_dict=config.algo.config.module_dict.encoder,
                    num_actions=actions_dim,
                    module_dim_dict=module_dim_dict,
                    input_key="actor_obs",
                    transforms_cfg=config.transforms,
                    image_resolution=config.image_resolution,
                    use_data_augmentation=config.use_data_augmentation,
                ).to(device)
            else:
                policy = PPOVisionStateActorFixSigma(
                    obs_dim_dict=algo_obs_dim_dict,
                    mlp_module_config_dict=config.algo.config.module_dict.actor,
                    vision_module_config_dict=config.algo.config.module_dict.encoder,
                    num_actions=actions_dim,
                    input_key="actor_obs",
                    module_dim_dict=module_dim_dict,
                ).to(device)

            ref_model = PPOStateActorFixSigma(
                obs_dim_dict=algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.teacher_actor,
                num_actions=actions_dim,
                input_key="teacher_obs",
                module_dim_dict=module_dim_dict,
            ).to(device)
        else:
            # Standard PPO: state-based actor with adaptive noise + critic
            policy = PPOStateActor(
                obs_dim_dict=algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=actions_dim,
                input_key="actor_obs",
                init_noise_std=config.algo.config.init_noise_std,
            ).to(device)

            value_model = PPOCritic(algo_obs_dim_dict, config.algo.config.module_dict.critic).to(
                device
            )

    accelerator.wait_for_everyone()

    # --- Callbacks ---
    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    # --- Save config and initialize trainer ---
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # --- Training loop ---
    trainer.train()

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
