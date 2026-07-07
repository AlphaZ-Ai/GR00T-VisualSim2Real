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
Evaluation script for trained RL agents (teacher or student).

Loads a checkpoint and its associated training config, sets up the environment
and model, optionally exports the policy as ONNX, then runs evaluation episodes.

Usage:
    python groot/rl/eval_agent_trl.py +checkpoint=<path_to_checkpoint.pt> [+num_envs=1]
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()


def process_output_dim_in_config(config):
    """Process and adapt output dimensions for actor and teacher_actor backbones.

    When output_dim is set to -1 in the config, this function auto-calculates
    the correct dimension based on the homie command keys.
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


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # --- Logging setup ---
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    # --- Load and merge training config from checkpoint directory ---
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    # Resume wandb run if meta.yaml exists
    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path, "r"))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    # --- Setup Isaac Sim ---
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
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher

        import argparse

        import isaaclab

        parser = argparse.ArgumentParser(description="Evaluate an RL agent.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = (
            config.simulator.config.cameras.enable_cameras
            or config.simulator.config.get("render_results", False)
        )
        args_cli.headless = config.headless

        # Copy headless rendering kit file if needed
        dest_path = Path(isaaclab.__file__).resolve().parent / "apps"
        current_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
        if args_cli.enable_cameras and args_cli.headless:
            source_file = current_file_dir_path / "apps/phc.isaaclab.python.headless.rendering.kit"
            shutil.copy(source_file, dest_path)
            args_cli.experience = dest_path / "phc.isaaclab.python.headless.rendering.kit"

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    # --- Imports that must come after Isaac Sim initialization ---
    from accelerate import Accelerator
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.agents.modules.ppo_modules import (
        PPOCritic,
        PPOStateActor,
        PPOStateActorFixSigma,
    )
    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config

    # --- Config processing ---
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)

    # Set rendering output directory
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    if config.env.config.get("save_rendering_dir", None) is None:
        if config.simulator.config.get("render_results", False):
            from datetime import datetime

            parent_dir = checkpoint.parent.name
            config.env.config.save_rendering_dir = str(
                Path("logs_rl") / "renderings" / datetime.now().strftime("%Y-%m-%d") / parent_dir
            )
        else:
            config.env.config.save_rendering_dir = str(
                checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
            )
    config.env.config.ckpt_dir = str(checkpoint.parent)

    # --- Setup TRL training args (reused for eval) ---
    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)
    if "eval_output_dir" in config:
        training_args.eval_output_dir = config.eval_output_dir

    ref_model = None
    value_model = None

    # --- Accelerator and seeding ---
    accelerator = Accelerator()
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

    # --- Create environment ---
    env = instantiate(config=config.env, device=device)

    # --- Build policy and value models ---
    process_output_dim_in_config(config)

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy = instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
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
        if getattr(config.algo.config, "use_dagger", False):
            module_dim_dict = getattr(config.algo.config, "module_dim", {})
            policy = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
            ref_model = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.teacher_actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
        else:
            policy = PPOStateActor(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                input_key="actor_obs",
                init_noise_std=config.algo.config.init_noise_std,
            ).to(device)
            value_model = PPOCritic(
                env.config.robot.algo_obs_dim_dict,
                config.algo.config.module_dict.critic,
            ).to(device)

    accelerator.wait_for_everyone()

    # --- Callbacks ---
    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    # --- Build trainer (loads checkpoint weights) ---
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        value_model=value_model,
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        accelerator=accelerator,
    )

    # --- Optional ONNX export (only with single env) ---
    EXPORT_ONNX = config.num_envs == 1
    checkpoint_path = str(checkpoint)

    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{trainer.state.global_step:06d}.onnx"
    new_cp_path = (
        f"{os.path.dirname(config.checkpoint)}/model_step_{trainer.state.global_step:06d}.pt"
    )

    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if EXPORT_ONNX:
        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gr00t.rl.utils.inference_helpers import (
            export_policy_as_onnx,
            export_policy_CNN_as_onnx,
            export_policy_CNN_as_onnx_with_obj_pred,
            export_policy_z_as_onnx,
        )

        algo = trainer
        example_obs_dict = algo.get_example_obs()

        if config.env.config.get("use_z", False):
            export_policy_z_as_onnx(
                algo.inference_model,
                env,
                exported_policy_path,
                exported_onnx_name,
                example_obs_dict,
            )
        elif config.simulator.config.cameras.enable_cameras:
            if hasattr(config.algo.config.actor.backbone, "obj_pred_mlp"):
                export_policy_CNN_as_onnx_with_obj_pred(
                    algo.inference_model,
                    env,
                    exported_policy_path,
                    exported_onnx_name,
                    example_obs_dict,
                )
            else:
                export_policy_CNN_as_onnx(
                    algo.inference_model,
                    env,
                    exported_policy_path,
                    exported_onnx_name,
                    example_obs_dict,
                )
        else:
            export_policy_as_onnx(
                algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict
            )
        logger.info(
            f"Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}"
        )

    if config.get("export_onnx_only", False):
        exit()

    # --- Run evaluation ---
    # Pass a save dir so `save_videos` can write per-episode camera-view mp4s
    # (the distill trainer's eval asserts save_dirpath is set when save_videos).
    save_dirpath = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    trainer.eval(save_dirpath=save_dirpath)
    logger.info("Finished evaluation")
    os._exit(0)


if __name__ == "__main__":
    main()
