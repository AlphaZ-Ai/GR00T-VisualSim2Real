# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import os
from typing import Optional

from pxr import Sdf, UsdGeom

try:
    with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
        DEFAULT_ISAACSIM_VERSION = f.read().strip()
except FileNotFoundError:
    DEFAULT_ISAACSIM_VERSION = "4.5"


if DEFAULT_ISAACSIM_VERSION == "4.5":

    ### IsaacSim 4.5

    import builtins
    import copy
    import importlib.util
    import os
    import sys

    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    import isaaclab.terrains as terrain_gen
    import numpy as np
    import omni
    import torch

    # Common imports for all IsaacSim versions
    import torch.nn.functional as F
    from gr00t.rl.simulator.isaacsim.isaaclab_viewpoint_camera_controller import (
        ViewportCameraController,
    )
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        DeformableObject,
        RigidObject,
    )
    from isaaclab.envs import ViewerCfg
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.managers import EventManager, SceneEntityCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import (
        ContactSensor,
        ContactSensorCfg,
        FrameTransformer,
        FrameTransformerCfg,
        RayCaster,
        RayCasterCfg,
        TiledCamera,
        TiledCameraCfg,
        patterns,
    )

    # from gr00t.rl.simulator.isaaclab_cfg import IsaacLabCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from isaaclab.utils.timer import Timer
    from loguru import logger
    from pxr import UsdShade

    from gr00t.rl.simulator.base_simulator.base_simulator import BaseSimulator
    from gr00t.rl.simulator.isaacsim.event_cfg import EventCfg
    from gr00t.rl.simulator.isaacsim.events import (
        cleanup_usd_memory,
        optimize_material_randomization_memory,
        randomize_body_com,
        randomize_dome_light,
        randomize_floor_material,
        randomize_robot_material,
        randomize_table_material,
    )
    from gr00t.rl.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG
    from gr00t.rl.utils.helpers import dict_to_true_list

elif DEFAULT_ISAACSIM_VERSION == "4.2":
    #### IsaacSim 4.2

    import builtins
    import copy
    import importlib.util
    import os
    import sys

    import numpy as np
    import omni
    import omni.isaac.lab.envs.mdp as mdp
    import omni.isaac.lab.sim as sim_utils
    import omni.isaac.lab.terrains as terrain_gen
    import torch
    import torch.nn.functional as F
    from gr00t.rl.simulator.isaacsim.isaaclab_viewpoint_camera_controller import (
        ViewportCameraController,
    )
    from loguru import logger
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.assets import Articulation, ArticulationCfg
    from omni.isaac.lab.assets.rigid_object import RigidObject
    from omni.isaac.lab.envs import ViewerCfg
    from omni.isaac.lab.managers import EventManager, SceneEntityCfg
    from omni.isaac.lab.managers import EventTermCfg as EventTerm
    from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.lab.sensors import (
        ContactSensor,
        ContactSensorCfg,
        RayCaster,
        RayCasterCfg,
        TiledCamera,
        TiledCameraCfg,
        patterns,
    )

    # from gr00t.rl.simulator.isaaclab_cfg import IsaacLabCfg
    from omni.isaac.lab.sim import PhysxCfg, SimulationCfg, SimulationContext
    from omni.isaac.lab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from omni.isaac.lab.utils.timer import Timer
    from pxr import UsdShade

    from gr00t.rl.simulator.base_simulator.base_simulator import BaseSimulator
    from gr00t.rl.simulator.isaacsim.event_cfg import EventCfg
    from gr00t.rl.simulator.isaacsim.events import (
        cleanup_usd_memory,
        optimize_material_randomization_memory,
        randomize_body_com,
        randomize_dome_light,
        randomize_floor_material,
        randomize_robot_material,
        randomize_table_material,
    )
    from gr00t.rl.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG
    from gr00t.rl.utils.helpers import dict_to_true_list

else:
    raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")


def list_mdl_files_recursive(folder_path, mdl_files):
    # List all items in the current folder
    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Failed to list: {folder_path} (error {result})")
        return

    for entry in entries:
        full_path = os.path.join(folder_path, entry.relative_path)
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            # It's a folder → recurse into it
            list_mdl_files_recursive(full_path, mdl_files)
        else:
            # It's a file → check extension
            if entry.relative_path.endswith(".mdl"):
                mdl_files.append(full_path)


class IsaacSim(BaseSimulator):
    def __init__(self, config, device, **kwargs):
        super().__init__(config, device)

        # ==================== Telemetry ====================
        # Capture rank information (both torch.distributed and generic environment variables)
        self.rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "N/A"))
        try:
            self.rank = int(self.rank)
        except (ValueError, TypeError):
            # keep as string if not an int
            pass
        logger.info(
            f"[TELEMETRY] Rank {self.rank}: Entering IsaacSim.__init__ on device {self.sim_device}"
        )
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            logger.info(
                f"[TELEMETRY] Rank {self.rank}: CUDA device idx={dev_idx}, name={torch.cuda.get_device_name(dev_idx)}, "
                f"memory reserved={torch.cuda.memory_reserved(dev_idx)/1e6:.1f}MB, memory allocated={torch.cuda.memory_allocated(dev_idx)/1e6:.1f}MB"
            )
        # ====================================================

        if "task" in config:
            self.task_type = config.task.type  # {"manipulation", "locomotion"}
            self.task_name = config.task.name
            self.task_config = config.task

        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand

        # for reset event manager
        self.num_envs = self.simulator_config.scene.num_envs
        self.device = self.sim_device

        # callback to allow env to tap into the simulator scene creation
        if "scene_creation_callback" in kwargs:
            self.scene_creation_callback = kwargs["scene_creation_callback"]
        else:
            self.scene_creation_callback = None

        sim_config: SimulationCfg = SimulationCfg(
            dt=1.0 / self.simulator_config.sim.fps,
            render_interval=self.simulator_config.sim.render_interval,
            device=self.sim_device,
            physx=PhysxCfg(
                solver_type=self.simulator_config.sim.physx.solver_type,
                max_position_iteration_count=255,
                max_velocity_iteration_count=255,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
        )

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        # self.sim.set_camera_view([0.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1./self.simulator_config.sim.fps}")
        logger.info(
            f"\tRendering step-size   : {1./self.simulator_config.sim.fps * self.simulator_config.sim.substeps}"
        )

        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.simulator_config.scene.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            replicate_physics=self.simulator_config.scene.replicate_physics,
        )
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()

        # disable gravity for arms
        if self.simulator_config.get("disable_gravity_for_arms", False):
            arm_body_names = self.robot_config.arm_body_names
            arm_body_ids = torch.tensor(
                [self.robot_config.body_names.index(name) for name in arm_body_names],
                dtype=torch.long,
                device="cpu",
            )
            gravity_flags = (
                self.scene.articulations["robot"].root_physx_view.get_disable_gravities().clone()
            )
            gravity_flags[:, arm_body_ids] = 1
            self.scene.articulations["robot"].root_physx_view.set_disable_gravities(
                gravity_flags, arm_body_ids
            )

        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros(
            (self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu"
        )

        self.event_types = set()
        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )
            self.event_types.add("startup")

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": self.domain_rand_config["friction_range"],
                    "dynamic_friction_range": self.domain_rand_config["friction_range"],
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 64,
                },
            )

            self.event_types.add("startup")

        if self.domain_rand_config.get("randomize_base_com", False):

            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            "torso_link",
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor(
                            [
                                float(self.domain_rand_config["base_com_range"]["x"][0]),
                                float(self.domain_rand_config["base_com_range"]["y"][0]),
                                float(self.domain_rand_config["base_com_range"]["z"][0]),
                            ]
                        ),
                        torch.tensor(
                            [
                                float(self.domain_rand_config["base_com_range"]["x"][1]),
                                float(self.domain_rand_config["base_com_range"]["y"][1]),
                                float(self.domain_rand_config["base_com_range"]["z"][1]),
                            ]
                        ),
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )
            self.event_types.add("startup")

        if self.domain_rand_config.get("randomize_dome_light", False):
            texture_folder_paths = self.domain_rand_config["dome_light_texture_folder_paths"]
            texture_list = []
            for texture_folder_path in texture_folder_paths:
                _, all_list = omni.client.list(texture_folder_path)
                texture_list.extend(
                    [
                        os.path.join(texture_folder_path, item.relative_path)
                        for item in all_list
                        if item.relative_path.endswith(".hdr")
                    ]
                )
            self.events_cfg.random_dome_light = EventTerm(
                func=randomize_dome_light,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("dome_light"),
                    "texture_files": texture_list,
                    "intensity_min_max": tuple(
                        self.domain_rand_config["dome_light_intensity_range"]
                    ),
                    "yaw_min_max": tuple(self.domain_rand_config["dome_light_yaw_range"]),
                    "intensity_distribution": self.domain_rand_config[
                        "dome_light_intensity_distribution"
                    ],
                    "yaw_distribution": self.domain_rand_config["dome_light_yaw_distribution"],
                },
                min_step_count_between_reset=20,
            )
            self.event_types.add("reset")

        if self.domain_rand_config.get("randomize_robot_material", False):
            self.events_cfg.random_robot_material = EventTerm(
                func=randomize_robot_material,
                mode="reset",
                params={
                    "shader_prim_paths": self.domain_rand_config[
                        "robot_material_shader_prim_paths"
                    ],
                    "roughness_range": tuple(
                        self.domain_rand_config["robot_material_roughness_range"]
                    ),
                    "metallic_range": tuple(
                        self.domain_rand_config["robot_material_metallic_range"]
                    ),
                    "specular_range": tuple(
                        self.domain_rand_config["robot_material_specular_range"]
                    ),
                    "distribution": self.domain_rand_config["robot_material_distribution"],
                },
                min_step_count_between_reset=20,
            )
            self.event_types.add("reset")

        if self.domain_rand_config.get("randomize_floor_material", False):
            material_folder_paths = self.domain_rand_config["floor_material_folder_paths"]
            material_list = []
            for material_folder_path in material_folder_paths:
                _, all_list = omni.client.list(material_folder_path)
                material_list.extend(
                    [
                        os.path.join(material_folder_path, item.relative_path)
                        for item in all_list
                        if item.relative_path.endswith(".mdl")
                    ]
                )
            material_count = 0
            for material_path in material_list:
                subidentifiers = asyncio.run(
                    omni.kit.material.library.get_subidentifier_from_mdl(material_path)
                )
                for subidentifier in subidentifiers:
                    success, result = omni.kit.commands.execute(
                        "CreateMdlMaterialPrimCommand",
                        mtl_url=str(material_path),
                        mtl_name=str(subidentifier),
                        mtl_path=f"/World/Looks/FloorMaterials/FloorMaterial_{material_count}",
                    )
                    shader_prim = (
                        omni.usd.get_context()
                        .get_stage()
                        .GetPrimAtPath(
                            f"/World/Looks/FloorMaterials/FloorMaterial_{material_count}/Shader"
                        )
                    )
                    shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(
                        True
                    )
                    shader_prim.CreateAttribute(
                        "inputs:world_or_object", Sdf.ValueTypeNames.Bool
                    ).Set(True)
                    material_count += 1

            self.events_cfg.random_floor_material = EventTerm(
                func=randomize_floor_material,
                mode="reset",
                params={
                    "material_count": material_count,
                },
                min_step_count_between_reset=20,
            )
            self.event_types.add("reset")

        if self.domain_rand_config.get("randomize_table_material", False):
            # Load table materials from cloud directory
            table_material_folder_paths = self.domain_rand_config.get(
                "table_material_folder_paths",
                [
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Materials/Base/Wood/"
                ],
            )
            table_material_list = []

            # Collect all material files from specified folders
            for table_material_folder_path in table_material_folder_paths:
                try:
                    _, all_list = omni.client.list(table_material_folder_path)
                    if all_list:
                        table_material_list.extend(
                            [
                                os.path.join(table_material_folder_path, item.relative_path)
                                for item in all_list
                                if item.relative_path.endswith(".mdl")
                            ]
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not list materials from {table_material_folder_path}: {e}"
                    )
                    continue

            material_count = 0

            for table_material_path in table_material_list:
                subidentifiers = asyncio.run(
                    omni.kit.material.library.get_subidentifier_from_mdl(table_material_path)
                )
                for subidentifier in subidentifiers:
                    success, result = omni.kit.commands.execute(
                        "CreateMdlMaterialPrimCommand",
                        mtl_url=str(table_material_path),
                        mtl_name=str(subidentifier),
                        mtl_path=f"/World/Looks/TableMaterials/TableMaterial_{material_count}",
                    )
                    shader_prim = (
                        omni.usd.get_context()
                        .get_stage()
                        .GetPrimAtPath(
                            f"/World/Looks/TableMaterials/TableMaterial_{material_count}/Shader"
                        )
                    )
                    shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(
                        True
                    )
                    shader_prim.CreateAttribute(
                        "inputs:world_or_object", Sdf.ValueTypeNames.Bool
                    ).Set(True)
                    material_count += 1

            table_prim_paths = []
            for env_id in range(self.simulator_config.scene.num_envs):
                table_prim_paths.append(f"/World/envs/env_{env_id}/table_grid/table_0_0/table_top")
                table_prim_paths.append(f"/World/envs/env_{env_id}/table_grid/table_1_0/table_top")
                # table_prim_paths.append(f"/World/envs/env_{env_id}/table_grid/table/table")

            self.events_cfg.table_material_randomization = EventTerm(
                func=randomize_table_material,
                mode="reset",
                params={
                    "table_prim_paths": table_prim_paths,
                    "material_count": material_count,
                },
            )

        self.torso_index = self._robot.find_bodies("torso_link", preserve_order=True)[0][0]
        self.right_palm_index = self._robot.find_bodies(
            "right_dex1_base_link", preserve_order=True
        )[0][0]

        # if self.domain_rand_config.get("randomize_object_material", False):
        #     # Load object materials from cloud directory - can use different materials than tables
        #     object_material_folder_paths = self.domain_rand_config.get("object_material_folder_paths", [
        #         "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Materials/Base/Masonry/",
        #     ])
        #     object_material_list = []
        #     for object_material_folder_path in object_material_folder_paths:
        #         _, all_list = omni.client.list(object_material_folder_path)
        #         object_material_list.extend([os.path.join(object_material_folder_path, item.relative_path) for item in all_list if item.relative_path.endswith(".mdl")])

        #     object_material_count = 0
        #     for material_path in object_material_list:
        #         subidentifiers = asyncio.run(omni.kit.material.library.get_subidentifier_from_mdl(material_path))
        #         for subidentifier in subidentifiers:
        #             success, result = omni.kit.commands.execute("CreateMdlMaterialPrimCommand",
        #                 mtl_url=str(material_path),
        #                 mtl_name=str(subidentifier),
        #                 mtl_path=f"/World/Looks/ObjectMaterials/ObjectMaterial_{object_material_count}"
        #             )
        #             shader_prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"/World/Looks/ObjectMaterials/ObjectMaterial_{object_material_count}/Shader")
        #             shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        #             shader_prim.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
        #             object_material_count += 1

        #     # Get object prim paths from task objects (excluding table which is handled separately)
        #     object_prim_paths = []
        #     if hasattr(self, "_task"):
        #         for object_name in self._task.keys():
        #             if object_name != "table":  # Skip table as it's handled separately
        #                 # Build object prim paths for all environments
        #                 for env_id in range(self.simulator_config.scene.num_envs):
        #                     object_prim_paths.append(f"/World/envs/env_{env_id}/{object_name}")

        #     self.events_cfg.random_object_material = EventTerm(
        #         func=randomize_object_material,
        #         mode="reset",
        #         params={
        #             "object_prim_paths": object_prim_paths,
        #             "material_count": object_material_count,
        #         },
        #         min_step_count_between_reset = 20,
        #     )

        # if self.domain_rand_config.get("push_robots", False):
        #     self.events_cfg.push_robots = EventTerm(
        #         func=mdp.push_by_setting_velocity,
        #         mode="on_push_by_setting_velocity",
        #         params={"velocity_range": {"x": (-self.domain_rand_config["max_push_vel_xy"],
        #                                          self.domain_rand_config["max_push_vel_xy"]),
        #                                    "y": (-self.domain_rand_config["max_push_vel_xy"],
        #                                          self.domain_rand_config["max_push_vel_xy"]),
        #                                 #    "z": (-self.domain_rand_config["max_push_vel_xy"],  # ZL: We don't push in the z direction yet.
        #                                         #  self.domain_rand_config["max_push_vel_xy"]),
        #                                    "roll": (-self.domain_rand_config["max_push_ang_vel"],
        #                                          self.domain_rand_config["max_push_ang_vel"]),
        #                                    "pitch": (-self.domain_rand_config["max_push_ang_vel"],
        #                                          self.domain_rand_config["max_push_ang_vel"]),
        #                                    "yaw": (-self.domain_rand_config["max_push_ang_vel"],
        #                                          self.domain_rand_config["max_push_ang_vel"]),
        #                                    },
        #                 },
        #     )
        #     self.event_types.add("on_push_by_setting_velocity")

        # if self.domain_rand_config.get("push_robots_by_force_torque", False):
        #     self.events_cfg.push_robots_by_force_torque = EventTerm(
        #         func=mdp.apply_external_force_torque,
        #         mode="on_push_by_force_torque",
        #          params={
        #             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        #             "force_range": (-100.0, 100.0),
        #             "torque_range": (-100.0, 100.0),
        #         },)

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)

        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation

        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()

        # print the environment information
        logger.info("Completed setting up the environment...")

        # ---------- Telemetry ----------
        logger.info(f"[TELEMETRY] Rank {getattr(self, 'rank', 'N/A')}: _setup_scene finished")
        # --------------------------------

        # Optimize material randomization memory usage
        try:
            optimize_material_randomization_memory()
            logger.info("Material randomization memory optimization completed")
        except Exception as e:
            logger.warning(f"Material randomization memory optimization failed: {e}")

    def cleanup_memory(self):
        """
        Clean up USD memory. Call this periodically during training to prevent memory leaks.
        """
        try:
            cleanup_usd_memory()
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def _setup_scene(self):
        # ---------- Telemetry ----------
        logger.info(f"[TELEMETRY] Rank {getattr(self, 'rank', 'N/A')}: _setup_scene started")
        # --------------------------------
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file
        # prapare to override the spawn configuration in groot/rl/simulator/isaacsim_articulation_cfg.py
        if DEFAULT_ISAACSIM_VERSION == "4.5":
            pass
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            pass
        else:
            raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")

        asset_abs_path = os.path.abspath(os.path.join(asset_root, asset_path))
        assert os.path.isfile(asset_abs_path)
        logger.warning(
            f"num_position_iterations: {self.simulator_config.sim.physx.num_position_iterations}"
        )
        logger.warning(
            f"num_velocity_iterations: {self.simulator_config.sim.physx.num_velocity_iterations}"
        )
        # import ipdb; ipdb.set_trace()
        spawn = sim_utils.UsdFileCfg(
            usd_path=asset_abs_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions),
                solver_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations,
            ),
        )

        # urdf_path = self.robot_config.asset.urdf_file
        # asset_abs_path = os.path.abspath(os.path.join(asset_root, urdf_path))
        # # import ipdb; ipdb.set_trace()
        # assert(os.path.isfile(asset_abs_path))
        # spawn=sim_utils.UrdfFileCfg(
        #     fix_base=False,
        #     replace_cylinders_with_capsules=True,
        #     asset_path=asset_abs_path,
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions),
        #         solver_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
        #         solver_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations
        #     ),
        #     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        #     ),
        #     # merge_fixed_joints=False
        # )

        # prepare to override the articulation configuration in groot/rl/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping

        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")

        # IdealPDActuatorCfg
        # actuators = {
        #     dof_names_list[i]: IdealPDActuatorCfg(
        #         joint_names_expr=[dof_names_list[i]],
        #         effort_limit=dof_effort_limit_list[i],
        #         velocity_limit=dof_vel_limit_list[i],
        #         stiffness=0,
        #         damping=0,
        #         armature=dof_armature_list[i],
        #         friction=dof_joint_friction_list[i],
        #     ) for i in range(len(dof_names_list))
        # }

        # IdealPD
        # actuators = dict()
        # actuators["all"] = IdealPDActuatorCfg(
        #     joint_names_expr=[dof_names_list[i] for i in range(len(dof_names_list))],
        #     effort_limit={
        #         dof_names_list[i]: dof_effort_limit_list[i] for i in range(len(dof_names_list))
        #     },
        #     velocity_limit={
        #         dof_names_list[i]: dof_vel_limit_list[i] for i in range(len(dof_names_list))
        #     },
        #     stiffness=0,
        #     damping=0,
        #     armature={
        #         dof_names_list[i]: dof_armature_list[i] for i in range(len(dof_names_list))
        #     },
        #     friction={
        #         dof_names_list[i]: dof_joint_friction_list[i] for i in range(len(dof_names_list))
        #     }
        # )

        # ImplicitID
        stiffness_dict = {".*" + key + ".*": value for key, value in stiffness_dict.items()}
        damping_dict = {".*" + key + ".*": value for key, value in damping_dict.items()}
        actuators = dict()
        actuators["all"] = ImplicitActuatorCfg(
            joint_names_expr=[dof_names_list[i] for i in range(len(dof_names_list))],
            effort_limit_sim={
                dof_names_list[i]: dof_effort_limit_list[i] for i in range(len(dof_names_list))
            },
            velocity_limit_sim={
                dof_names_list[i]: dof_vel_limit_list[i] for i in range(len(dof_names_list))
            },
            # stiffness=0,
            # damping=0,
            stiffness=stiffness_dict,
            damping=damping_dict,
            armature={
                dof_names_list[i]: dof_armature_list[i] * 3 for i in range(len(dof_names_list))
            },
            friction={
                dof_names_list[i]: dof_joint_friction_list[i] * 0
                for i in range(len(dof_names_list))
            },
        )
        logger.warning(f"dof_armature_list {dof_armature_list}")
        logger.warning(f"dof_joint_friction_list {dof_joint_friction_list}")

        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
        )
        # robot_articulation_config: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot", init_state=init_state)

        if "task_config" in self.__dict__ and hasattr(self.task_config, "target_obj"):
            which_hand = self.task_config.get("which_hand", "right_hand")
            if which_hand == "right_hand":
                g1_hand_links = [
                    "/World/envs/env_.*/Robot/{}".format(n)
                    for n in self.robot_config.body_names
                    if "right_dex1" in n
                ]
            elif which_hand == "left_hand":
                g1_hand_links = [
                    "/World/envs/env_.*/Robot/{}".format(n)
                    for n in self.robot_config.body_names
                    if "left_dex1" in n
                ]
            elif which_hand == "any_hand":
                g1_hand_links = [
                    "/World/envs/env_.*/Robot/{}".format(n)
                    for n in self.robot_config.body_names
                    if ("left_dex1" in n or "right_dex1" in n)
                ]
            else:
                raise ValueError(f"Invalid which_hand: {which_hand}")
            object_contact_prim_path = f"/World/envs/env_.*/{self.task_config.target_obj}"
            if self.task_config.get("target_obj_contact_sub_prim_path", None) is not None:
                object_contact_prim_path = os.path.join(
                    object_contact_prim_path, self.task_config.target_obj_contact_sub_prim_path
                )
            object_to_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path=object_contact_prim_path,
                filter_prim_paths_expr=g1_hand_links,
            )
            object_to_hand_frame_transformer_config: FrameTransformerCfg = FrameTransformerCfg(
                prim_path=object_contact_prim_path,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(prim_path=link) for link in g1_hand_links
                ],
            )

            table_to_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/table_grid",
                filter_prim_paths_expr=g1_hand_links,
            )

            target_obj_table_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/table_grid",
                filter_prim_paths_expr=object_contact_prim_path,
            )
        elif "task_config" in self.__dict__ and hasattr(self.task_config, "hold_object"):
            which_hand = self.task_config.get("which_hand", "right_hand")
            g1_right_hand_links = [
                "/World/envs/env_.*/Robot/{}".format(n)
                for n in self.robot_config.body_names
                if ("right_dex1" in n)
            ]
            g1_left_hand_links = [
                "/World/envs/env_.*/Robot/{}".format(n)
                for n in self.robot_config.body_names
                if ("left_dex1" in n)
            ]
            if which_hand == "right_hand":
                g1_hold_hand_links = g1_right_hand_links
                g1_pick_hand_links = g1_right_hand_links
            elif which_hand == "left_hand":
                g1_hold_hand_links = g1_left_hand_links
                g1_pick_hand_links = g1_left_hand_links
            else:
                raise ValueError(f"Invalid which_hand: {which_hand}")
            g1_hand_links = g1_right_hand_links + g1_left_hand_links
            hold_object_contact_prim_path = f"/World/envs/env_.*/{self.task_config.hold_object}"
            hold_object_to_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path=hold_object_contact_prim_path,
                filter_prim_paths_expr=g1_hold_hand_links,
            )
            hold_object_to_hand_frame_transformer_config: FrameTransformerCfg = FrameTransformerCfg(
                prim_path=hold_object_contact_prim_path,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(prim_path=link) for link in g1_hold_hand_links
                ],
            )

            front_object_contact_prim_path = f"/World/envs/env_.*/{self.task_config.front_object}"
            front_object_to_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path=front_object_contact_prim_path,
                filter_prim_paths_expr=g1_pick_hand_links,
            )
            front_object_to_hand_frame_transformer_config: FrameTransformerCfg = (
                FrameTransformerCfg(
                    prim_path=front_object_contact_prim_path,
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(prim_path=link) for link in g1_pick_hand_links
                    ],
                )
            )

            table_to_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/table_grid",
                filter_prim_paths_expr=g1_hand_links,
            )
            front_object_table_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                prim_path=front_object_contact_prim_path,
                filter_prim_paths_expr=["/World/envs/env_.*/table_grid"],
            )

        g1_body_links = [
            "/World/envs/env_.*/Robot/{}".format(n)
            for n in self.robot_config.body_names
            if ("hip" in n or "torso" in n or "pelvis" in n)
        ]

        table_to_body_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/table_grid", filter_prim_paths_expr=g1_body_links
        )

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            update_period=0.005,
            track_air_time=True,
        )

        if hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                # Get resolution parameter (assuming square grid)
                resolution = getattr(self.simulator_config.heightmap, "resolution", 9)
                grid_size = getattr(self.simulator_config.heightmap, "grid_size", 1.6)

                # Calculate pattern resolution based on grid size and desired resolution
                # This determines the spacing between ray samples to achieve the desired number of points
                pattern_resolution = grid_size / (resolution - 1) if resolution > 1 else grid_size

                # Add a height scanner to the torso to detect the height of the terrain mesh
                height_scanner_config = RayCasterCfg(
                    prim_path="/World/envs/env_.*/Robot/pelvis",
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                    attach_yaw_only=True,
                    # Apply a grid pattern based on the configured resolution
                    pattern_cfg=patterns.GridPatternCfg(
                        resolution=pattern_resolution,
                        size=[grid_size, grid_size],
                    ),
                    debug_vis=True,
                    mesh_prim_paths=["/World/ground"],
                )
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                self._height_scanner = RayCaster(height_scanner_config)
                self.scene.sensors["height_scanner"] = self._height_scanner

        # parse camera types:
        if hasattr(self.simulator_config, "cameras") and hasattr(
            self.simulator_config.cameras, "camera_types"
        ):
            camera_types = dict_to_true_list(self.simulator_config.cameras.camera_types)
        else:
            camera_types = ["depth", "rgb"]

        logger.info(f"camera_types: {camera_types}")

        if getattr(self.simulator_config, "render_results", False):
            eval_cam_cfg = getattr(self.simulator_config, "eval_camera", {})
            eval_cam_width = eval_cam_cfg.get("width", 1280) if hasattr(eval_cam_cfg, "get") else 1280
            eval_cam_height = eval_cam_cfg.get("height", 720) if hasattr(eval_cam_cfg, "get") else 720
            # Camera parented to robot pelvis. Side selectable via EVAL_CAM_SIDE env var (default
            # "left"): LEFT = 2m to the robot's left (+Y), RIGHT = mirror (-Y). 0.6m above the pelvis,
            # looking back at the front/upper body (where the hands grasp the handle), angled down.
            # world convention: camera +X=forward, +Z=up, +Y=left; rot is the computed look-at quat.
            if os.environ.get("EVAL_CAM_SIDE", "left").lower() == "right":
                _eval_cam_pos, _eval_cam_rot = (0.0, -2.0, 0.6), (0.7724, -0.031, 0.0378, 0.6332)
            else:
                _eval_cam_pos, _eval_cam_rot = (0.0, 2.0, 0.6), (0.7724, 0.031, 0.0378, -0.6332)
            eval_camera_config = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/pelvis/eval_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=_eval_cam_pos, rot=_eval_cam_rot, convention="world"
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=4.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                width=eval_cam_width,
                height=eval_cam_height,
            )
            self.eval_camera = TiledCamera(eval_camera_config)
            self.scene.sensors["eval_camera"] = self.eval_camera

        if (self.terrain_config.mesh_type == "heightfield") or (
            self.terrain_config.mesh_type == "trimesh"
        ):

            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion,
                            noise_range=(0.02, 0.10),
                            noise_step=0.02,
                            border_width=0.25,
                            # proportion=proportion, noise_range=(0.002, 0.02), noise_step=0.002, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion,
                            grid_width=0.95,
                            grid_height_range=(0.01, 0.15),
                            platform_width=2.0,
                        )
                    elif terrain_type == "stairs_up":
                        sub_terrains[terrain_type] = (
                            terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                                proportion=proportion,
                                step_height_range=(0.01, self.terrain_config.stairs.max_height),
                                # step_height_range=(0.25, 0.25),
                                # step_height_range=(0.01, 0.40),
                                step_width=self.terrain_config.stairs.step_width,
                                platform_width=self.terrain_config.stairs.platform_width,
                                border_width=self.terrain_config.stairs.border_size,
                                holes=False,
                            )
                        )
                    elif terrain_type == "stairs_down":
                        sub_terrains[terrain_type] = terrain_gen.MeshPyramidStairsTerrainCfg(
                            proportion=proportion,
                            step_height_range=(0.01, self.terrain_config.stairs.max_height),
                            # step_height_range=(0.3, 0.3),
                            # step_height_range=(0.01, 0.40),
                            step_width=self.terrain_config.stairs.step_width,
                            platform_width=self.terrain_config.stairs.platform_width,
                            border_width=self.terrain_config.stairs.border_size,
                            holes=False,
                        )

            # create a visual material
            visual_material_cfg = sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            )

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                # visual_material=sim_utils.MdlFileCfg(
                #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                #     project_uvw=True,
                # ),
                visual_material=visual_material_cfg,
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            # terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing

        self.vis_spheres = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/goal_marker_sphere",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    ),
                },
            )
        )

        self._task = {}
        self.task_root_origin = {}
        if "task_type" in self.__dict__ and self.task_type == "manipulation":
            import_cfg_path = os.path.join(
                "gr00t/rl/data/tasks/", f"{self.task_name}/scenario_cfg/isaacsim.py"
            )
            # Load the module from the file
            module_name = f"{self.task_name}"
            spec = importlib.util.spec_from_file_location(module_name, import_cfg_path)
            task_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = task_module
            spec.loader.exec_module(task_module)

            # if there is a function _init_usd_path_list in the task module, call it
            if hasattr(task_module, "_init_usd_path_list"):
                task_module._init_usd_path_list()

            if hasattr(self.config, "task") and hasattr(self.config.task, "max_object_number"):
                TaskObjCfgDict, Random_object_idx = (
                    task_module.get_TaskObjCfgDict_and_USDIdx_based_on_num_envs(
                        self.num_envs, self.env_config, self.sim_device
                    )
                )
                # save random_object_idx to self.simulator.xxx
                self.random_object_idx = Random_object_idx
                # also save this as one hot encoding
                # NOTE qben: seems useless and can cause bug for multi-object tasks
                # self.random_object_idx_one_hot = torch.zeros(self.num_envs, self.config.task.max_object_number)
                # self.random_object_idx_one_hot[torch.arange(self.num_envs), Random_object_idx] = 1
                # self.random_object_idx_one_hot = self.random_object_idx_one_hot.to(self.sim_device)

                # Get object class names for each environment
                self.object_class_names = task_module.get_object_class_name_for_indices(
                    Random_object_idx
                )

                # Log class distribution summary instead of individual assignments
                from collections import Counter

                class_counts = Counter(self.object_class_names)
                logger.info(
                    f"Object class distribution across {len(self.object_class_names)} environments:"
                )
                for class_name, count in sorted(class_counts.items()):
                    logger.info(
                        f"  {class_name}: {count} environments ({count/len(self.object_class_names)*100:.1f}%)"
                    )
            else:
                TaskObjCfgDict = task_module.TaskObjCfgDict

            # import ipdb; ipdb.set_trace()
            for name, obj_cfg in TaskObjCfgDict.items():
                if name not in self.config.task.objects:
                    continue
                obj_cfg = obj_cfg.replace(prim_path=f"/World/envs/env_.*/{name}")
                if isinstance(obj_cfg.spawn, sim_utils.MultiAssetSpawnerCfg):
                    obj_cfg.spawn.assets_cfg = obj_cfg.spawn.assets_cfg[: self.scene.cfg.num_envs]
                # import ipdb; ipdb.set_trace()
                self._task[name] = obj_cfg.class_type(obj_cfg)
                # import ipdb; ipdb.set_trace()
                if isinstance(self._task[name], Articulation):
                    self.scene.articulations[name] = self._task[name]
                elif isinstance(self._task[name], RigidObject):
                    self.scene.rigid_objects[name] = self._task[name]
                elif isinstance(self._task[name], DeformableObject):
                    self.scene.deformable_objects[name] = self._task[name]
                else:
                    raise NotImplementedError(f"Task object {name} is not supported.")

                if not isinstance(self._task[name], DeformableObject):
                    self.task_root_origin[name] = torch.cat(
                        [
                            torch.tensor(obj_cfg.init_state.pos, device=self.sim_device),
                            torch.tensor(obj_cfg.init_state.rot, device=self.sim_device),
                            torch.tensor(obj_cfg.init_state.lin_vel, device=self.sim_device),
                            torch.tensor(obj_cfg.init_state.ang_vel, device=self.sim_device),
                        ]
                    )

            self.task_root_origin["robot"] = torch.cat(
                [
                    torch.tensor(robot_articulation_config.init_state.pos, device=self.sim_device),
                    torch.tensor(robot_articulation_config.init_state.rot, device=self.sim_device),
                    torch.tensor(
                        robot_articulation_config.init_state.lin_vel, device=self.sim_device
                    ),
                    torch.tensor(
                        robot_articulation_config.init_state.ang_vel, device=self.sim_device
                    ),
                ]
            )

        if hasattr(self, "task_config"):
            if hasattr(self.task_config, "target_obj_contact_sensor"):
                self.object_to_hand_contact_sensor = ContactSensor(
                    object_to_hand_contact_sensor_config
                )
                # self.object_table_contact_sensor = ContactSensor(object_table_contact_sensor_config)

                self.scene.sensors["object_to_hand_contact_sensor"] = (
                    self.object_to_hand_contact_sensor
                )
                # self.scene.sensors["object_table_contact_sensor"] = self.object_table_contact_sensor

                target_obj_transform_prim_path = f"/World/envs/env_.*/{self.task_config.target_obj}"
                if self.task_config.get("target_obj_transform_sub_prim_path", None) is not None:
                    target_obj_transform_prim_path = os.path.join(target_obj_transform_prim_path, self.task_config.target_obj_transform_sub_prim_path)

                left_hand_frame_transformer_config = FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/left_dex1_base_link",
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(prim_path=target_obj_transform_prim_path),
                    ],
                )
                right_hand_frame_transformer_config = FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/right_dex1_base_link",
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(prim_path=target_obj_transform_prim_path),
                    ],
                )

            elif hasattr(self.task_config, "hold_object_contact_sensor"):
                self.hold_object_to_hand_contact_sensor = ContactSensor(
                    hold_object_to_hand_contact_sensor_config
                )
                self.scene.sensors["hold_object_to_hand_contact_sensor"] = (
                    self.hold_object_to_hand_contact_sensor
                )
                self.front_object_to_hand_contact_sensor = ContactSensor(
                    front_object_to_hand_contact_sensor_config
                )
                self.scene.sensors["front_object_to_hand_contact_sensor"] = (
                    self.front_object_to_hand_contact_sensor
                )

                which_hand = self.task_config.get("which_hand", "right_hand")

                hold_object_transform_prim_path = (
                    f"/World/envs/env_.*/{self.task_config.hold_object}"
                )
                front_object_transform_prim_path = (
                    f"/World/envs/env_.*/{self.task_config.front_object}"
                )

                if which_hand == "right_hand":
                    front_hand_frame_transformer_config = FrameTransformerCfg(
                        prim_path="/World/envs/env_.*/Robot/right_dex1_base_link",
                        target_frames=[
                            FrameTransformerCfg.FrameCfg(
                                prim_path=front_object_transform_prim_path
                            ),
                        ],
                    )
                    hold_hand_frame_transformer_config = FrameTransformerCfg(
                        prim_path="/World/envs/env_.*/Robot/right_dex1_base_link",
                        target_frames=[
                            FrameTransformerCfg.FrameCfg(prim_path=hold_object_transform_prim_path),
                        ],
                    )
                elif which_hand == "left_hand":
                    front_hand_frame_transformer_config = FrameTransformerCfg(
                        prim_path="/World/envs/env_.*/Robot/left_dex1_base_link",
                        target_frames=[
                            FrameTransformerCfg.FrameCfg(
                                prim_path=front_object_transform_prim_path
                            ),
                        ],
                    )
                    hold_hand_frame_transformer_config = FrameTransformerCfg(
                        prim_path="/World/envs/env_.*/Robot/left_dex1_base_link",
                        target_frames=[
                            FrameTransformerCfg.FrameCfg(prim_path=hold_object_transform_prim_path),
                        ],
                    )

            if "table_grid" in self.scene.rigid_objects:
                self.table_to_hand_contact_sensor = ContactSensor(
                    table_to_hand_contact_sensor_config
                )
                self.scene.sensors["table_to_hand_contact_sensor"] = (
                    self.table_to_hand_contact_sensor
                )

                self.table_to_body_contact_sensor = ContactSensor(
                    table_to_body_contact_sensor_config
                )
                self.scene.sensors["table_to_body_contact_sensor"] = (
                    self.table_to_body_contact_sensor
                )

                self.front_object_table_contact_sensor = ContactSensor(
                    front_object_table_contact_sensor_config
                )
                self.scene.sensors["front_object_table_contact_sensor"] = (
                    self.front_object_table_contact_sensor
                )

            if hasattr(self.task_config, "target_obj_contact_sensor"):
                self.scene.sensors["left_hand_frame_transformer"] = FrameTransformer(
                    left_hand_frame_transformer_config
                )
                self.scene.sensors["right_hand_frame_transformer"] = FrameTransformer(
                    right_hand_frame_transformer_config
                )
                self.object_to_hand_frame_transformer = FrameTransformer(
                    object_to_hand_frame_transformer_config
                )
                self.scene.sensors["object_to_hand_frame_transformer"] = (
                    self.object_to_hand_frame_transformer
                )
            elif hasattr(self.task_config, "hold_object_contact_sensor"):
                self.scene.sensors["front_hand_frame_transformer"] = FrameTransformer(
                    front_hand_frame_transformer_config
                )
                self.scene.sensors["hold_hand_frame_transformer"] = FrameTransformer(
                    hold_hand_frame_transformer_config
                )
                self.hold_object_to_hand_frame_transformer = FrameTransformer(
                    hold_object_to_hand_frame_transformer_config
                )
                self.scene.sensors["hold_object_to_hand_frame_transformer"] = (
                    self.hold_object_to_hand_frame_transformer
                )
                self.front_object_to_hand_frame_transformer = FrameTransformer(
                    front_object_to_hand_frame_transformer_config
                )
                self.scene.sensors["front_object_to_hand_frame_transformer"] = (
                    self.front_object_to_hand_frame_transformer
                )

        if hasattr(self, "simulator_config") and hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                self._height_scanner = RayCaster(height_scanner_config)
                self.scene.sensors["height_scanner"] = self._height_scanner

        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins

        if hasattr(self.simulator_config, "cameras") and getattr(
            self.simulator_config.cameras, "enable_cameras", False
        ):
            camera_pos_offset = torch.tensor(
                self.config.simulator.config.cameras.camera_pos_offset, device=self.sim_device
            )
            camera_rot_offset = torch.tensor(
                self.config.simulator.config.cameras.camera_rot_offset, device=self.sim_device
            )
            ego_camera_config = TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Robot/{self.simulator_config.cameras.camera_attached_link}/ego_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=camera_pos_offset, rot=camera_rot_offset, convention="world"
                ),
                data_types=camera_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self.simulator_config.cameras.camera_focal_length,
                    focus_distance=self.simulator_config.cameras.camera_focus_distance,
                    horizontal_aperture=self.simulator_config.cameras.camera_horizontal_aperture,
                    vertical_aperture=self.simulator_config.cameras.camera_vertical_aperture,
                    clipping_range=eval(self.simulator_config.cameras.camera_clipping_range),
                ),
                width=self.simulator_config.cameras.camera_resolutions[1],
                height=self.simulator_config.cameras.camera_resolutions[0],
                debug_vis=True,
            )

            self.ego_camera = TiledCamera(ego_camera_config)
            self.scene.sensors["ego_camera"] = self.ego_camera
        else:
            self.ego_camera = None

        if self.scene_creation_callback is not None:
            self.scene_creation_callback(self)

        # import ipdb; ipdb.set_trace()

        # clone, filter, and replicate

        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])
        # env_origins was set in the scene.clone_environments()
        self.env_origins = self.scene.env_origins

        # Get table top positions after scene setup is complete
        if hasattr(self, "_task") and "table_grid" in self._task:
            try:
                self.table_top_pos = self.get_all_table_top_positions()
                logger.info(
                    f"Successfully retrieved table top positions for {len(self.table_top_pos)} environments"
                )
            except Exception as e:
                logger.warning(f"Failed to get table top positions with RigidPrim method: {e}")
                try:
                    # Fallback to USD method
                    self.table_top_pos = self.get_all_table_top_positions_usd_method()
                    logger.info(
                        f"Successfully retrieved table top positions using USD method for {len(self.table_top_pos)} environments"
                    )
                except Exception as e2:
                    logger.warning(f"Failed to get table top positions with USD method: {e2}")
                    self.table_top_pos = None
        else:
            self.table_top_pos = None

        # add lights
        if self.config.simulator.config.get("randomize_dome_light", False):
            from gr00t.rl.isaac_utils.playground.env_rand.domelight import RandomDomeLightCfg

            dome_light_cfg = RandomDomeLightCfg(
                texture_file_folders=[
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Indoor/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Clear/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Cloudy/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Night/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Studio/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Environments/2024_1/DomeLights/Clear/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Environments/2024_1/DomeLights/Cloudy/",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Environments/2024_1/DomeLights/Indoor/",
                ],
                dynamic_randomize_texture=True,
                dynamic_randomize_texture_interval=1.0,
            )
            dome_light = dome_light_cfg.func("/World/DomeLight", dome_light_cfg)
            self.scene.extras["dome_light"] = dome_light
        else:
            dome_light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
            dome_light = dome_light_cfg.func(
                "/World/DomeLight", dome_light_cfg
            )  # added for manipulation
            self.scene.extras["dome_light"] = dome_light

            # self.vis_spheres = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/goal_marker_sphere",
            #         markers={
            #             "sphere": sim_utils.SphereCfg(
            #             radius=0.05,
            #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),),
            #         }))

            # if "task_type" in self.__dict__ and self.task_type=="manipulation":

            # self.vis_spheres = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/goal_marker_sphere",
            #         markers={
            #             "sphere": sim_utils.SphereCfg(
            #             radius=0.05,
            #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),),
            #         }))

            # if "task_type" in self.__dict__ and self.task_type=="manipulation":

            #     # get a visual cube
            #     self.visual_cube = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/goal_marker_cube",
            #         markers={
            #             "cube": sim_utils.CuboidCfg(
            #         size=(0.075, 0.075, 0.075),
            #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
            #         }))

            self.visual_cube_red = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/goal_marker_cube_red",
                    markers={
                        "cube": sim_utils.CuboidCfg(
                            size=(0.075, 0.075, 0.075),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0)
                            ),
                        ),
                    },
                )
            )

            self.visual_cube_green = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/goal_marker_cube_green",
                    markers={
                        "cube": sim_utils.CuboidCfg(
                            size=(0.075, 0.075, 0.075),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0)
                            ),
                        ),
                    },
                )
            )

        # Add visualization markers for object position prediction
        # self.pred_obj_pos_marker = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/predicted_obj_pos",
        #     markers={
        #         "cube": sim_utils.CuboidCfg(
        #         size=(0.075, 0.075, 0.075),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),),  # Blue for predicted
        #     }))

        # self.gt_obj_pos_marker = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/ground_truth_obj_pos",
        #     markers={
        #         "cube": sim_utils.CuboidCfg(
        #         size=(0.075, 0.075, 0.075),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),),  # Orange for ground truth
        #     }))

        # Add visualization markers for object position prediction
        # self.pred_obj_pos_marker = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/predicted_obj_pos",
        #     markers={
        #         "cube": sim_utils.CuboidCfg(
        #         size=(0.075, 0.075, 0.075),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),),  # Blue for predicted
        #     }))

        # self.gt_obj_pos_marker = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/ground_truth_obj_pos",
        #     markers={
        #         "cube": sim_utils.CuboidCfg(
        #         size=(0.075, 0.075, 0.075),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),),  # Orange for ground truth
        #     }))

        # ---------- Telemetry ----------
        logger.info(
            f"[TELEMETRY] Rank {getattr(self, 'rank', 'N/A')}: _setup_scene completed all assets and sensors initialization"
        )
        # --------------------------------

    def setup_keyboard(self):
        # TODO(haoru): fix: this is a ABI breaking change on IsaacLab side c.a. 2025-08. The commented out code is for the new version.
        try:
            from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard

            self.keyboard_interface = Se2Keyboard()
        except Exception:
            from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

            self.keyboard_interface = Se2Keyboard(cfg=Se2KeyboardCfg())

    def add_keyboard_callback(self, key, callback):
        self.keyboard_interface.add_callback(key, callback)

    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            if DEFAULT_ISAACSIM_VERSION == "4.5":
                ### IsaacSim 4.5
                from isaacsim.util.debug_draw import _debug_draw
            elif DEFAULT_ISAACSIM_VERSION == "4.2":
                ### IsaacSim 4.2
                from omni.isaac.debug_draw import _debug_draw
            else:
                raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")
            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1.0 / self.simulator_config.sim.fps
        if not self.headless:
            self.setup_keyboard()

    def setup_terrain(self, mesh_type):
        pass

    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, pos_tensor):
        pass

    def load_assets(self):
        """
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        """

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        # import ipdb; ipdb.set_trace()
        self.body_ids, self.body_names = self._robot.find_bodies(
            self.robot_config.body_names, preserve_order=True
        )

        self.contact_to_body_idx = [
            self.contact_sensor.body_names.index(body_name) for body_name in self.body_names
        ]

        # self.simulator._robot.find_bodies("d435_link", preserve_order=True)
        # if get the camera attached link, then get the body id
        if (
            hasattr(self.simulator_config, "cameras")
            and self.simulator_config.cameras.enable_cameras
        ):
            self.camera_body_id = self._robot.find_bodies(
                self.simulator_config.cameras.camera_attached_link, preserve_order=True
            )[0]
            logger.info(
                f"Camera attached link: {self.simulator_config.cameras.camera_attached_link}, Camera body id: {self.camera_body_id}"
            )

        # import ipdb; ipdb.set_trace()
        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, rl config)
        # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        """
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19], ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        """

        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning(
                "The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim."
            )

        # assert if  aligns with config
        assert self.num_dof == len(
            self.robot_config.dof_names
        ), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(
            self.robot_config.body_names
        ), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def set_ego_camera_pose(self, base_pos, base_quat_wxyz):
        if self.ego_camera is not None:
            self.ego_camera.set_world_poses(base_pos, base_quat_wxyz, convention="world")

    def create_envs(self, num_envs, env_origins, base_init_state):

        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        return self.scene, self._robot

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.torque_limits = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False
        )

        self.dof_pos_limits_termination = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            )

            self.dof_pos_limits_termination[i, 0] = (
                m - 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
            )
            self.dof_pos_limits_termination[i, 1] = (
                m + 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
            )

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        """
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        """
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else:  # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices

    def prepare_sim(self):
        self.refresh_sim_tensors()  # initialize tensors

    def get_height_map(self):
        if hasattr(self.scene, "sensors") and "height_scanner" in self.scene.sensors:
            height_tensor = self.scene.sensors["height_scanner"].data.ray_hits_w[:, :, 2]
            batch_size = height_tensor.shape[0]

            # Use configurable resolution from config if available, otherwise default to 9
            resolution = (
                getattr(self.simulator_config.heightmap, "resolution", 9)
                if hasattr(self.simulator_config, "heightmap")
                else 9
            )

            height_map = height_tensor.view(batch_size, resolution, resolution)
            return height_map
        else:
            raise ValueError("Height scanner not found in the scene.")

    def get_rgb_image(self):
        # import ipdb; ipdb.set_trace()
        if self.ego_camera is not None:
            # Get RGB image from the ego camera
            # The image is typically in shape [batch_size, height, width, channels]
            rgb_image = self.scene.sensors["ego_camera"].data.output["rgb"]

            # Convert to float tensor if it's not already
            if rgb_image.dtype != torch.float:
                rgb_image = rgb_image.float()

            # If values are in 0-255 range, normalize to 0-1
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image / 255.0

            if hasattr(self.simulator_config.cameras, "image_mean") and hasattr(
                self.simulator_config.cameras, "image_std"
            ):
                image_mean = torch.tensor(
                    self.simulator_config.cameras.image_mean, device=self.sim_device
                )
                image_std = torch.tensor(
                    self.simulator_config.cameras.image_std, device=self.sim_device
                )
                rgb_image = (rgb_image - image_mean) / image_std

            return rgb_image
        else:
            # Return empty tensor if camera is not available
            camera_resolution = (
                self.simulator_config.cameras.camera_resolutions
                if hasattr(self.simulator_config, "cameras")
                else [0, 0]
            )
            return torch.zeros(
                (self.scene.cfg.num_envs, camera_resolution[0], camera_resolution[1], 3),
                device=self.sim_device,
                dtype=torch.float,
            )

    def get_depth_image(self):
        if self.ego_camera is not None:
            # Get depth image from the ego camera
            depth_image = self.scene.sensors["ego_camera"].data.output["depth"]
            # Convert to float tensor if it's not already
            if depth_image.dtype != torch.float:
                depth_image = depth_image.float()
            # mask the inf to 0
            depth_image = torch.where(
                depth_image == float("inf"), torch.zeros_like(depth_image), depth_image
            )
            if self.simulator_config.cameras.stack_depth_image:
                depth_image = torch.cat([depth_image, depth_image, depth_image], dim=-1)

            return depth_image
        else:
            return torch.zeros(
                (
                    self.scene.cfg.num_envs,
                    self.simulator_config.cameras.camera_resolutions[0],
                    self.simulator_config.cameras.camera_resolutions[1],
                    1,
                ),
                device=self.sim_device,
                dtype=torch.float,
            )

    def compute_grad_norm(self):
        if hasattr(self.scene, "sensors") and "height_scanner" in self.scene.sensors:
            height_tensor = self.scene.sensors["height_scanner"].data.ray_hits_w[:, :, 2]
            batch_size = height_tensor.shape[0]

            # Use configurable resolution from config if available, otherwise default to 9
            resolution = (
                getattr(self.simulator_config.heightmap, "resolution", 9)
                if hasattr(self.simulator_config, "heightmap")
                else 9
            )

            height_map = height_tensor.view(batch_size, resolution, resolution)

            height_map_unsqueezed = height_map.unsqueeze(1)

            kernel_x = torch.tensor(
                [[-0.5, 0, 0.5]], dtype=height_map.dtype, device=height_map.device
            ).view(1, 1, 1, 3)
            kernel_y = torch.tensor(
                [[-0.5], [0], [0.5]], dtype=height_map.dtype, device=height_map.device
            ).view(1, 1, 3, 1)

            grad_x = F.conv2d(height_map_unsqueezed, kernel_x, padding=(0, 1))
            grad_y = F.conv2d(height_map_unsqueezed, kernel_y, padding=(1, 0))
            grad_norm = torch.sqrt(grad_x**2 + grad_y**2)

            return grad_norm.squeeze(1)
        else:
            raise ValueError("Height scanner not found in the scene.")

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        ############################################################################################
        # ZL: This function is a very bad design. It should be refactored.
        # If refresh_sim_tensors is not called properly, a lot of values will not be properly updated.
        ############################################################################################
        self.all_root_states = {}
        self.all_root_base_quat = {}
        self.all_root_states["robot"] = self._robot.data.root_state_w  # (num_envs, 13)

        for task_name, task_obj in self._task.items():
            self.all_root_states[task_name] = task_obj.data.root_state_w  # (num_envs, 13)
            self.all_root_base_quat[task_name] = self.all_root_states[task_name][
                :, [4, 5, 6, 3]
            ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency

        self.all_root_base_quat["robot"] = self.base_quat

        if hasattr(self, "task_config"):
            if hasattr(self.task_config, "target_obj_contact_sensor"):
                self.object_to_hand_contact_forces = (
                    self.object_to_hand_contact_sensor.data.force_matrix_w.clone()
                )
            elif hasattr(self.task_config, "hold_object_contact_sensor"):
                self.hold_object_to_hand_contact_forces = (
                    self.hold_object_to_hand_contact_sensor.data.force_matrix_w.clone()
                )
                self.front_object_to_hand_contact_forces = (
                    self.front_object_to_hand_contact_sensor.data.force_matrix_w.clone()
                )
                self.front_object_table_contact_forces = (
                    self.front_object_table_contact_sensor.data.force_matrix_w.clone()
                )
            if "table_grid" in self.scene.rigid_objects:
                self.table_to_hand_contact_forces = (
                    self.table_to_hand_contact_sensor.data.force_matrix_w.clone()
                )
                self.table_to_body_contact_forces = (
                    self.table_to_body_contact_sensor.data.force_matrix_w.clone()
                )

            if hasattr(self.task_config, "hold_object"):
                which_hand = self.task_config.which_hand
                if which_hand == "left_hand":
                    self.hold_hand_transform_pos = self.scene.sensors[
                        "hold_hand_frame_transformer"
                    ].data.target_pos_source.clone()
                    self.hold_hand_transform_rot = self.scene.sensors[
                        "hold_hand_frame_transformer"
                    ].data.target_quat_source.clone()
                    self.grasp_hand_transform_pos = self.scene.sensors[
                        "front_hand_frame_transformer"
                    ].data.target_pos_source.clone()
                    self.grasp_hand_transform_rot = self.scene.sensors[
                        "front_hand_frame_transformer"
                    ].data.target_quat_source.clone()
                elif which_hand == "right_hand":
                    self.hold_hand_transform_pos = self.scene.sensors[
                        "hold_hand_frame_transformer"
                    ].data.target_pos_source.clone()
                    self.hold_hand_transform_rot = self.scene.sensors[
                        "hold_hand_frame_transformer"
                    ].data.target_quat_source.clone()
                    self.grasp_hand_transform_pos = self.scene.sensors[
                        "front_hand_frame_transformer"
                    ].data.target_pos_source.clone()
                    self.grasp_hand_transform_rot = self.scene.sensors[
                        "front_hand_frame_transformer"
                    ].data.target_quat_source.clone()
                else:
                    raise ValueError(f"Invalid hand: {which_hand}")
            else:
                self.left_hand_transform_pos = self.scene.sensors[
                    "left_hand_frame_transformer"
                ].data.target_pos_source.clone()
                self.left_hand_transform_rot = self.scene.sensors[
                    "left_hand_frame_transformer"
                ].data.target_quat_source.clone()
                self.right_hand_transform_pos = self.scene.sensors[
                    "right_hand_frame_transformer"
                ].data.target_pos_source.clone()
                self.right_hand_transform_rot = self.scene.sensors[
                    "right_hand_frame_transformer"
                ].data.target_quat_source.clone()

            # import ipdb; ipdb.set_trace()

        # camera
        # if hasattr(self.simulator_config, "cameras") and self.simulator_config.cameras.enable_cameras:
        #     # body_pos_w has shape [num_envs, num_bodies, 3]
        #     # body_quat_w has shape [num_envs, num_bodies, 4]
        #     # camera_body_id is shape [1] (tensor), so indexing gives [num_envs, 1, ...]
        #     self._camera_pos = self._robot.data.body_pos_w[:, self.camera_body_id, :]  # [num_envs, 1, 3]
        #     self._camera_quat = self._robot.data.body_quat_w[:, self.camera_body_id, :]  # [num_envs, 1, 4] in wxyz format

    @property
    def _camera_pos(self):
        return self._robot.data.body_pos_w[:, self.camera_body_id, :]

    @property
    def _camera_quat(self):
        return self._robot.data.body_quat_w[:, self.camera_body_id, :]

    @property
    def contact_forces(self):
        return self.contact_sensor.data.net_forces_w[
            :, self.contact_to_body_idx, :
        ]  # (num_envs, num_bodies, 3)

    @property
    def base_quat(self):
        return self._robot.data.root_state_w[
            :, [4, 5, 6, 3]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency

    @property
    def robot_root_states(self):
        return self._robot.data.root_state_w[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]

    @property
    def dof_pos(self):
        return self._robot.data.joint_pos[:, self.dof_ids]

    @property
    def dof_vel(self):
        return self._robot.data.joint_vel[:, self.dof_ids]

    @property
    def dof_acc(self):
        return self._robot.data.joint_acc[:, self.dof_ids]

    @property
    def _rigid_body_pos(self):
        return self._robot.data.body_pos_w[:, self.body_ids, :]

    @property
    def _rigid_body_rot(self):
        return self._robot.data.body_quat_w[:, self.body_ids][
            :, :, [1, 2, 3, 0]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency

    @property
    def _rigid_body_vel(self):
        return self._robot.data.body_lin_vel_w[:, self.body_ids, :]

    @property
    def _rigid_body_ang_vel(self):
        return self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    def get_task_root_state(self, obj_name):
        return self._task[obj_name].data.root_state_w

    def get_task_dof_pos(self, obj_name):
        return self._task[obj_name].data.joint_pos

    def get_task_dof_vel(self, obj_name):
        return self._task[obj_name].data.joint_vel

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)

    def set_actor_root_state_tensor(
        self, set_env_ids, root_states
    ):  # TODO: only robot actor root state
        self._robot.write_root_state_to_sim(root_states[set_env_ids, :], set_env_ids)

    def write_root_state_to_sim(self, root_states, set_env_ids):
        self._robot.write_root_state_to_sim(root_states, set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)

    def write_joint_state_to_sim(self, dof_pos, dof_vel, set_env_ids):
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)

    def clear_external_force_and_torque(self):
        force_b = self._robot._external_force_b.clone()
        torque_b = self._robot._external_torque_b.clone()
        force_b[:, :] = 0.0
        torque_b[:, :] = 0.0
        self._robot.set_external_force_and_torque(force_b, torque_b)

    def wsdpt_push_robot(self, torso_force, right_palm_force, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.scene.cfg.num_envs, device=self.sim_device)
        if len(env_ids) == 0:
            return
        force_b = self._robot._external_force_b.clone()
        torque_b = self._robot._external_torque_b.clone()
        force_b[env_ids, self.torso_index] = torso_force
        force_b[env_ids, self.right_palm_index] = right_palm_force
        self._robot.set_external_force_and_torque(force_b, torque_b)

    def set_task_root_state_tensor(self, set_env_ids, root_states):
        for name, task_obj in self._task.items():
            if name in root_states.keys():
                task_obj.write_root_state_to_sim(root_states[name][set_env_ids, :], set_env_ids)
                task_obj.reset()

    def set_task_dof_state_tensor(self, set_env_ids, dof_states):
        for name, task_obj in self._task.items():
            if name in dof_states.keys():
                dof_pos, dof_vel, dof_ids = dof_states[name]
                task_obj.write_joint_state_to_sim(
                    dof_pos[set_env_ids, :], dof_vel[set_env_ids, :], dof_ids, set_env_ids
                )

    def apply_torques_at_task_dof(self, set_env_ids, torques):
        for name, task_obj in self._task.items():
            if name in torques.keys():
                if isinstance(task_obj, Articulation):
                    task_obj.set_joint_effort_target(
                        torques[name][set_env_ids, :], env_ids=set_env_ids
                    )
                else:
                    raise ValueError(f"Task {name} is not an articulation.")

    def set_task_visual_state_tensor(self, set_env_ids):
        stage = omni.usd.get_context().get_stage()
        obj_names = []
        if hasattr(self.task_config, "target_obj"):
            obj_names.append(self.task_config.target_obj)
        if hasattr(self.task_config, "hold_object"):
            obj_names.append(self.task_config.hold_object)
        if hasattr(self.task_config, "front_object"):
            obj_names.append(self.task_config.front_object)
        if hasattr(self.task_config, "back_object"):
            obj_names.append(self.task_config.back_object)
        if hasattr(self.task_config, "tray_object"):
            obj_names.append(self.task_config.tray_object)
        for obj_name in obj_names:
            for env_enumerator, env_id in enumerate(set_env_ids):
                for name, task_obj in self._task.items():
                    # Construct the actual path with the specific env_id
                    obj_path = f"/World/envs/env_{env_id}/{obj_name}"
                    # Try possible shader paths based on object type
                    if obj_name == "tray_object":
                        shader_paths = [
                            f"{obj_path}/tn__tray282810_YF7A/tn__Part1_f5/Looks/Diffuse/PreviewSurface",  # for tray object
                        ]
                    else:
                        shader_paths = [
                            f"{obj_path}/material/Shader",  # for asset objects
                            f"{obj_path}/geometry/material/Shader",  # for haoru's randomized size objects
                        ]

                    shader_prim = None
                    valid_shader_path = None

                    for shader_path in shader_paths:
                        temp_prim = stage.GetPrimAtPath(shader_path)
                        if temp_prim and temp_prim.IsValid():
                            shader_prim = temp_prim
                            valid_shader_path = shader_path
                            break

                    if shader_prim is None:
                        # Skip if no valid shader prim found
                        print(
                            f"Warning: No valid shader found for object at {obj_path}. Tried paths: {shader_paths}"
                        )
                        continue

                    # Check if this is actually a shader prim before creating UsdShade.Shader
                    try:
                        shader = UsdShade.Shader(shader_prim)
                        if not shader:
                            print(
                                f"Warning: Could not create UsdShade.Shader from prim at {valid_shader_path}"
                            )
                            continue

                        # Set new color
                        color = np.random.rand(3)
                        diffuse_input = shader.GetInput("diffuseColor")
                        if diffuse_input:
                            diffuse_input.Set((color[0], color[1], color[2]))
                        else:
                            print(
                                f"Warning: No diffuseColor input found for shader at {valid_shader_path}"
                            )
                    except Exception as e:
                        print(f"Error accessing shader at {valid_shader_path}: {str(e)}")
                        continue

    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller

    def render(self, sync_frame_time=True):
        pass

    def apply_commands(self, commands_description):
        if commands_description == "forward_command":
            self.commands[:, 0] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "backward_command":
            self.commands[:, 0] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "left_command":
            self.commands[:, 1] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "right_command":
            self.commands[:, 1] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_left_command":
            self.commands[:, 3] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_right_command":
            self.commands[:, 3] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "zero_command":
            self.commands[:, :4] = 0
            logger.info(f"Current Command: {self.commands[:, ]}")

    # debug visualization
    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)

    def draw_spheres_batch(self, pos, rot=None, scales=None):
        self.vis_spheres.visualize(pos, rot, scales)

    def draw_cubes_batch_red(self, pos, rot=None, scales=None):
        if pos.shape[0] > 0:
            self.visual_cube_red.set_visibility(True)
            if rot is None:
                # Generate random quaternion in wxyz format
                rot = torch.randn((pos.shape[0], 4))
                rot = rot / torch.norm(rot, dim=1, keepdim=True)  # Normalize to unit quaternions
            self.visual_cube_red.visualize(pos, rot, scales)
        else:
            self.visual_cube_red.set_visibility(False)

    def draw_cubes_batch_green(self, pos, rot=None, scales=None):
        if pos.shape[0] > 0:
            self.visual_cube_green.set_visibility(True)
            if rot is None:
                # Generate random quaternion in wxyz format
                rot = torch.randn((pos.shape[0], 4))
                rot = rot / torch.norm(rot, dim=1, keepdim=True)  # Normalize to unit quaternions
            self.visual_cube_green.visualize(pos, rot, scales)
        else:
            self.visual_cube_green.set_visibility(False)

    def draw_cubes_batch(self, pos, rot=None, scales=None):
        """
        Draw a batch of cubes at specified positions with optional rotations and scales.

        Args:
            pos (torch.Tensor): Tensor of shape [batch_size, 3] representing the positions of the cubes.
            rot (torch.Tensor, optional): Tensor of shape [batch_size, 4] representing the rotations of the cubes in quaternion format.
            scales (torch.Tensor, optional): Tensor of shape [batch_size, 3] representing the scales of the cubes.
        """
        self.visual_cube.visualize(pos, rot, scales)

    def draw_predicted_obj_pos(self, pred_obj_pos, robot_base_pos=None):
        """
        Draw predicted object positions as blue spheres.

        Args:
            pred_obj_pos (torch.Tensor): Predicted object positions in robot base frame [batch_size, 3]
            robot_base_pos (torch.Tensor, optional): Robot base positions [batch_size, 3]. If None, uses current robot positions.
        """
        if pred_obj_pos.shape[0] > 0:
            self.pred_obj_pos_marker.set_visibility(True)

            # Transform from robot base frame to world frame
            if robot_base_pos is not None:
                # Use provided robot base positions
                robot_pos = robot_base_pos
                # Need to get robot quaternions - assuming they're available in robot_root_states
                robot_quat = self.base_quat[: pred_obj_pos.shape[0]]  # [x,y,z,w] format
            else:
                # Use current robot base positions and orientations
                robot_pos = self.robot_root_states[: pred_obj_pos.shape[0], 0:3]
                robot_quat = self.base_quat[: pred_obj_pos.shape[0]]  # [x,y,z,w] format

            # Convert quaternion from [x,y,z,w] to [w,x,y,z] for quat_rotate
            from gr00t.rl.isaac_utils.rotations import quat_rotate

            # Apply rotation transformation: rotate pred_obj_pos by robot's orientation
            world_pred_pos_relative = quat_rotate(robot_quat, pred_obj_pos, w_last=True)

            # Apply translation: add robot's position
            world_pred_pos = world_pred_pos_relative + robot_pos

            world_pred_pos[:, 2] -= 0.55

            # Generate default rotations (identity quaternions in wxyz format)
            rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pred_obj_pos.device).repeat(
                pred_obj_pos.shape[0], 1
            )

            self.pred_obj_pos_marker.visualize(world_pred_pos, rot, None)
        else:
            self.pred_obj_pos_marker.set_visibility(False)

    def draw_gt_obj_pos(self, gt_obj_pos, robot_base_pos=None):
        """
        Draw ground truth object positions as orange spheres.

        Args:
            gt_obj_pos (torch.Tensor): Ground truth object positions in robot base frame [batch_size, 3]
            robot_base_pos (torch.Tensor, optional): Robot base positions [batch_size, 3]. If None, uses current robot positions.
        """
        if gt_obj_pos.shape[0] > 0:
            self.gt_obj_pos_marker.set_visibility(True)

            # Transform from robot base frame to world frame
            if robot_base_pos is not None:
                # Use provided robot base positions
                robot_pos = robot_base_pos
                # Need to get robot quaternions - assuming they're available in robot_root_states
                robot_quat = self.base_quat[: gt_obj_pos.shape[0]]  # [x,y,z,w] format
            else:
                # Use current robot base positions and orientations
                robot_pos = self.robot_root_states[: gt_obj_pos.shape[0], 0:3]
                robot_quat = self.base_quat[: gt_obj_pos.shape[0]]  # [x,y,z,w] format

            # Convert quaternion from [x,y,z,w] to [w,x,y,z] for quat_rotate
            from gr00t.rl.isaac_utils.rotations import quat_rotate

            # import ipdb; ipdb.set_trace()
            # Apply rotation transformation: rotate gt_obj_pos by robot's orientation
            world_gt_pos_relative = quat_rotate(robot_quat, gt_obj_pos, w_last=True)

            # Apply translation: add robot's position
            world_gt_pos = world_gt_pos_relative + robot_pos

            # add offset to avoid visual overlap
            world_gt_pos[:, 2] -= 0.55
            # Generate default rotations (identity quaternions in wxyz format)
            rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gt_obj_pos.device).repeat(
                gt_obj_pos.shape[0], 1
            )

            self.gt_obj_pos_marker.visualize(world_gt_pos, rot, None)
        else:
            self.gt_obj_pos_marker.set_visibility(False)

    def draw_obj_pos_predictions(self, pred_obj_pos, gt_obj_pos, robot_base_pos=None):
        """
        Draw both predicted and ground truth object positions.

        Args:
            pred_obj_pos (torch.Tensor): Predicted object positions in robot base frame [batch_size, 3]
            gt_obj_pos (torch.Tensor): Ground truth object positions in robot base frame [batch_size, 3]
            robot_base_pos (torch.Tensor, optional): Robot base positions [batch_size, 3]. If None, uses current robot positions.
        """
        self.draw_predicted_obj_pos(pred_obj_pos, robot_base_pos)
        self.draw_gt_obj_pos(gt_obj_pos, robot_base_pos)

    def get_object_class_names(self, env_ids=None):
        """
        Get object class names for specified environments.

        Args:
            env_ids (torch.Tensor, optional): Environment IDs. If None, returns all.

        Returns:
            list: Object class names for the specified environments.
        """
        if not hasattr(self, "object_class_names"):
            return None

        if env_ids is None:
            return self.object_class_names
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().numpy()
            return [self.object_class_names[i] for i in env_ids]

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [(start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)

    def draw_lines_batch(self, point_lists, colors, sizes):
        for point_list, color, size in zip(point_lists, colors, sizes):
            self.draw.draw_lines_spline(point_list, color, size, False)

    def adjust_camera_pos(self, pos):
        self.sim.set_camera_view(pos, [-0.5, 0.0, 0.5])

    def trigger_reset_events(
        self, env_ids: torch.Tensor | None = None, global_env_step_count: Optional[int] = None
    ):
        if "reset" in self.event_types:
            self.event_manager.apply(
                env_ids=env_ids, mode="reset", global_env_step_count=global_env_step_count
            )
        else:
            raise ValueError("No reset events found in the simulator.")

    def trigger_on_push_by_setting_velocity_events(
        self, env_ids: torch.Tensor | None = None, global_env_step_count: Optional[int] = None
    ):
        self.event_manager.apply(
            env_ids=env_ids,
            mode="on_push_by_setting_velocity",
            global_env_step_count=global_env_step_count,
        )

    def trigger_on_push_by_force_torque_events(
        self, env_ids: torch.Tensor | None = None, global_env_step_count: Optional[int] = None
    ):
        self.event_manager.apply(
            env_ids=env_ids,
            mode="on_push_by_force_torque",
            global_env_step_count=global_env_step_count,
        )

    def get_all_table_top_positions_usd_method(self):
        """Alternative method using USD API directly."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        positions = []

        for env_id in range(self.num_envs):
            table_prim_path = f"/World/envs/env_{env_id}/table_grid/table_1_0/table_top"
            try:
                # Get the prim from USD stage
                prim = stage.GetPrimAtPath(table_prim_path)
                if prim.IsValid():
                    # Use UsdGeom to get world transform
                    xformable = UsdGeom.Xformable(prim)
                    if xformable:
                        # Get world transform matrix
                        world_matrix = xformable.ComputeLocalToWorldTransform(0)  # time=0
                        translation = world_matrix.ExtractTranslation()
                        positions.append([translation[0], translation[1], translation[2]])
                    else:
                        logger.warning(f"Prim is not xformable at path: {table_prim_path}")
                        positions.append([0.0, 0.0, 0.0])
                else:
                    logger.warning(f"Prim not found at path: {table_prim_path}")
                    positions.append([0.0, 0.0, 0.0])
            except Exception as e:
                logger.warning(f"Error getting position for env {env_id}: {e}")
                positions.append([0.0, 0.0, 0.0])

        return torch.tensor(positions, device=self.device)

    def get_camera_pose_from_prim_path(self):
        """
        Get camera poses using the camera's prim path.
        Returns position and quaternion (in wxyz format) for all environments.
        """
        if self.ego_camera is None:
            return None, None

        import omni.usd

        stage = omni.usd.get_context().get_stage()

        positions = []
        rotations = []

        # Iterate through all environments
        for env_id in range(self.num_envs):
            # Construct the actual prim path for this environment
            prim_path = f"/World/envs/env_{env_id}/Robot/{self.simulator_config.cameras.camera_attached_link}/ego_camera"
            prim = stage.GetPrimAtPath(prim_path)

            if prim and prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                # Get world transform matrix
                world_matrix = xformable.ComputeLocalToWorldTransform(0)

                # Extract translation
                translation = world_matrix.ExtractTranslation()
                positions.append([translation[0], translation[1], translation[2]])

                # Extract rotation as quaternion (wxyz format from Gf.Quatd)
                rotation_quat = world_matrix.ExtractRotationQuat()
                rotations.append(
                    [
                        rotation_quat.GetReal(),  # w
                        rotation_quat.GetImaginary()[0],  # x
                        rotation_quat.GetImaginary()[1],  # y
                        rotation_quat.GetImaginary()[2],  # z
                    ]
                )
            else:
                logger.warning(f"Camera prim not found or invalid at: {prim_path}")
                positions.append([0.0, 0.0, 0.0])
                rotations.append([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
        rot_tensor = torch.tensor(rotations, dtype=torch.float32, device=self.device)

        return pos_tensor, rot_tensor
