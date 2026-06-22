# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import glob
import logging
import os

logging.getLogger("asyncio").setLevel(logging.WARNING)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from gr00t.rl.isaac_utils.playground.env_rand.door import DoorSpawnerCfg, spawn_door

door_spawner_cfg = DoorSpawnerCfg(
    func=spawn_door,
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=4,
        fix_root_link=True,
    ),
    activate_contact_sensors=True,
    build_latch=True,
    add_floors=True,
    door_open_lr=["right"],
    door_open_io=["out"],
    door_handle_tblr=(0.95, 0.85, 0.08, 0.15),
    randomize_material=True,
    use_preloaded_materials=True,
    preloaded_materials_num_transform=20,
    preloaded_materials_num_color=100,
    dynamic_material_randomization=False,
    dynamic_material_randomization_interval=1.0,
)

# Optional: use pre-generated door USDs (fast startup) instead of procedural spawn_door.
# OPT-IN via DOOR_ASSET_DIR (default = procedural, which is the proven path).
# CAVEAT: the offline generator (generate_door_assets.py) currently saves geometry + joints +
# customData but NOT the rigid-body physics APIs that spawn_door applies at runtime, so loading the
# generated USDs via UsdFileCfg fails contact-sensor activation ("no rigid bodies present"). To use
# this path the generator must bake the rigid-body/articulation physics into the saved USD first.
_door_asset_dir = os.environ.get("DOOR_ASSET_DIR")  # unset -> procedural
_door_usd_files = sorted(glob.glob(os.path.join(_door_asset_dir, "*.usd"))) if _door_asset_dir else []

if _door_usd_files:
    _door_assets_cfg = [
        sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(_p),
            # NOTE: contact sensors are activated at the MultiAssetSpawnerCfg (env) level, not here —
            # per-asset activation runs on the uncomposed /World/Template proto prim and fails to find
            # the door's rigid bodies ("No contact sensors added ... no rigid bodies present").
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        )
        for _p in _door_usd_files
    ]
    _random_choice = True  # each env draws a random door from the pre-generated pool
    print(f"[door scenario] using {len(_door_usd_files)} pre-generated doors from {_door_asset_dir}")
else:
    _door_assets_cfg = [door_spawner_cfg] * 4096  # procedural fallback
    _random_choice = False
    print(f"[door scenario] no pre-generated doors in {_door_asset_dir}; using procedural spawn_door")

multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=_door_assets_cfg,
    random_choice=_random_choice,
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
)

TaskObjCfgDict = {
    "door": ArticulationCfg(
        spawn=multi_spawner_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*hinge.*": 0.0,
                ".*handle.*": 0.0,
                ".*latch.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=[".*hinge.*"],
                velocity_limit_sim=100.0,
                stiffness=None,
                damping=None,
            ),
            "handle": ImplicitActuatorCfg(
                joint_names_expr=[".*handle.*"],
                velocity_limit_sim=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )
}
