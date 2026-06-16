# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

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
    door_open_lr=["left"],
    door_open_io=["in"],
    door_handle_tblr=(0.95, 0.85, 0.08, 0.15),
    randomize_material=True,
    use_preloaded_materials=True,
    preloaded_materials_num_transform=20,
    preloaded_materials_num_color=100,
    dynamic_material_randomization=False,
    dynamic_material_randomization_interval=1.0,
)

multi_spawner_cfg = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[door_spawner_cfg] * 2048,
    random_choice=False,
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
