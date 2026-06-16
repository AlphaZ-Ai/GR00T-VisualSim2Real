# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Literal, Optional

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.kit.commands
import omni.usd
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_visual_material, clone
from isaaclab.utils import configclass
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

def _get_material_randomization():
    """Lazy import to avoid loading omni.kit.scripting before SimulationApp is ready."""
    from gr00t.rl.isaac_utils.playground.scripts.material_randomization import (
        get_material_randomization_class,
    )

    return get_material_randomization_class()


MR = None  # Will be lazily loaded on first use
from gr00t.rl.isaac_utils.playground.utils.math_utils import Rotation, set_prim_transform
from gr00t.rl.isaac_utils.playground.utils.usd_utils import (
    add_collider,
    add_mass,
    add_rigid_body,
    create_plane,
    create_prim,
    create_rect_light,
    preload_materials,
    write_custom_data_to_prim,
)


@configclass
class DoorSpawnerCfg(sim_utils.RigidObjectSpawnerCfg):
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = None
    door_width: tuple[float, float] = (0.8, 1.1)
    door_height: tuple[float, float] = (1.9, 2.2)
    door_handle_tblr: tuple[float, float, float, float] = (1.0, 0.85, 0.08, 0.15)
    door_handle_type: list[Literal["knob", "lever", "pushbar", "handle", "flat"]] = ["lever"]
    door_open_lr: list[Literal["left", "right"]] = ["left", "right"]
    door_open_io: list[Literal["in", "out"]] = ["in", "out"]
    door_weight: tuple[float, float] = (80.0, 120.0)
    add_walls: bool = False
    wall_minimum_clearance_fblr: tuple[float, float, float, float] = (3.0, 3.0, 1.0, 1.0)
    wall_maximum_clearance_fblr: tuple[float, float, float, float] = (10.0, 10.0, 10.0, 10.0)
    build_latch: bool = False
    add_floors: bool = False
    add_lights: bool = False
    add_ceiling: bool = False

    randomize_material: bool = False
    door_frame_material_prim_paths: list[str] = []
    door_panel_material_prim_paths: list[str] = []
    handle_material_prim_paths: list[str] = []
    wall_material_prim_paths: list[str] = []
    use_preloaded_materials: bool = False
    preloaded_materials_num_transform: int = 1
    preloaded_materials_num_color: int = 1

    dynamic_material_randomization: bool = False
    dynamic_material_randomization_interval: float = 1.0

    # deterministic generation
    rand_door_width: Optional[float] = None
    rand_door_height: Optional[float] = None
    rand_door_handle_height: Optional[float] = None
    rand_door_handle_width: Optional[float] = None
    rand_door_weight: Optional[float] = None
    rand_door_handle_type: Optional[Literal["knob", "lever", "pushbar", "handle", "flat"]] = None
    rand_door_open_lr: Optional[Literal["left", "right"]] = None
    rand_door_open_io: Optional[Literal["in", "out"]] = None
    rand_total_wall_height: Optional[float] = None
    rand_axle_length: Optional[float] = None
    rand_handle_length: Optional[float] = None
    rand_hook_length: Optional[float] = None
    rand_handle_radius: Optional[float] = None
    rand_spawn_hook: Optional[bool] = None
    rand_hinge_drive_max_force: Optional[float] = None
    rand_hinge_drive_stiffness: Optional[float] = None
    rand_handle_drive_max_force: Optional[float] = None
    rand_front: Optional[bool] = None
    rand_rear: Optional[bool] = None
    rand_left: Optional[bool] = None
    rand_left_front: Optional[bool] = None
    rand_right_front: Optional[bool] = None
    rand_left_rear: Optional[bool] = None
    rand_right_rear: Optional[bool] = None


def _update_joint_transform(stage, joint_path, prim0_path, prim1_path):
    # Get the joint prim
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        raise ValueError(f"Joint prim not found at path: {joint_path}")

    # Get the two prims
    prim0 = stage.GetPrimAtPath(prim0_path)
    prim1 = stage.GetPrimAtPath(prim1_path)
    if not prim0 or not prim1:
        raise ValueError(f"One or both prims not found at paths: {prim0_path}, {prim1_path}")

    # Create an XformCache
    xform_cache = UsdGeom.XformCache()

    # Get the relative transform between the two prims
    prim1_to_prim0, _ = xform_cache.ComputeRelativeTransform(prim1, prim0)

    # Get the joint's local position and orientation attributes
    local_pos0_attr = joint_prim.GetAttribute("physics:localPos0")
    local_orient0_attr = joint_prim.GetAttribute("physics:localRot0")
    local_pos1_attr = joint_prim.GetAttribute("physics:localPos1")
    local_orient1_attr = joint_prim.GetAttribute("physics:localRot1")

    prim0_to_joint = Gf.Matrix4d()
    prim0_to_joint = Gf.Matrix4d.SetTransform(
        prim0_to_joint,
        Gf.Rotation(Gf.Quatd(local_orient0_attr.Get())),
        Gf.Vec3d(local_pos0_attr.Get()),
    )

    # Compute the relative transform between prim1 and the joint
    relative_transform = prim0_to_joint * prim1_to_prim0.GetInverse()

    # Set the joint's local position and orientation to match the relative transform
    local_pos1_attr.Set(relative_transform.ExtractTranslation())
    local_orient1_attr.Set(Gf.Quatf(relative_transform.ExtractRotationQuat()))


@clone
def spawn_door(
    prim_path: str,
    cfg: DoorSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    global MR
    if MR is None:
        MR = _get_material_randomization()
    stage: Usd.Stage = omni.usd.get_context().get_stage()
    # determine the door width
    door_width = (
        np.random.uniform(cfg.door_width[0], cfg.door_width[1])
        if cfg.rand_door_width is None
        else cfg.rand_door_width
    )
    # determine the door height
    door_height = (
        np.random.uniform(cfg.door_height[0], cfg.door_height[1])
        if cfg.rand_door_height is None
        else cfg.rand_door_height
    )
    # determine the door handle location
    door_handle_height = (
        np.random.uniform(cfg.door_handle_tblr[1], cfg.door_handle_tblr[0])
        if cfg.rand_door_handle_height is None
        else cfg.rand_door_handle_height
    )
    door_handle_width = (
        np.random.uniform(cfg.door_handle_tblr[3], cfg.door_handle_tblr[2])
        if cfg.rand_door_handle_width is None
        else cfg.rand_door_handle_width
    )
    # determine the door weight
    door_weight = (
        np.random.uniform(cfg.door_weight[0], cfg.door_weight[1])
        if cfg.rand_door_weight is None
        else cfg.rand_door_weight
    )
    # determine the door handle type
    door_handle_type = (
        np.random.choice(cfg.door_handle_type)
        if cfg.rand_door_handle_type is None
        else cfg.rand_door_handle_type
    )
    # determine the door open direction
    door_open_lr = (
        np.random.choice(cfg.door_open_lr)
        if cfg.rand_door_open_lr is None
        else cfg.rand_door_open_lr
    )
    if door_open_lr == "left":
        door_open_lr = 1
    elif door_open_lr == "right":
        door_open_lr = -1
    else:
        raise ValueError(f"Invalid door open direction: {door_open_lr}")
    door_open_io = (
        np.random.choice(cfg.door_open_io)
        if cfg.rand_door_open_io is None
        else cfg.rand_door_open_io
    )
    if door_open_io == "in":
        door_open_io = 1
    elif door_open_io == "out":
        door_open_io = -1
    else:
        raise ValueError(f"Invalid door open direction: {door_open_io}")

    if cfg.randomize_material:
        if cfg.use_preloaded_materials:
            (
                door_frame_material_prim_paths,
                door_panel_material_prim_paths,
                handle_material_prim_paths,
            ) = preload_door_materials(
                stage, cfg.preloaded_materials_num_transform, cfg.preloaded_materials_num_color
            )
            wall_material_prim_paths = door_frame_material_prim_paths
        else:
            door_frame_material_prim_paths = cfg.door_frame_material_prim_paths
            door_panel_material_prim_paths = cfg.door_panel_material_prim_paths
            handle_material_prim_paths = cfg.handle_material_prim_paths
            wall_material_prim_paths = cfg.wall_material_prim_paths

    # spawn the prims
    create_prim(prim_path, "Xform")
    root_prim_path = os.path.join(prim_path, "root")
    create_prim(root_prim_path, "Xform")
    add_rigid_body(stage, root_prim_path)
    panel_prim_path = os.path.join(prim_path, "door_panel")
    create_prim(panel_prim_path, "Xform")
    add_rigid_body(stage, panel_prim_path)
    handle_prim_path = os.path.join(prim_path, "door_handle")
    create_prim(handle_prim_path, "Xform")
    add_rigid_body(stage, handle_prim_path)
    grasp_target_prim_path = os.path.join(prim_path, "grasp_target")
    create_prim(grasp_target_prim_path, "Xform")
    add_rigid_body(stage, grasp_target_prim_path)
    add_mass(stage, grasp_target_prim_path, mass=0.001)

    total_width = max(cfg.wall_maximum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[3]) * 2
    total_length = max(cfg.wall_maximum_clearance_fblr[0], cfg.wall_maximum_clearance_fblr[1]) * 2
    wall_total_length = (total_width - door_width) / 2
    half_wall_length = wall_total_length / 2
    half_door_width = door_width / 2
    half_door_height = door_height / 2
    total_wall_height = (
        np.random.uniform(2.4, 3.0)
        if cfg.rand_total_wall_height is None
        else cfg.rand_total_wall_height
    )
    half_wall_height = total_wall_height / 2
    gap_width = 0.002
    axle_length = (
        np.random.uniform(0.18, 0.21) if cfg.rand_axle_length is None else cfg.rand_axle_length
    )
    handle_length = (
        np.random.uniform(0.11, 0.14) if cfg.rand_handle_length is None else cfg.rand_handle_length
    )
    hook_length = (
        np.random.uniform(0.04, 0.06) if cfg.rand_hook_length is None else cfg.rand_hook_length
    )
    handle_radius = (
        np.random.uniform(0.011, 0.015)
        if cfg.rand_handle_radius is None
        else cfg.rand_handle_radius
    )

    # spawn the door cover
    covers_prim_path = os.path.join(root_prim_path, "covers")
    create_prim(covers_prim_path, "Scope")
    top_cover_prim_path = os.path.join(root_prim_path, "covers/top_cover")
    door_cover_width = np.random.uniform(0.03, 0.05)
    create_prim(top_cover_prim_path, "Cube")
    set_prim_transform(
        stage,
        top_cover_prim_path,
        (-0.02, 0, door_height + door_cover_width / 2),
        (0, 0, 0),
        (0.06, half_door_width + door_cover_width - gap_width, door_cover_width / 2),
    )

    left_cover_prim_path = os.path.join(root_prim_path, "covers/left_cover")
    create_prim(left_cover_prim_path, "Cube")
    set_prim_transform(
        stage,
        left_cover_prim_path,
        (-0.02, half_door_width + door_cover_width / 2 - gap_width, half_door_height),
        (0, 0, 0),
        (0.06, door_cover_width / 2, half_door_height),
    )

    right_cover_prim_path = os.path.join(root_prim_path, "covers/right_cover")
    create_prim(right_cover_prim_path, "Cube")
    set_prim_transform(
        stage,
        right_cover_prim_path,
        (-0.02, -half_door_width - door_cover_width / 2 + gap_width, half_door_height),
        (0, 0, 0),
        (0.06, door_cover_width / 2, half_door_height),
    )

    # spawn the door frame
    left_frame_prim_path = os.path.join(root_prim_path, "left_frame")
    create_prim(left_frame_prim_path, "Cube")
    set_prim_transform(
        stage,
        left_frame_prim_path,
        (-0.02, half_wall_length + half_door_width, half_wall_height),
        (0, 0, 0),
        (0.05, half_wall_length, half_wall_height),
    )
    add_mass(stage, left_frame_prim_path, mass=100.0)
    add_collider(stage, left_frame_prim_path)

    right_frame_prim_path = os.path.join(root_prim_path, "right_frame")
    create_prim(right_frame_prim_path, "Cube")
    set_prim_transform(
        stage,
        right_frame_prim_path,
        (-0.02, -half_wall_length - half_door_width, half_wall_height),
        (0, 0, 0),
        (0.05, half_wall_length, half_wall_height),
    )
    add_mass(stage, right_frame_prim_path, mass=100.0)
    add_collider(stage, right_frame_prim_path)

    top_frame_prim_path = os.path.join(root_prim_path, "top_frame")
    create_prim(top_frame_prim_path, "Cube")
    set_prim_transform(
        stage,
        top_frame_prim_path,
        (-0.02, 0, (total_wall_height - door_height) / 2 + door_height),
        (0, 0, 0),
        (0.05, half_door_width, (total_wall_height - door_height) / 2),
    )
    add_mass(stage, top_frame_prim_path, mass=100.0)
    add_collider(stage, top_frame_prim_path)

    # spawn the door panel
    panel_shape_prim_path = os.path.join(panel_prim_path, "panel")
    create_prim(panel_shape_prim_path, "Cube")
    set_prim_transform(
        stage,
        panel_shape_prim_path,
        (0, 0, half_door_height),
        (0, 0, 0),
        (0.02, half_door_width - gap_width, half_door_height - gap_width),
    )
    add_mass(stage, panel_shape_prim_path, mass=door_weight)
    add_collider(stage, panel_shape_prim_path)
    build_frame(
        stage, panel_prim_path, panel_shape_prim_path, door_width, door_height, 0.02, gap_width
    )

    # spawn the door handle
    # first, transform the handle prim to the door handle location
    set_prim_transform(
        stage,
        handle_prim_path,
        (0, (half_door_width - door_handle_width) * door_open_lr, door_handle_height),
        (0, 0, 0),
        (1.0, 1.0, 1.0),
    )
    axle_prim_path = os.path.join(handle_prim_path, "axle")
    # then, create the axle prim
    create_prim(axle_prim_path, "Cylinder")
    set_prim_transform(stage, axle_prim_path, (0, 0, 0), (0, 90, 0), (1.0, 1.0, 1.0))
    axle_geom: UsdGeom.Cylinder = UsdGeom.Cylinder.Define(stage, axle_prim_path)
    axle_geom.GetRadiusAttr().Set(handle_radius)
    axle_geom.GetHeightAttr().Set(axle_length)
    add_mass(stage, axle_prim_path, mass=0.2)
    add_collider(stage, axle_prim_path)

    # then, create the handle lever prim
    handle_shape_inside_prim_path = os.path.join(handle_prim_path, "handle_inside")
    create_prim(handle_shape_inside_prim_path, "Capsule")
    set_prim_transform(
        stage,
        handle_shape_inside_prim_path,
        (-axle_length / 2, (-handle_length / 2) * door_open_lr, 0),
        (90, 0, 0),
        (1.0, 1.0, 1.0),
    )
    handle_shape_inside_geom: UsdGeom.Capsule = UsdGeom.Capsule.Define(
        stage, handle_shape_inside_prim_path
    )
    handle_shape_inside_geom.GetRadiusAttr().Set(handle_radius)
    handle_shape_inside_geom.GetHeightAttr().Set(handle_length)
    add_mass(stage, handle_shape_inside_prim_path, mass=0.1)
    add_collider(stage, handle_shape_inside_prim_path)

    handle_shape_outside_prim_path = os.path.join(handle_prim_path, "handle_outside")
    create_prim(handle_shape_outside_prim_path, "Capsule")
    set_prim_transform(
        stage,
        handle_shape_outside_prim_path,
        (axle_length / 2, (-handle_length / 2) * door_open_lr, 0),
        (90, 0, 0),
        (1.0, 1.0, 1.0),
    )
    handle_shape_outside_geom: UsdGeom.Capsule = UsdGeom.Capsule.Define(
        stage, handle_shape_outside_prim_path
    )
    handle_shape_outside_geom.GetRadiusAttr().Set(handle_radius)
    handle_shape_outside_geom.GetHeightAttr().Set(handle_length)
    add_mass(stage, handle_shape_outside_prim_path, mass=0.1)
    add_collider(stage, handle_shape_outside_prim_path)

    spawn_hook = np.random.rand() < 0.5 if cfg.rand_spawn_hook is None else cfg.rand_spawn_hook
    if spawn_hook:
        # spawn the hook prim
        hook_inside_prim_path = os.path.join(handle_prim_path, "hook_inside")
        create_prim(hook_inside_prim_path, "Cylinder")
        set_prim_transform(
            stage,
            hook_inside_prim_path,
            (-axle_length / 2 + hook_length / 2, -handle_length * door_open_lr, 0),
            (0, 90, 0),
            (1.0, 1.0, 1.0),
        )
        hook_inside_geom: UsdGeom.Cylinder = UsdGeom.Cylinder.Define(stage, hook_inside_prim_path)
        hook_inside_geom.GetRadiusAttr().Set(handle_radius)
        hook_inside_geom.GetHeightAttr().Set(hook_length)
        add_mass(stage, hook_inside_prim_path, mass=0.05)
        add_collider(stage, hook_inside_prim_path)

        hook_outside_prim_path = os.path.join(handle_prim_path, "hook_outside")
        create_prim(hook_outside_prim_path, "Cylinder")
        set_prim_transform(
            stage,
            hook_outside_prim_path,
            (axle_length / 2 - hook_length / 2, -handle_length * door_open_lr, 0),
            (0, 90, 0),
            (1.0, 1.0, 1.0),
        )
        hook_outside_geom: UsdGeom.Cylinder = UsdGeom.Cylinder.Define(stage, hook_outside_prim_path)
        hook_outside_geom.GetRadiusAttr().Set(handle_radius)
        hook_outside_geom.GetHeightAttr().Set(hook_length)
        add_mass(stage, hook_outside_prim_path, mass=0.05)
        add_collider(stage, hook_outside_prim_path)

    # spawn keyhole
    if np.random.rand() < 0.5:
        keyhole_prim_path = os.path.join(panel_prim_path, "keyhole")
        create_prim(keyhole_prim_path, "Cylinder")
        set_prim_transform(
            stage,
            keyhole_prim_path,
            (
                0,
                (half_door_width - door_handle_width) * door_open_lr,
                door_handle_height + np.random.uniform(0.05, 0.1),
            ),
            (0, 90, 0),
            (1.0, 1.0, 1.0),
        )
        keyhole_geom: UsdGeom.Cylinder = UsdGeom.Cylinder.Define(stage, keyhole_prim_path)
        keyhole_geom.GetRadiusAttr().Set(0.02)
        keyhole_geom.GetHeightAttr().Set(0.07)

    # add articulation
    articulation_root_api = UsdPhysics.ArticulationRootAPI.Apply(
        stage.GetPrimAtPath(root_prim_path)
    )
    stage.GetPrimAtPath(root_prim_path).CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
    ).Set(cfg.build_latch)

    hinge_joint_prim_path = os.path.join(root_prim_path, "hinge_joint")
    hinge_joint = UsdPhysics.RevoluteJoint.Define(stage, hinge_joint_prim_path)
    hinge_joint.CreateBody0Rel().SetTargets([root_prim_path])
    hinge_joint.CreateBody1Rel().SetTargets([panel_prim_path])
    hinge_joint.GetAxisAttr().Set("Z")
    hinge_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.02, -half_door_width * door_open_lr, 0))
    # door_open_io: "in"(=1, pull) swings the panel TOWARD the robot, "out"(=-1, push) away. The
    # 180-deg-about-X local rot reverses the hinge swing direction; XOR it with the lr flip so pull
    # doors open toward the robot. Joint convention stays positive=open for both (limits [0,150]).
    if (door_open_lr == 1) != (door_open_io == 1):
        hinge_joint.CreateLocalRot0Attr().Set(Gf.Quatf(real=0.0, imaginary=(Gf.Vec3f(1, 0, 0))))
    hinge_joint.GetLowerLimitAttr().Set(0.0)
    hinge_joint.GetUpperLimitAttr().Set(150)
    hinge_drive = UsdPhysics.DriveAPI.Apply(hinge_joint.GetPrim(), "angular")
    hinge_drive.GetTargetPositionAttr().Set(-10.0)
    hinge_drive.GetMaxForceAttr().Set(
        np.random.uniform(2.5, 4.5)
        if cfg.rand_hinge_drive_max_force is None
        else cfg.rand_hinge_drive_max_force
    )
    hinge_drive.GetDampingAttr().Set(50.0)
    hinge_drive.GetStiffnessAttr().Set(
        np.random.uniform(1, 10.0)
        if cfg.rand_hinge_drive_stiffness is None
        else cfg.rand_hinge_drive_stiffness
    )
    _update_joint_transform(stage, hinge_joint_prim_path, root_prim_path, panel_prim_path)

    handle_joint_prim_path = os.path.join(panel_prim_path, "handle_joint")
    handle_joint = UsdPhysics.RevoluteJoint.Define(stage, handle_joint_prim_path)
    handle_joint.CreateBody0Rel().SetTargets([panel_prim_path])
    handle_joint.CreateBody1Rel().SetTargets([handle_prim_path])
    handle_joint.GetAxisAttr().Set("X")
    handle_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0, (half_door_width - door_handle_width) * door_open_lr, door_handle_height)
    )
    if door_open_lr == -1:
        handle_joint.CreateLocalRot0Attr().Set(Gf.Quatf(real=0.0, imaginary=(Gf.Vec3f(0, 0, 1))))
    handle_joint.GetLowerLimitAttr().Set(0.0)
    handle_joint.GetUpperLimitAttr().Set(45)
    handle_drive = UsdPhysics.DriveAPI.Apply(handle_joint.GetPrim(), "angular")
    handle_drive.GetTargetPositionAttr().Set(-15.0)
    handle_drive.GetMaxForceAttr().Set(
        np.random.uniform(1.0, 2.0)
        if cfg.rand_handle_drive_max_force is None
        else cfg.rand_handle_drive_max_force
    )
    handle_drive.GetDampingAttr().Set(0.5)
    handle_drive.GetStiffnessAttr().Set(50.0)
    _update_joint_transform(stage, handle_joint_prim_path, panel_prim_path, handle_prim_path)

    # build latch
    if cfg.build_latch:
        latch_link_prim_path = os.path.join(prim_path, "latch_link")
        create_prim(latch_link_prim_path, "Xform")
        set_prim_transform(
            stage,
            latch_link_prim_path,
            (-0.083, (half_door_width - 0.005) * door_open_lr, door_height - 0.1),
            (0, 0, 0),
            (1.0, 1.0, 1.0),
        )
        add_rigid_body(stage, latch_link_prim_path)

        latch_geom_prim_path = os.path.join(latch_link_prim_path, "latch_geom")
        create_prim(latch_geom_prim_path, "Cone")
        set_prim_transform(
            stage,
            latch_geom_prim_path,
            (0, 0, 0),
            (-90 * door_open_lr, 0, -26.56 * door_open_lr),
            (1.0, 1.0, 1.0),
        )
        cone_geom: UsdGeom.Cone = UsdGeom.Cone.Define(stage, latch_geom_prim_path)
        cone_geom.GetRadiusAttr().Set(0.025)
        cone_geom.GetHeightAttr().Set(0.05)
        cone_geom.GetPurposeAttr().Set("guide")
        add_mass(stage, latch_geom_prim_path, mass=0.1)
        add_collider(stage, latch_geom_prim_path)

        latch_joint_prim_path = os.path.join(panel_prim_path, "latch_joint")
        latch_joint = UsdPhysics.PrismaticJoint.Define(stage, latch_joint_prim_path)
        latch_joint.CreateBody0Rel().SetTargets([panel_prim_path])
        latch_joint.CreateBody1Rel().SetTargets([latch_link_prim_path])
        latch_joint.GetAxisAttr().Set("Y")
        latch_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(-0.083, (half_door_width - 0.005) * door_open_lr, door_height - 0.1)
        )
        if door_open_lr == 1:
            latch_joint.CreateLocalRot0Attr().Set(Gf.Quatf(real=0.0, imaginary=(Gf.Vec3f(0, 0, 1))))
        latch_joint.GetLowerLimitAttr().Set(0.0)
        latch_joint.GetUpperLimitAttr().Set(0.03)
        latch_mimic_joint = PhysxSchema.PhysxMimicJointAPI.Apply(
            latch_joint.GetPrim(), UsdPhysics.Tokens.rotX
        )
        latch_mimic_joint.GetReferenceJointRel().AddTarget(handle_joint_prim_path)
        latch_mimic_joint.GetGearingAttr().Set(-1.0 * 0.03 / 45.0)
        latch_mimic_joint.GetOffsetAttr().Set(0.0)
        _update_joint_transform(stage, latch_joint_prim_path, panel_prim_path, latch_link_prim_path)

    # adjust grasp target
    set_prim_transform(
        stage,
        grasp_target_prim_path,
        (
            -0.15,
            (half_door_width - door_handle_width - handle_length / 2) * door_open_lr,
            door_handle_height + 0.02,
        ),
        (0, 0, 0),
        (1.0, 1.0, 1.0),
    )
    grasp_target_joint_prim_path = os.path.join(handle_prim_path, "grasp_target_joint")
    grasp_target_joint = UsdPhysics.FixedJoint.Define(stage, grasp_target_joint_prim_path)
    grasp_target_joint.CreateBody0Rel().SetTargets([grasp_target_prim_path])
    grasp_target_joint.CreateBody1Rel().SetTargets([handle_prim_path])
    grasp_target_joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(-0.15, -handle_length / 2 * door_open_lr, 0.02)
    )
    # _update_joint_transform(stage, grasp_target_joint_prim_path, grasp_target_prim_path, handle_prim_path)

    # set material
    if cfg.randomize_material and door_frame_material_prim_paths and handle_material_prim_paths:
        door_frame_material_prim_path = np.random.choice(door_frame_material_prim_paths)
        door_panel_material_prim_path = np.random.choice(
            door_panel_material_prim_paths if door_panel_material_prim_paths else door_frame_material_prim_paths
        )
        handle_material_prim_path = np.random.choice(handle_material_prim_paths)
        bind_visual_material(left_frame_prim_path, door_frame_material_prim_path, stage)
        bind_visual_material(right_frame_prim_path, door_frame_material_prim_path, stage)
        bind_visual_material(top_frame_prim_path, door_frame_material_prim_path, stage)
        bind_visual_material(panel_prim_path, door_panel_material_prim_path, stage)
        bind_visual_material(handle_prim_path, handle_material_prim_path, stage)
        bind_visual_material(covers_prim_path, handle_material_prim_path, stage)

        if cfg.dynamic_material_randomization:
            MR.add_to_prim(
                stage,
                left_frame_prim_path,
                cfg.dynamic_material_randomization_interval,
                door_frame_material_prim_paths,
            )
            MR.add_to_prim(
                stage,
                right_frame_prim_path,
                cfg.dynamic_material_randomization_interval,
                door_frame_material_prim_paths,
            )
            MR.add_to_prim(
                stage,
                top_frame_prim_path,
                cfg.dynamic_material_randomization_interval,
                door_frame_material_prim_paths,
            )
            MR.add_to_prim(
                stage,
                panel_prim_path,
                cfg.dynamic_material_randomization_interval,
                door_panel_material_prim_paths,
            )
            MR.add_to_prim(
                stage,
                handle_prim_path,
                cfg.dynamic_material_randomization_interval,
                handle_material_prim_paths,
            )
            MR.add_to_prim(
                stage,
                covers_prim_path,
                cfg.dynamic_material_randomization_interval,
                handle_material_prim_paths,
            )

    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)

    # skip wall spawning logics
    if cfg.add_walls:
        wall_thickness = 0.05
        half_wall_thickness = wall_thickness / 2

        front = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[0], cfg.wall_maximum_clearance_fblr[0]
            )
            if cfg.rand_front is None
            else cfg.rand_front
        )
        rear = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[1], cfg.wall_maximum_clearance_fblr[1]
            )
            if cfg.rand_rear is None
            else cfg.rand_rear
        )
        left_front = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[2]
            )
            if cfg.rand_left_front is None
            else cfg.rand_left_front
        )
        right_front = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[3], cfg.wall_maximum_clearance_fblr[3]
            )
            if cfg.rand_right_front is None
            else cfg.rand_right_front
        )
        left_rear = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[2]
            )
            if cfg.rand_left_rear is None
            else cfg.rand_left_rear
        )
        right_rear = (
            np.random.uniform(
                cfg.wall_minimum_clearance_fblr[3], cfg.wall_maximum_clearance_fblr[3]
            )
            if cfg.rand_right_rear is None
            else cfg.rand_right_rear
        )

        # build front wall
        front_wall_prim_path = os.path.join(root_prim_path, "front_wall")
        create_prim(front_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            front_wall_prim_path,
            (-front, 0, total_wall_height / 2),
            (0, 0, 0),
            (half_wall_thickness, total_width / 2, total_wall_height / 2),
        )
        add_mass(stage, front_wall_prim_path, mass=100.0)
        add_collider(stage, front_wall_prim_path)

        # build rear wall
        rear_wall_prim_path = os.path.join(root_prim_path, "rear_wall")
        create_prim(rear_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            rear_wall_prim_path,
            (rear, 0, total_wall_height / 2),
            (0, 0, 0),
            (half_wall_thickness, total_width / 2, total_wall_height / 2),
        )
        add_mass(stage, rear_wall_prim_path, mass=100.0)
        add_collider(stage, rear_wall_prim_path)

        # build left front wall
        left_front_wall_prim_path = os.path.join(root_prim_path, "left_front_wall")
        create_prim(left_front_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            left_front_wall_prim_path,
            (-total_length / 4, left_front, total_wall_height / 2),
            (0, 0, 0),
            (total_length / 4, half_wall_thickness, total_wall_height / 2),
        )
        add_mass(stage, left_front_wall_prim_path, mass=100.0)
        add_collider(stage, left_front_wall_prim_path)

        # build right front wall
        right_front_wall_prim_path = os.path.join(root_prim_path, "right_front_wall")
        create_prim(right_front_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            right_front_wall_prim_path,
            (-total_length / 4, -right_front, total_wall_height / 2),
            (0, 0, 0),
            (total_length / 4, half_wall_thickness, total_wall_height / 2),
        )
        add_mass(stage, right_front_wall_prim_path, mass=100.0)
        add_collider(stage, right_front_wall_prim_path)

        # build left rear wall
        left_rear_wall_prim_path = os.path.join(root_prim_path, "left_rear_wall")
        create_prim(left_rear_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            left_rear_wall_prim_path,
            (total_length / 4, left_rear, total_wall_height / 2),
            (0, 0, 0),
            (total_length / 4, half_wall_thickness, total_wall_height / 2),
        )
        add_mass(stage, left_rear_wall_prim_path, mass=100.0)
        add_collider(stage, left_rear_wall_prim_path)

        # build right rear wall
        right_rear_wall_prim_path = os.path.join(root_prim_path, "right_rear_wall")
        create_prim(right_rear_wall_prim_path, "Cube")
        set_prim_transform(
            stage,
            right_rear_wall_prim_path,
            (total_length / 4, -right_rear, total_wall_height / 2),
            (0, 0, 0),
            (total_length / 4, half_wall_thickness, total_wall_height / 2),
        )
        add_mass(stage, right_rear_wall_prim_path, mass=100.0)
        add_collider(stage, right_rear_wall_prim_path)

        # set material
        if cfg.randomize_material and wall_material_prim_paths:
            bind_visual_material(
                front_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                rear_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                left_front_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                right_front_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                left_rear_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                right_rear_wall_prim_path, np.random.choice(wall_material_prim_paths), stage
            )

            if cfg.dynamic_material_randomization:
                MR.add_to_prim(
                    stage,
                    front_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    rear_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    left_front_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    right_front_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    left_rear_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    right_rear_wall_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )

    if cfg.add_ceiling:
        # spawn the ceiling
        ceiling_prim_path = os.path.join(root_prim_path, "ceiling")
        create_prim(ceiling_prim_path, "Cube")
        set_prim_transform(
            stage,
            ceiling_prim_path,
            (0, 0, total_wall_height + half_wall_thickness),
            (0, 0, 0),
            (total_length / 2, total_width / 2, half_wall_thickness),
        )
        if cfg.randomize_material and wall_material_prim_paths:
            bind_visual_material(
                ceiling_prim_path, np.random.choice(wall_material_prim_paths), stage
            )

    if cfg.add_floors:
        # build floor visual
        front_floor_prim_path = os.path.join(root_prim_path, "front_floor")
        create_plane(stage, front_floor_prim_path, (total_length / 2, total_width))
        set_prim_transform(
            stage, front_floor_prim_path, (-total_length / 4, 0, 0.001), (0, 0, 0), (1.0, 1.0, 1.0)
        )

        rear_floor_prim_path = os.path.join(root_prim_path, "rear_floor")
        create_plane(stage, rear_floor_prim_path, (total_length / 2, total_width))
        set_prim_transform(
            stage, rear_floor_prim_path, (total_length / 4, 0, 0.001), (0, 0, 0), (1.0, 1.0, 1.0)
        )

        if cfg.randomize_material and wall_material_prim_paths:
            bind_visual_material(
                front_floor_prim_path, np.random.choice(wall_material_prim_paths), stage
            )
            bind_visual_material(
                rear_floor_prim_path, np.random.choice(wall_material_prim_paths), stage
            )

            if cfg.dynamic_material_randomization:
                MR.add_to_prim(
                    stage,
                    front_floor_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )
                MR.add_to_prim(
                    stage,
                    rear_floor_prim_path,
                    cfg.dynamic_material_randomization_interval,
                    wall_material_prim_paths,
                )

    if cfg.add_lights:
        # build lights
        front_light_prim_path = os.path.join(root_prim_path, "front_light")
        create_rect_light(
            stage,
            front_light_prim_path,
            (1.2, 0.6),
            temperature=np.random.uniform(4000, 6000),
            intensity=np.random.uniform(4000, 5000),
        )
        set_prim_transform(
            stage,
            front_light_prim_path,
            (
                -front / 2,
                np.random.uniform(-right_front + 0.5, left_front - 0.5) / 2,
                total_wall_height - 0.05,
            ),
            (0, 0, 0),
            (1.0, 1.0, 1.0),
        )

        rear_light_prim_path = os.path.join(root_prim_path, "rear_light")
        create_rect_light(
            stage,
            rear_light_prim_path,
            (1.2, 0.6),
            temperature=np.random.uniform(4000, 6000),
            intensity=np.random.uniform(4000, 5000),
        )
        set_prim_transform(
            stage,
            rear_light_prim_path,
            (
                rear / 2,
                np.random.uniform(-right_rear + 0.5, left_rear - 0.5) / 2,
                total_wall_height - 0.05,
            ),
            (0, 0, 0),
            (1.0, 1.0, 1.0),
        )

    # encode some metadata
    metadata_key = "customData"
    metadata_value = {
        "doorWidth": door_width,
        "doorHeight": door_height,
        "doorHandleHeight": door_handle_height,
        "doorHandleWidth": door_handle_width,
        "doorWeight": door_weight,
        "doorHandleType": door_handle_type,
        "doorOpenLR": door_open_lr,
        "doorOpenIO": door_open_io,
        "totalWallHeight": total_wall_height,
        "axleLength": axle_length,
        "handleLength": handle_length,
        "hookLength": hook_length,
        "handleRadius": handle_radius,
        "spawnHook": spawn_hook,
        "hingeDriveMaxForce": hinge_drive.GetMaxForceAttr().Get(),
        "hingeDriveStiffness": hinge_drive.GetStiffnessAttr().Get(),
        "handleDriveMaxForce": handle_drive.GetMaxForceAttr().Get(),
    }
    if cfg.add_walls:
        metadata_value["front"] = front
        metadata_value["rear"] = rear
        metadata_value["leftFront"] = left_front
        metadata_value["rightFront"] = right_front
        metadata_value["leftRear"] = left_rear
        metadata_value["rightRear"] = right_rear

    write_custom_data_to_prim(stage, prim_path, metadata_value)

    if translation is None:
        translation = (0.0, 0.0, 0.0)
    if orientation is None:
        orientation = (1.0, 0.0, 0.0, 0.0)
    set_prim_transform(
        stage,
        prim_path,
        translation=translation,
        rotation=Rotation(*orientation).to_euler_xyz(degrees=True),
        scale=(1.0, 1.0, 1.0),
    )

    return prim_utils.get_prim_at_path(root_prim_path)


def preload_door_materials(
    stage: Usd.Stage, random_transform_each_num: int = 1, random_color_each_num: int = 1
):
    textured_material_list = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Ash.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Ash_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Bamboo.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Bamboo_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Beadboard.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Birch.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Birch_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cherry.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cherry_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Cork.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Mahogany.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Mahogany_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Oak.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Oak_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Parquet_Floor.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Plywood.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Timber.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Timber_Cladding.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Walnut.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Wood/Walnut_Planks.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Plastic.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Plastic_ABS.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Rubber_Smooth.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_OU_Walnut.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_UX_Walnut_Cherry.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Veneer_Z5_Maple.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Plastics/Vinyl.mdl",
    ]

    random_color_material_list = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Gloss_Finish.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Matte_Finish.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Miscellaneous/Paint_Satin_Finish.mdl",
    ]

    metallic_material_list = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Black.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Blue.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Charcoal.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Anodized_Red.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Cast.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Aluminum_Polished.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Brass.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Bronze.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Brushed_Antique_Copper.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Cast_Metal_Silver_Vein.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Chrome.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Copper.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/CorrugatedMetal.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Gold.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Iron.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Metal_Door.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Metal_Seamed_Roof.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/RustedMetal.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Silver.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Blued.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Carbon.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Cast.mdl",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/Steel_Stainless.mdl",
    ]

    door_frame_material_prim_paths = preload_materials(
        stage,
        textured_material_list,
        "/World/Looks/DoorFrameMaterials",
        random_transform_each_num=random_transform_each_num,
        random_color_material_list=random_color_material_list,
        random_color_each_num=random_color_each_num,
    )
    door_panel_material_prim_paths = preload_materials(
        stage,
        [],
        "/World/Looks/DoorPanelMaterials",
        random_transform_each_num=random_transform_each_num,
        random_color_material_list=random_color_material_list,
        random_color_each_num=random_color_each_num,
    )
    handle_material_prim_paths = preload_materials(
        stage,
        metallic_material_list,
        "/World/Looks/HandleMaterials",
        random_transform_each_num=random_transform_each_num,
    )

    return (
        door_frame_material_prim_paths,
        door_panel_material_prim_paths,
        handle_material_prim_paths,
    )


def get_deterministic_door_config(cfg: DoorSpawnerCfg, metadata: dict) -> DoorSpawnerCfg:
    cfg.rand_door_width = metadata["doorWidth"]
    cfg.rand_door_height = metadata["doorHeight"]
    cfg.rand_door_handle_height = metadata["doorHandleHeight"]
    cfg.rand_door_handle_width = metadata["doorHandleWidth"]
    cfg.rand_door_weight = metadata["doorWeight"]
    cfg.rand_door_handle_type = metadata["doorHandleType"]
    cfg.rand_door_open_lr = metadata["doorOpenLR"]
    cfg.rand_door_open_io = metadata["doorOpenIO"]
    cfg.rand_total_wall_height = metadata["totalWallHeight"]
    cfg.rand_axle_length = metadata["axleLength"]
    cfg.rand_handle_length = metadata["handleLength"]
    cfg.rand_hook_length = metadata["hookLength"]
    cfg.rand_handle_radius = metadata["handleRadius"]
    cfg.rand_spawn_hook = metadata["spawnHook"]
    cfg.rand_hinge_drive_max_force = metadata["hingeDriveMaxForce"]
    cfg.rand_hinge_drive_stiffness = metadata["hingeDriveStiffness"]
    cfg.rand_handle_drive_max_force = metadata["handleDriveMaxForce"]

    if cfg.add_walls:
        cfg.rand_front = metadata["front"]
        cfg.rand_rear = metadata["rear"]
        cfg.rand_left_front = metadata["leftFront"]
        cfg.rand_right_front = metadata["rightFront"]
        cfg.rand_left_rear = metadata["leftRear"]
        cfg.rand_right_rear = metadata["rightRear"]
    return cfg


def build_frame(
    stage: Usd.Stage,
    door_panel_prim_path: str,
    door_panel_geom_prim_path: str,
    width: float,
    height: float,
    thickness: float,
    gap_width: float,
):
    width = width - 2 * gap_width
    height = height - 2 * gap_width
    frame_width = np.random.uniform(0.04, 0.08)
    num_subpanels = np.random.randint(0, 5)
    if num_subpanels <= 2:
        frame_width = np.random.uniform(0.1, 0.15)
    reserve_bottom = np.random.uniform(0.0, 0.15)

    if num_subpanels == 0:
        return
    else:
        door_panel_geom: UsdGeom.Cube = UsdGeom.Cube.Define(stage, door_panel_geom_prim_path)
        door_panel_geom.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    subpanel_height = (height - reserve_bottom - (num_subpanels + 1) * frame_width) / num_subpanels
    subpanel_width = width - 2 * frame_width

    half_width = width / 2
    half_frame_width = frame_width / 2
    half_thickness = thickness / 2
    half_subpanel_width = subpanel_width / 2
    half_height = height / 2

    for i in range(num_subpanels):
        subpanel_top_frame_prim_path = os.path.join(door_panel_prim_path, f"subpanel_{i}_top_frame")
        create_prim(subpanel_top_frame_prim_path, "Cube")
        top_panel_trans_z = height - (subpanel_height + frame_width) * i - half_frame_width
        set_prim_transform(
            stage,
            subpanel_top_frame_prim_path,
            (0, 0, top_panel_trans_z + gap_width),
            (0, 0, 0),
            (half_thickness, half_subpanel_width, half_frame_width),
        )

        if i == num_subpanels - 1:
            subpanel_bottom_frame_prim_path = os.path.join(
                door_panel_prim_path, f"subpanel_{i}_bottom_frame"
            )
            create_prim(subpanel_bottom_frame_prim_path, "Cube")
            bottom_panel_trans_z = (
                height
                - (subpanel_height + frame_width) * i
                - frame_width
                - subpanel_height
                - half_frame_width
            )
            set_prim_transform(
                stage,
                subpanel_bottom_frame_prim_path,
                (0, 0, bottom_panel_trans_z + gap_width),
                (0, 0, 0),
                (half_thickness, half_subpanel_width, half_frame_width),
            )
            reserve_bottom_frame_prim_path = os.path.join(
                door_panel_prim_path, "reserve_bottom_frame"
            )
            create_prim(reserve_bottom_frame_prim_path, "Cube")
            reserve_bottom_frame_trans_z = reserve_bottom / 2 + gap_width
            set_prim_transform(
                stage,
                reserve_bottom_frame_prim_path,
                (0, 0, reserve_bottom_frame_trans_z),
                (0, 0, 0),
                (half_thickness, half_subpanel_width, reserve_bottom / 2),
            )

    left_frame_prim_path = os.path.join(door_panel_prim_path, "left_frame")
    create_prim(left_frame_prim_path, "Cube")
    set_prim_transform(
        stage,
        left_frame_prim_path,
        (0, half_width - half_frame_width, half_height + gap_width),
        (0, 0, 0),
        (half_thickness, half_frame_width, half_height),
    )

    right_frame_prim_path = os.path.join(door_panel_prim_path, "right_frame")
    create_prim(right_frame_prim_path, "Cube")
    set_prim_transform(
        stage,
        right_frame_prim_path,
        (0, -half_width + half_frame_width, half_height + gap_width),
        (0, 0, 0),
        (half_thickness, half_frame_width, half_height),
    )
