# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate randomized articulated door USD assets offline.

This script creates N door USD files with randomized geometry, handle placement,
joint parameters, and optional latch mechanisms — matching the same distributions
used during online training in spawn_door().

Usage:
    python gr00t/rl/scripts/generate_door_assets.py \
        --num_doors 100 \
        --output_dir data/door_assets \
        --seed 42 \
        --build_latch

    # Match training config exactly:
    python gr00t/rl/scripts/generate_door_assets.py \
        --num_doors 100 \
        --output_dir data/door_assets \
        --build_latch \
        --add_floors \
        --door_open_lr right \
        --door_open_io out
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate randomized door USD assets offline.")
    parser.add_argument("--num_doors", type=int, default=100, help="Number of door assets to generate.")
    parser.add_argument("--output_dir", type=str, default="data/door_assets", help="Output directory for USD files.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--build_latch", action="store_true", help="Build latch mechanism on doors.")
    parser.add_argument("--add_floors", action="store_true", help="Add floor planes to door assets.")
    parser.add_argument("--add_walls", action="store_true", help="Add surrounding walls.")
    parser.add_argument(
        "--door_open_lr",
        nargs="+",
        default=["right"],
        choices=["left", "right"],
        help="Door hinge side(s) to sample from.",
    )
    parser.add_argument(
        "--door_open_io",
        nargs="+",
        default=["out"],
        choices=["in", "out"],
        help="Door opening direction(s) to sample from.",
    )
    parser.add_argument(
        "--door_handle_tblr",
        nargs=4,
        type=float,
        default=[0.95, 0.85, 0.08, 0.15],
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Handle position range as fraction of door (top, bottom, left, right).",
    )
    parser.add_argument(
        "--randomize_material",
        action="store_true",
        help="Apply random Omniverse materials (requires Nucleus access).",
    )
    parser.add_argument(
        "--preloaded_materials_num_transform",
        type=int,
        default=20,
        help="Number of UV-transform variants per textured material.",
    )
    parser.add_argument(
        "--preloaded_materials_num_color",
        type=int,
        default=100,
        help="Number of random-color paint material variants.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Launch Isaac Sim headless via isaaclab's AppLauncher (not a bare SimulationApp):
    # the isaaclab extensions (e.g. isaaclab.sim) are only registered when the app is
    # launched with the isaaclab experience, which AppLauncher does. A raw SimulationApp
    # boots without them, causing `ModuleNotFoundError: No module named 'isaaclab.sim'`.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import numpy as np
    import omni.usd
    from pxr import UsdGeom

    from gr00t.rl.isaac_utils.playground.env_rand.door import (
        DoorSpawnerCfg,
        _update_joint_transform,
        build_frame,
    )
    from gr00t.rl.isaac_utils.playground.utils.math_utils import set_prim_transform
    from gr00t.rl.isaac_utils.playground.utils.usd_utils import (
        add_collider,
        add_mass,
        add_rigid_body,
        create_plane,
        create_prim,
        write_custom_data_to_prim,
    )

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.randomize_material:
        from gr00t.rl.isaac_utils.playground.env_rand.door import preload_door_materials

    cfg = DoorSpawnerCfg(
        build_latch=args.build_latch,
        add_floors=args.add_floors,
        add_walls=args.add_walls,
        door_open_lr=args.door_open_lr,
        door_open_io=args.door_open_io,
        door_handle_tblr=tuple(args.door_handle_tblr),
        randomize_material=args.randomize_material,
    )

    metadata_all = {}

    for i in range(args.num_doors):
        # Create a fresh stage for each door
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Preload materials on this stage so they are embedded in the exported USD
        material_prim_paths = None
        if args.randomize_material:
            material_prim_paths = preload_door_materials(
                stage,
                args.preloaded_materials_num_transform,
                args.preloaded_materials_num_color,
            )

        # Build the door at a TOP-LEVEL prim so it can be set as the stage default prim (USD requires
        # the default prim to be a root prim). This makes the asset referenceable via UsdFileCfg.
        prim_path = "/Door"

        metadata = _build_door(
            stage,
            prim_path,
            cfg,
            material_prim_paths=material_prim_paths,
            # Pass through all the utility functions so we don't re-import
            _create_prim=create_prim,
            _add_rigid_body=add_rigid_body,
            _add_mass=add_mass,
            _add_collider=add_collider,
            _set_prim_transform=set_prim_transform,
            _create_plane=create_plane,
            _write_custom_data=write_custom_data_to_prim,
            _update_joint_transform=_update_joint_transform,
            _build_frame=build_frame,
        )

        # Set the door root as the stage default prim so the asset can be referenced via
        # sim_utils.UsdFileCfg (MultiAssetSpawnerCfg). Without a default prim the reference brings in
        # nothing usable and contact-sensor activation fails with "no rigid bodies present".
        door_root_prim = stage.GetPrimAtPath(prim_path)
        if door_root_prim and door_root_prim.IsValid():
            stage.SetDefaultPrim(door_root_prim)

        output_path = os.path.abspath(os.path.join(args.output_dir, f"door_{i:04d}.usd"))
        stage.Export(output_path)
        metadata_all[f"door_{i:04d}"] = metadata
        print(f"[{i + 1}/{args.num_doors}] {output_path}")

    # Save metadata index
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_all, f, indent=2)
    print(f"\nGenerated {args.num_doors} door assets in {os.path.abspath(args.output_dir)}/")
    print(f"Metadata saved to {metadata_path}")

    simulation_app.close()


def _build_door(
    stage,
    prim_path,
    cfg,
    material_prim_paths=None,
    *,
    _create_prim,
    _add_rigid_body,
    _add_mass,
    _add_collider,
    _set_prim_transform,
    _create_plane,
    _write_custom_data,
    _update_joint_transform,
    _build_frame,
):
    """Build a single door on the given stage.

    Mirrors the logic in spawn_door() but without the @clone decorator and
    isaaclab spawner framework dependency, so it can run standalone.
    """
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from isaaclab.sim.utils import bind_visual_material

    # --- Sample randomized parameters (same distributions as spawn_door) ---
    door_width = np.random.uniform(cfg.door_width[0], cfg.door_width[1])
    door_height = np.random.uniform(cfg.door_height[0], cfg.door_height[1])
    door_handle_height = np.random.uniform(cfg.door_handle_tblr[1], cfg.door_handle_tblr[0])
    door_handle_width = np.random.uniform(cfg.door_handle_tblr[3], cfg.door_handle_tblr[2])
    door_weight = np.random.uniform(cfg.door_weight[0], cfg.door_weight[1])
    door_handle_type = np.random.choice(cfg.door_handle_type)

    door_open_lr = np.random.choice(cfg.door_open_lr)
    door_open_lr = 1 if door_open_lr == "left" else -1
    door_open_io = np.random.choice(cfg.door_open_io)
    door_open_io = 1 if door_open_io == "in" else -1

    total_width = max(cfg.wall_maximum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[3]) * 2
    total_length = max(cfg.wall_maximum_clearance_fblr[0], cfg.wall_maximum_clearance_fblr[1]) * 2
    wall_total_length = (total_width - door_width) / 2
    half_wall_length = wall_total_length / 2
    half_door_width = door_width / 2
    half_door_height = door_height / 2
    total_wall_height = np.random.uniform(2.4, 3.0)
    half_wall_height = total_wall_height / 2
    gap_width = 0.002
    axle_length = np.random.uniform(0.18, 0.21)
    handle_length = np.random.uniform(0.11, 0.14)
    hook_length = np.random.uniform(0.04, 0.06)
    handle_radius = np.random.uniform(0.011, 0.015)

    # --- Create prim hierarchy ---
    _create_prim(prim_path, "Xform")
    root_prim_path = os.path.join(prim_path, "root")
    _create_prim(root_prim_path, "Xform")
    _add_rigid_body(stage, root_prim_path)
    panel_prim_path = os.path.join(prim_path, "door_panel")
    _create_prim(panel_prim_path, "Xform")
    _add_rigid_body(stage, panel_prim_path)
    handle_prim_path = os.path.join(prim_path, "door_handle")
    _create_prim(handle_prim_path, "Xform")
    _add_rigid_body(stage, handle_prim_path)
    grasp_target_prim_path = os.path.join(prim_path, "grasp_target")
    _create_prim(grasp_target_prim_path, "Xform")
    _add_rigid_body(stage, grasp_target_prim_path)
    _add_mass(stage, grasp_target_prim_path, mass=0.001)

    # --- Door covers ---
    covers_prim_path = os.path.join(root_prim_path, "covers")
    _create_prim(covers_prim_path, "Scope")
    door_cover_width = np.random.uniform(0.03, 0.05)

    top_cover_prim_path = os.path.join(root_prim_path, "covers/top_cover")
    _create_prim(top_cover_prim_path, "Cube")
    _set_prim_transform(
        stage,
        top_cover_prim_path,
        (-0.02, 0, door_height + door_cover_width / 2),
        (0, 0, 0),
        (0.06, half_door_width + door_cover_width - gap_width, door_cover_width / 2),
    )

    left_cover_prim_path = os.path.join(root_prim_path, "covers/left_cover")
    _create_prim(left_cover_prim_path, "Cube")
    _set_prim_transform(
        stage,
        left_cover_prim_path,
        (-0.02, half_door_width + door_cover_width / 2 - gap_width, half_door_height),
        (0, 0, 0),
        (0.06, door_cover_width / 2, half_door_height),
    )

    right_cover_prim_path = os.path.join(root_prim_path, "covers/right_cover")
    _create_prim(right_cover_prim_path, "Cube")
    _set_prim_transform(
        stage,
        right_cover_prim_path,
        (-0.02, -half_door_width - door_cover_width / 2 + gap_width, half_door_height),
        (0, 0, 0),
        (0.06, door_cover_width / 2, half_door_height),
    )

    # --- Door frame ---
    left_frame_prim_path = os.path.join(root_prim_path, "left_frame")
    _create_prim(left_frame_prim_path, "Cube")
    _set_prim_transform(
        stage,
        left_frame_prim_path,
        (-0.02, half_wall_length + half_door_width, half_wall_height),
        (0, 0, 0),
        (0.05, half_wall_length, half_wall_height),
    )
    _add_mass(stage, left_frame_prim_path, mass=100.0)
    _add_collider(stage, left_frame_prim_path)

    right_frame_prim_path = os.path.join(root_prim_path, "right_frame")
    _create_prim(right_frame_prim_path, "Cube")
    _set_prim_transform(
        stage,
        right_frame_prim_path,
        (-0.02, -half_wall_length - half_door_width, half_wall_height),
        (0, 0, 0),
        (0.05, half_wall_length, half_wall_height),
    )
    _add_mass(stage, right_frame_prim_path, mass=100.0)
    _add_collider(stage, right_frame_prim_path)

    top_frame_prim_path = os.path.join(root_prim_path, "top_frame")
    _create_prim(top_frame_prim_path, "Cube")
    _set_prim_transform(
        stage,
        top_frame_prim_path,
        (-0.02, 0, (total_wall_height - door_height) / 2 + door_height),
        (0, 0, 0),
        (0.05, half_door_width, (total_wall_height - door_height) / 2),
    )
    _add_mass(stage, top_frame_prim_path, mass=100.0)
    _add_collider(stage, top_frame_prim_path)

    # --- Door panel ---
    panel_shape_prim_path = os.path.join(panel_prim_path, "panel")
    _create_prim(panel_shape_prim_path, "Cube")
    _set_prim_transform(
        stage,
        panel_shape_prim_path,
        (0, 0, half_door_height),
        (0, 0, 0),
        (0.02, half_door_width - gap_width, half_door_height - gap_width),
    )
    _add_mass(stage, panel_shape_prim_path, mass=door_weight)
    _add_collider(stage, panel_shape_prim_path)
    _build_frame(stage, panel_prim_path, panel_shape_prim_path, door_width, door_height, 0.02, gap_width)

    # --- Door handle ---
    _set_prim_transform(
        stage,
        handle_prim_path,
        (0, (half_door_width - door_handle_width) * door_open_lr, door_handle_height),
        (0, 0, 0),
        (1.0, 1.0, 1.0),
    )

    axle_prim_path = os.path.join(handle_prim_path, "axle")
    _create_prim(axle_prim_path, "Cylinder")
    _set_prim_transform(stage, axle_prim_path, (0, 0, 0), (0, 90, 0), (1.0, 1.0, 1.0))
    axle_geom = UsdGeom.Cylinder.Define(stage, axle_prim_path)
    axle_geom.GetRadiusAttr().Set(handle_radius)
    axle_geom.GetHeightAttr().Set(axle_length)
    _add_mass(stage, axle_prim_path, mass=0.2)
    _add_collider(stage, axle_prim_path)

    handle_shape_inside_prim_path = os.path.join(handle_prim_path, "handle_inside")
    _create_prim(handle_shape_inside_prim_path, "Capsule")
    _set_prim_transform(
        stage,
        handle_shape_inside_prim_path,
        (-axle_length / 2, (-handle_length / 2) * door_open_lr, 0),
        (90, 0, 0),
        (1.0, 1.0, 1.0),
    )
    handle_shape_inside_geom = UsdGeom.Capsule.Define(stage, handle_shape_inside_prim_path)
    handle_shape_inside_geom.GetRadiusAttr().Set(handle_radius)
    handle_shape_inside_geom.GetHeightAttr().Set(handle_length)
    _add_mass(stage, handle_shape_inside_prim_path, mass=0.1)
    _add_collider(stage, handle_shape_inside_prim_path)

    handle_shape_outside_prim_path = os.path.join(handle_prim_path, "handle_outside")
    _create_prim(handle_shape_outside_prim_path, "Capsule")
    _set_prim_transform(
        stage,
        handle_shape_outside_prim_path,
        (axle_length / 2, (-handle_length / 2) * door_open_lr, 0),
        (90, 0, 0),
        (1.0, 1.0, 1.0),
    )
    handle_shape_outside_geom = UsdGeom.Capsule.Define(stage, handle_shape_outside_prim_path)
    handle_shape_outside_geom.GetRadiusAttr().Set(handle_radius)
    handle_shape_outside_geom.GetHeightAttr().Set(handle_length)
    _add_mass(stage, handle_shape_outside_prim_path, mass=0.1)
    _add_collider(stage, handle_shape_outside_prim_path)

    # --- Optional hook ---
    spawn_hook = np.random.rand() < 0.5
    if spawn_hook:
        hook_inside_prim_path = os.path.join(handle_prim_path, "hook_inside")
        _create_prim(hook_inside_prim_path, "Cylinder")
        _set_prim_transform(
            stage,
            hook_inside_prim_path,
            (-axle_length / 2 + hook_length / 2, -handle_length * door_open_lr, 0),
            (0, 90, 0),
            (1.0, 1.0, 1.0),
        )
        hook_inside_geom = UsdGeom.Cylinder.Define(stage, hook_inside_prim_path)
        hook_inside_geom.GetRadiusAttr().Set(handle_radius)
        hook_inside_geom.GetHeightAttr().Set(hook_length)
        _add_mass(stage, hook_inside_prim_path, mass=0.05)
        _add_collider(stage, hook_inside_prim_path)

        hook_outside_prim_path = os.path.join(handle_prim_path, "hook_outside")
        _create_prim(hook_outside_prim_path, "Cylinder")
        _set_prim_transform(
            stage,
            hook_outside_prim_path,
            (axle_length / 2 - hook_length / 2, -handle_length * door_open_lr, 0),
            (0, 90, 0),
            (1.0, 1.0, 1.0),
        )
        hook_outside_geom = UsdGeom.Cylinder.Define(stage, hook_outside_prim_path)
        hook_outside_geom.GetRadiusAttr().Set(handle_radius)
        hook_outside_geom.GetHeightAttr().Set(hook_length)
        _add_mass(stage, hook_outside_prim_path, mass=0.05)
        _add_collider(stage, hook_outside_prim_path)

    # --- Optional keyhole ---
    if np.random.rand() < 0.5:
        keyhole_prim_path = os.path.join(panel_prim_path, "keyhole")
        _create_prim(keyhole_prim_path, "Cylinder")
        _set_prim_transform(
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
        keyhole_geom = UsdGeom.Cylinder.Define(stage, keyhole_prim_path)
        keyhole_geom.GetRadiusAttr().Set(0.02)
        keyhole_geom.GetHeightAttr().Set(0.07)

    # --- Articulation ---
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_prim_path))
    stage.GetPrimAtPath(root_prim_path).CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
    ).Set(cfg.build_latch)

    # Hinge joint
    hinge_joint_prim_path = os.path.join(root_prim_path, "hinge_joint")
    hinge_joint = UsdPhysics.RevoluteJoint.Define(stage, hinge_joint_prim_path)
    hinge_joint.CreateBody0Rel().SetTargets([root_prim_path])
    hinge_joint.CreateBody1Rel().SetTargets([panel_prim_path])
    hinge_joint.GetAxisAttr().Set("Z")
    hinge_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.02, -half_door_width * door_open_lr, 0))
    if door_open_lr == 1:
        hinge_joint.CreateLocalRot0Attr().Set(Gf.Quatf(real=0.0, imaginary=(Gf.Vec3f(1, 0, 0))))
    hinge_joint.GetLowerLimitAttr().Set(0.0)
    hinge_joint.GetUpperLimitAttr().Set(150)
    hinge_drive = UsdPhysics.DriveAPI.Apply(hinge_joint.GetPrim(), "angular")
    hinge_drive.GetTargetPositionAttr().Set(-10.0)
    hinge_drive_max_force = np.random.uniform(2.5, 4.5)
    hinge_drive.GetMaxForceAttr().Set(hinge_drive_max_force)
    hinge_drive.GetDampingAttr().Set(50.0)
    hinge_drive_stiffness = np.random.uniform(1.0, 10.0)
    hinge_drive.GetStiffnessAttr().Set(hinge_drive_stiffness)
    _update_joint_transform(stage, hinge_joint_prim_path, root_prim_path, panel_prim_path)

    # Handle joint
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
    handle_drive_max_force = np.random.uniform(1.0, 2.0)
    handle_drive.GetMaxForceAttr().Set(handle_drive_max_force)
    handle_drive.GetDampingAttr().Set(0.5)
    handle_drive.GetStiffnessAttr().Set(50.0)
    _update_joint_transform(stage, handle_joint_prim_path, panel_prim_path, handle_prim_path)

    # --- Optional latch ---
    if cfg.build_latch:
        latch_link_prim_path = os.path.join(prim_path, "latch_link")
        _create_prim(latch_link_prim_path, "Xform")
        _set_prim_transform(
            stage,
            latch_link_prim_path,
            (-0.083, (half_door_width - 0.005) * door_open_lr, door_height - 0.1),
            (0, 0, 0),
            (1.0, 1.0, 1.0),
        )
        _add_rigid_body(stage, latch_link_prim_path)

        latch_geom_prim_path = os.path.join(latch_link_prim_path, "latch_geom")
        _create_prim(latch_geom_prim_path, "Cone")
        _set_prim_transform(
            stage,
            latch_geom_prim_path,
            (0, 0, 0),
            (-90 * door_open_lr, 0, -26.56 * door_open_lr),
            (1.0, 1.0, 1.0),
        )
        cone_geom = UsdGeom.Cone.Define(stage, latch_geom_prim_path)
        cone_geom.GetRadiusAttr().Set(0.025)
        cone_geom.GetHeightAttr().Set(0.05)
        cone_geom.GetPurposeAttr().Set("guide")
        _add_mass(stage, latch_geom_prim_path, mass=0.1)
        _add_collider(stage, latch_geom_prim_path)

        latch_joint_prim_path = os.path.join(panel_prim_path, "latch_joint")
        latch_joint = UsdPhysics.PrismaticJoint.Define(stage, latch_joint_prim_path)
        latch_joint.CreateBody0Rel().SetTargets([panel_prim_path])
        latch_joint.CreateBody1Rel().SetTargets([latch_link_prim_path])
        latch_joint.GetAxisAttr().Set("Y")
        latch_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(-0.083, (half_door_width - 0.005) * door_open_lr, door_height - 0.1)
        )
        if door_open_lr == 1:
            latch_joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(real=0.0, imaginary=(Gf.Vec3f(0, 0, 1)))
            )
        latch_joint.GetLowerLimitAttr().Set(0.0)
        latch_joint.GetUpperLimitAttr().Set(0.03)
        latch_mimic_joint = PhysxSchema.PhysxMimicJointAPI.Apply(
            latch_joint.GetPrim(), UsdPhysics.Tokens.rotX
        )
        latch_mimic_joint.GetReferenceJointRel().AddTarget(handle_joint_prim_path)
        latch_mimic_joint.GetGearingAttr().Set(-1.0 * 0.03 / 45.0)
        latch_mimic_joint.GetOffsetAttr().Set(0.0)
        _update_joint_transform(stage, latch_joint_prim_path, panel_prim_path, latch_link_prim_path)

    # --- Grasp target ---
    _set_prim_transform(
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

    # --- Material binding ---
    door_frame_mats = None
    if material_prim_paths is not None:
        door_frame_mats, door_panel_mats, handle_mats = material_prim_paths
        if door_frame_mats and handle_mats:
            door_frame_mat = np.random.choice(door_frame_mats)
            door_panel_mat = np.random.choice(door_panel_mats if door_panel_mats else door_frame_mats)
            handle_mat = np.random.choice(handle_mats)
            bind_visual_material(left_frame_prim_path, door_frame_mat, stage)
            bind_visual_material(right_frame_prim_path, door_frame_mat, stage)
            bind_visual_material(top_frame_prim_path, door_frame_mat, stage)
            bind_visual_material(panel_prim_path, door_panel_mat, stage)
            bind_visual_material(handle_prim_path, handle_mat, stage)
            bind_visual_material(covers_prim_path, handle_mat, stage)

    # --- Articulation properties ---
    from isaaclab.sim import schemas
    import isaaclab.sim as sim_utils

    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=4,
        fix_root_link=True,
    )
    schemas.modify_articulation_root_properties(prim_path, articulation_props)

    # --- Optional walls ---
    if cfg.add_walls:
        wall_thickness = 0.05
        half_wall_thickness = wall_thickness / 2

        front = np.random.uniform(cfg.wall_minimum_clearance_fblr[0], cfg.wall_maximum_clearance_fblr[0])
        rear = np.random.uniform(cfg.wall_minimum_clearance_fblr[1], cfg.wall_maximum_clearance_fblr[1])
        left_front = np.random.uniform(cfg.wall_minimum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[2])
        right_front = np.random.uniform(cfg.wall_minimum_clearance_fblr[3], cfg.wall_maximum_clearance_fblr[3])
        left_rear = np.random.uniform(cfg.wall_minimum_clearance_fblr[2], cfg.wall_maximum_clearance_fblr[2])
        right_rear = np.random.uniform(cfg.wall_minimum_clearance_fblr[3], cfg.wall_maximum_clearance_fblr[3])

        wall_prim_paths = []
        for name, pos, scale in [
            ("front_wall", (-front, 0, total_wall_height / 2), (half_wall_thickness, total_width / 2, total_wall_height / 2)),
            ("rear_wall", (rear, 0, total_wall_height / 2), (half_wall_thickness, total_width / 2, total_wall_height / 2)),
            ("left_front_wall", (-total_length / 4, left_front, total_wall_height / 2), (total_length / 4, half_wall_thickness, total_wall_height / 2)),
            ("right_front_wall", (-total_length / 4, -right_front, total_wall_height / 2), (total_length / 4, half_wall_thickness, total_wall_height / 2)),
            ("left_rear_wall", (total_length / 4, left_rear, total_wall_height / 2), (total_length / 4, half_wall_thickness, total_wall_height / 2)),
            ("right_rear_wall", (total_length / 4, -right_rear, total_wall_height / 2), (total_length / 4, half_wall_thickness, total_wall_height / 2)),
        ]:
            wp = os.path.join(root_prim_path, name)
            _create_prim(wp, "Cube")
            _set_prim_transform(stage, wp, pos, (0, 0, 0), scale)
            _add_mass(stage, wp, mass=100.0)
            _add_collider(stage, wp)
            wall_prim_paths.append(wp)

        if door_frame_mats:
            for wp in wall_prim_paths:
                bind_visual_material(wp, np.random.choice(door_frame_mats), stage)

    # --- Optional floors ---
    if cfg.add_floors:
        front_floor_prim_path = os.path.join(root_prim_path, "front_floor")
        _create_plane(stage, front_floor_prim_path, (total_length / 2, total_width))
        _set_prim_transform(stage, front_floor_prim_path, (-total_length / 4, 0, 0.001), (0, 0, 0), (1.0, 1.0, 1.0))

        rear_floor_prim_path = os.path.join(root_prim_path, "rear_floor")
        _create_plane(stage, rear_floor_prim_path, (total_length / 2, total_width))
        _set_prim_transform(stage, rear_floor_prim_path, (total_length / 4, 0, 0.001), (0, 0, 0), (1.0, 1.0, 1.0))

        if door_frame_mats:
            bind_visual_material(front_floor_prim_path, np.random.choice(door_frame_mats), stage)
            bind_visual_material(rear_floor_prim_path, np.random.choice(door_frame_mats), stage)

    # --- Metadata ---
    metadata = {
        "doorWidth": float(door_width),
        "doorHeight": float(door_height),
        "doorHandleHeight": float(door_handle_height),
        "doorHandleWidth": float(door_handle_width),
        "doorWeight": float(door_weight),
        "doorHandleType": str(door_handle_type),
        "doorOpenLR": int(door_open_lr),
        "doorOpenIO": int(door_open_io),
        "totalWallHeight": float(total_wall_height),
        "axleLength": float(axle_length),
        "handleLength": float(handle_length),
        "hookLength": float(hook_length),
        "handleRadius": float(handle_radius),
        "spawnHook": bool(spawn_hook),
        "hingeDriveMaxForce": float(hinge_drive.GetMaxForceAttr().Get()),
        "hingeDriveStiffness": float(hinge_drive.GetStiffnessAttr().Get()),
        "handleDriveMaxForce": float(handle_drive.GetMaxForceAttr().Get()),
    }
    if cfg.add_walls:
        metadata.update({
            "front": float(front),
            "rear": float(rear),
            "leftFront": float(left_front),
            "rightFront": float(right_front),
            "leftRear": float(left_rear),
            "rightRear": float(right_rear),
        })

    _write_custom_data(stage, prim_path, metadata)

    return metadata


if __name__ == "__main__":
    main()
