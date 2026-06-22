# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils import configclass

from pxr import Usd, UsdLux
import numpy as np

from gr00t.rl.isaac_utils.playground.utils.nucleus_utils import list_files_with_extension
# NOTE: DomeLightRandomization is imported lazily inside spawn_random_dome_light (only when
# dynamic_randomize_texture is enabled) because it imports omni.kit.scripting, which is not always
# available (e.g. the OmniScriptingAPI is unregistered in some builds). Importing it at module level
# breaks dome-light spawning entirely even when dynamic randomization is off.


@clone
def spawn_random_dome_light(
    prim_path: str,
    cfg: "RandomDomeLightCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None
):
    if not cfg.texture_file_folders:
        raise ValueError("texture_file_folders is not set")
    texture_file_list = []
    for folder in cfg.texture_file_folders:
        files = list_files_with_extension(folder, ".hdr", recursive=True)
        if files:
            texture_file_list.extend(files)
    print(f"Found {len(texture_file_list)} texture files from {len(cfg.texture_file_folders)} folders")
    prim: Usd.Prim = prim_utils.create_prim(prim_path, "DomeLight", translation=translation, orientation=orientation)
    dome_light_prim: UsdLux.DomeLight = UsdLux.DomeLight.Define(prim.GetStage(), prim_path)
    dome_light_prim.CreateTextureFileAttr().Set(np.random.choice(texture_file_list))
    dome_light_prim.CreateIntensityAttr().Set(np.random.uniform(cfg.intensity_range[0], cfg.intensity_range[1]))

    if cfg.dynamic_randomize_texture:
        from gr00t.rl.isaac_utils.playground.scripts.domelight_randomization import (
            DomeLightRandomization,
        )

        DomeLightRandomization.add_to_prim(prim.GetStage(), prim_path, cfg.dynamic_randomize_texture_interval, texture_file_list, cfg.intensity_range)

    return prim


@configclass
class RandomDomeLightCfg(sim_utils.DomeLightCfg):
    texture_file_folders: list[str] = []
    intensity_range: tuple[float, float] = (1000.0, 5000.0)
    func: Callable = spawn_random_dome_light
    dynamic_randomize_texture: bool = False
    dynamic_randomize_texture_interval: float = 1.0
