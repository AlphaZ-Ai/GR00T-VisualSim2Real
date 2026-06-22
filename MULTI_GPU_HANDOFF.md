# Multi-GPU Student Distillation — Handoff / What to change & verify

Goal: run the vision **student distillation** (`door_open_homie_dagger-lstm`) data-parallel across N GPUs
to cut wall-clock (target ~24h on 8×L40S). The teacher (state-only) already scales fine; the student
does **not** on this box because of a **camera-rendering** limitation, not the training code.

---

## TL;DR root cause (verify this is gone in the new env first)

This Isaac Sim build renders via **`usdrt` (USD Runtime)**, which is hard-limited to **`cuda:0`**:

```
[Error] [usdrt.scenegraph.plugin] C++ UsdStage::SelectPrims: GPU 1 requested.
        GPUs other than cuda:0 are not currently supported
```

It's invoked by several subsystems (Fabric scene delegate, PhysX↔render GPU interop, camera prim
selection), so non-zero ranks can't render. Two reconciliation attempts both failed here:

- **`CUDA_VISIBLE_DEVICES` isolation** (each rank's GPU → `cuda:0`): `usdrt` OK, but the **Vulkan**
  renderer enumerates GPUs *physically* and ignores `CUDA_VISIBLE_DEVICES`, so it lands on a different
  physical GPU than CUDA → deadlock in `initialize_physics` (confirmed via `py-spy`).
- **`--distributed` + `app.useFabricSceneDelegate=false`**: `usdrt` is still invoked elsewhere and
  still rejects GPU 1.

Non-camera multi-GPU (the teacher) is unaffected — only the student renders cameras.

---

## Step 1 — Confirm the new environment can render on >1 GPU (do this BEFORE porting)

Use a newer IsaacLab/Isaac Sim where multi-GPU tiled-camera rendering is supported (verify on the
target image; this `isaacsim5.1` install is the one that fails). Fastest platform smoke test — run
IsaacLab's **own** distributed camera example and confirm it trains with no `usdrt` cuda:0 error:

```bash
# IsaacLab repo, NOT this one — just to validate the platform:
python -m torch.distributed.run --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py \
    --task <some Camera/RGB task> --headless --distributed --enable_cameras
```

If that errors with `GPUs other than cuda:0 are not currently supported`, the platform is still bad —
do not bother porting. If it trains on both GPUs, proceed.

Also check the render kit (`gr00t/rl/apps/phc.isaaclab.python.headless.rendering.kit`):
- `renderer.multiGpu.maxGpuCount = 1` is fine (1 GPU **per rank**, that's what we want).
- Confirm the new `usdrt`/Fabric version supports per-rank GPU (the whole point of Step 1).

## Step 2 — Launch command for THIS repo (once Step 1 passes)

The proper path is IsaacLab `--distributed` (each rank on `cuda:rank`, all GPUs visible). The code is
already reverted to this — do **not** re-add the `CUDA_VISIBLE_DEVICES` isolation hack.

```bash
accelerate launch --num_processes 8 --num_machines 1 --multi_gpu --mixed_precision no \
  gr00t/rl/train_agent_trl.py +exp=wbmanip/door_open_homie_dagger-lstm \
  ++project_name=g1_open_door_homie_student \
  ++num_envs=128 \                      # PER-GPU; pack VRAM (64 used ~24/46GB → 128–256 is safe)
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor.backbone.vision_module.module_config_dict.layer_config.trainable=True \
  ++algo.config.obj_pred_loss_coef=1.0 ++algo.config.actor.running_mean_std=True \
  ++algo.config.num_learning_epochs=1 ++algo.config.num_mini_batches=64 \
  ++algo.config.teacher_rollout_ratio=0.3
```

`num_envs` is **per process**, so 8×128 = 1024 total envs. Bump per-GPU envs until ~40GB/46GB used.

## Step 3 — Verification checklist (first ~3 min of the run)

- [ ] **No** `usdrt ... GPUs other than cuda:0` error in the log.
- [ ] All N GPUs show compute load in `nvidia-smi` (not just GPU 0).
- [ ] Teacher loads at **190-dim** (`algo_obs_dim_dict: teacher_obs: 190`) — see fixes below.
- [ ] Reaches `Learning iteration` with `dagger_bc_loss` decreasing.
- [ ] wandb project is **`g1_open_door_homie_student`**, a **new** run (no "resume wandb").

---

## Fixes already applied in this repo (env-independent bugs — keep them)

These were required just to get the student to *run* (the cameras-on path exposed them; the teacher
never hit them). They are already committed in the working tree:

| File | Fix |
|---|---|
| `gr00t/rl/train_agent_trl.py` | Rendering-kit dest path (`Path(isaaclab.__file__).resolve().parent/"apps"`, was 4× `.parent`) |
| `gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml` | `camera_attached_link: torso_link/d435_link` (d435 is nested under torso); `teacher_actor_path: models/teacher_push_right_step008900.pt`; `randomize_dome_light: True` |
| `gr00t/rl/simulator/isaacsim/isaacsim.py` | `find_bodies` uses first path segment (`torso_link`); dome light → local HDRs + `dynamic_randomize_texture=False` |
| `gr00t/rl/config/obs/wbmanip/door_open_homie_dagger.yaml` | `hand_force: ${robot.num_hand_body_links}*3` (Dex1=12, was hardcoded 48 for Dex3 → teacher 226≠190) |
| `gr00t/rl/envs/door/door_open_homie.py` | `dof_*_non_finger` uses `self.non_finger_dof_idx` (was hardcoded `[:, :-14]` for a 14-dof hand) |
| `gr00t/rl/scripts/generate_door_assets.py` | launches via `AppLauncher` (was bare `SimulationApp` → `No module named isaaclab.sim`) |

## Environment gotchas seen here (likely fine on a connected vendor image)

- **S3 unreachable** here: NVIDIA Omniverse asset bucket (`omniverse-content-production.s3...`) 404s.
  - Dome-light HDR skies were redirected to **local** files in `data/dome_light_textures/` (10 HDRs).
    On a connected env you may restore the original S3 sky folders in `isaacsim.py` if preferred.
  - The Dex3 robot's MDL materials are also on that bucket — the G1 was recolored with offline
    OmniPBR diffuse equivalents (silver body / charcoal joints / dark Dex1 grippers).
- **Dynamic dome-light randomization** needs `omni.kit.scripting`'s `OmniScriptingAPI`, which was
  unregistered here → we use **static** per-spawn randomization + the `randomize_dome_light` EventTerm
  (intensity/yaw at reset). If the new kit registers OmniScriptingAPI you can re-enable dynamic.
- **Pre-generated right-lever doors** (`data/door_assets/right_out_lever/`, 1000 doors) — opt-in via
  `DOOR_ASSET_DIR` in `gr00t/rl/data/tasks/door/scenario_cfg/isaacsim.py` (default = procedural
  `spawn_door`). Generator fix **DONE + VALIDATED** (trained 19 iterations, door contact penalties
  computing): the doors are now built at a **top-level `/Door` prim** with it set as the stage
  **default prim** so they're referenceable via `UsdFileCfg` (a nested `/World/Door` prim can't be a
  default prim, so the reference brought in no rigid bodies). Also fixed `domelight.py` to import
  `DomeLightRandomization` lazily (its top-level `omni.kit.scripting` import otherwise breaks
  dome-light spawning where that extension is absent).
  - **Remaining caveat (textures):** door materials are authored at `/World/Looks` (outside `/Door`),
    so referenced doors render **untextured**. Physics/contact (the "bake physics" ask) is solved;
    for full texture parity move the materials under `/Door` before export (e.g. `Sdf.BatchNamespaceEdit`
    `/World/Looks` → `/Door/Looks`) — or just use the procedural default, which textures correctly.
    Materials come from the same unreachable S3 bucket here, so this only matters in a connected env.

## Scale / ETA

- Paper: ~30×L40, 24h. On 8×L40S, matching total env-throughput (pack envs across 8 GPUs) targets
  roughly the same wall-clock; expect a converged student (~80–85% success, vs ~50–70% pre-GRPO) and
  then the separate **GRPO** stage (not in this repo — see notes; it'd need to be implemented).

## Teacher / artifacts

- Teacher checkpoint: `models/teacher_push_right_step008900.pt` (right-push, eval 74.6% success).
- Student exp: `gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`.
