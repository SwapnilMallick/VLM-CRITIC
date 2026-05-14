# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

VLM-JUDGE evaluates Vision Language Models on robotic manipulation tasks. The pipeline records a scripted Franka Panda "Lift" trajectory in robosuite, perturbs the robot into an out-of-distribution (OOD) state via Jacobian IK, generates candidate corrective actions, renders them, optionally runs VLM detection, and ranks actions by cube-to-gripper centroid distance.

## Environment Setup

Python 3.11 + MuJoCo. A venv is already present — always activate it first:

```bash
source vlm_judge_env/bin/activate
```

To recreate from scratch:
```bash
python3.11 -m venv vlm_judge_env
source vlm_judge_env/bin/activate
pip install robosuite h5py opencv-python mujoco
# Pinned versions: robosuite==1.5.2, mujoco==3.6.0, numpy==2.4.4, h5py==3.16.0, opencv-python==4.11.0.86
```

Optional VLM backends: `transformers torch torchvision sentencepiece` (Grounded-SAM, InternVL2); `llama-cpp-python` with Metal support (Qwen-VL).

## Pipeline: Six Stages in Order

### 1. Record reference trajectories
```bash
python record_demo.py --camera sideview --n_demos 1 --steps 200 --video --seed 0
```
Outputs `reference_demos_<camera>.h5` (HDF5) plus optional MP4 and per-frame PNGs.

**HDF5 schema** (`demo_0/` group):
- `images` — (T × 256 × 256 × 3, uint8, gzip-4)
- `sim_states` — (T × state_dim, float64) — full MuJoCo qpos+qvel; essential for exact scene restoration
- `robot0_eef_pos`, `robot0_joint_pos`, `robot0_joint_vel`, `robot0_gripper_qpos` — per-timestep robot state

**Scripted 4-phase Lift policy** (OSC_POSE, 7-dim `[dx,dy,dz,dax,day,daz,gripper]`):
- t < 60: hover 12 cm above cube, gripper open
- t < 120: descend to 1 cm above cube, gripper open
- t < 160: hold while closing gripper
- t ≥ 160: lift straight up, gripper closed

### 2. Generate OOD state
```bash
python create_ood_state.py --camera sideview --drift 0.05 --drift_dir x --ood_frac 0.3
```
Reads the HDF5, displaces the EEF via Jacobian IK, outputs `ood_state_<camera>.npz` (perturbed) and `ood_state_<camera>_ref.npz` (reference at same timestep), plus PNG renderings.

Drift direction accepts axis presets (`x`, `-x`, `y`, `-y`, `z`, `-z`) or a comma-separated vector (`"1,0.5,0"`). Use `=` syntax for negative presets: `--drift_dir=-x`.

**Output NPZ keys:** `image`, `eef_pos`, `joint_pos`, `joint_vel`, `gripper`, `sim_state` (the `sim_state` key is required by Stage 4 — re-run this script if using older files that lack it).

**IK algorithm:** load `q_ref` at `ood_frac * T`, compute 3×7 translational Jacobian via finite differences (eps=1e-5), pseudo-invert (`J⁺ · Δx`), clamp to Panda joint limits. Cartesian-space perturbation is chosen over joint-space noise because the displacement magnitude is interpretable in meters. The script reports requested vs. achieved EEF displacement for verification.

### 3. Generate candidate actions
```bash
python generate_candidate_actions.py --camera sideview --n 16 --seed 0
```
Outputs `candidate_actions_<camera>.npz`.

Each action is a unit-vector translation (`[dx,dy,dz]` normalized from `N(0,1)³`) with zero rotation and gripper fixed to `-1.0` (open). Unit vectors give uniform random directions on the sphere so every candidate explores maximum displacement. Gripper is fixed to open — not randomly sampled — because a random gripper state would conflate grasping vs. positioning quality. See `comments.txt` for rationale.

**Output NPZ keys:** `actions` (N×7), `ood_eef_pos` (3,), `ood_joint_pos` (7,), `gripper_action` (-1.0), `seed`.

### 4. Render candidate actions
```bash
python render_candidate_actions.py --camera sideview --steps 10
```
Restores the OOD scene before each candidate and steps the action `--steps` times (default: 10), saving `action_XXXX.png` + `action_XXXX_seg.npy` + `action_XXXX_seg.png` into `candidate_images_<camera>/`. Also saves `ood_state.png` and `ood_state_seg.npy` as the baseline.

**State restoration pattern** (used before every action for a clean comparison baseline):
1. `env.reset()` — reinitializes bookkeeping
2. `env.sim.set_state_from_flattened(ood_sim_state)` — restores exact cube+robot config
3. `env.sim.forward()` — updates derived quantities (site positions)
4. `env._get_observations(force_update=True)` — bypasses observation cache after manual state restore

Instance segmentation is enabled via `camera_segmentations="instance"` in `suite.make()`.

### 5. VLM evaluation (optional)
Three backends detect the red cube and robot gripper in candidate images. **These are not yet wired into the ranking pipeline** — Stage 6 uses ground-truth robosuite masks regardless.

```bash
# Grounded-SAM (Grounding DINO + SAM via HuggingFace)
python grounded_sam_segment.py --img_dir candidate_images_sideview

# Qwen-VL GGUF (requires llama-cpp-python + downloaded model files)
python qwen_vl_detect.py --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
    --img_dir candidate_images_sideview

# InternVL2-4B (downloaded automatically from HuggingFace, ~8 GB)
python intern_vl_detect.py --img_dir candidate_images_sideview
```

All VLM scripts skip `_seg.png`, `_vis.png`, and `_raw.txt` files automatically and output per-image `_<model>_vis.png` overlays plus a `<model>_detections.json` summary.

**Non-obvious GPU constraint:** Grounded-SAM runs SAM on CPU even when Grounding DINO is on GPU — MPS does not support the float64 ops that SAM requires.

### 6. Rank candidate actions
```bash
python compute_centroid_distance.py --dir candidate_images_sideview
```
Uses robosuite's ground-truth instance segmentation masks (`*_seg.npy`) to compute the cube-to-gripper centroid L2 distance. Actions ranked by delta vs. OOD baseline — **negative delta = gripper moved closer to cube (better)**. Results saved to `<dir>/centroid_distances.txt`.

**Label identification heuristics** (applied to the colorized `*_seg.png`):
- Red cube → **yellow** pixels: `r>120, g>120, b<100, |r-g|<80`
- Gripper → **pink** pixels: `r>120, b>120, g<100`
- Background: largest ID by pixel count

## Architecture

**Data flow:**
```
record_demo.py → .h5 → create_ood_state.py → .npz pair
  → generate_candidate_actions.py → .npz
  → render_candidate_actions.py → .png + _seg.npy
  → [VLM scripts (optional)]
  → compute_centroid_distance.py → ranked results
```

**Camera handling:** `frontview` and `sideview` raw output is vertically inverted by the camera mount. Each script that renders images defines `CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}` and applies `frame[::-1]` when needed. All scripts default to `--camera sideview`.

**File naming convention:** All outputs are `<type>_<camera>.<ext>` (e.g., `ood_state_sideview.npz`, `candidate_actions_frontview.npz`). Scripts auto-derive filenames from `--camera` when explicit paths are not given.

**Action space:** OSC_POSE, 7-dimensional `[dx, dy, dz, dax, day, daz, gripper]` where gripper is `-1` (open) or `+1` (close).

## Utilities

**`inspect_segmentation.py`** — debug tool to identify which segmentation IDs correspond to cube, gripper, and background. Prints a table of all IDs with pixel counts and mean RGB colors.

```bash
python inspect_segmentation.py --seg candidate_images_sideview/ood_state_seg.npy
```

**`render_directional_actions.py`** — alternative to Stage 4 that uses Jacobian IK (not policy rollout) to move the EEF exactly `--dist` metres in each of the 8 compass directions (N, NE, E, SE, S, SW, W, NW) and renders the result. Each direction gets its own subfolder inside `directional_actions_<camera>/` containing `image.png`, `image_seg.npy`, `image_seg.png`, and `state.npz` (which includes a `direction` and `dist_m` key alongside the usual state fields). Also saves the OOD baseline as `ood_state.png`.

```bash
python render_directional_actions.py --camera agentview --dist 0.025
python render_directional_actions.py --ood ood_state_sideview.npz --dist 0.05
```

Compass convention (viewed from above): North = +y, East = +x. All displacements are purely horizontal (z=0). Outputs can be passed directly to `compute_centroid_distance.py` or the VLM scripts by pointing `--dir` / `--img_dir` at the subfolder for a single direction, or the parent `directional_actions_<camera>/` to rank across all directions.

> **Note:** After `env.reset()`, the robot sim object is re-created internally, so `site_id` and `robot` must be re-fetched from `env.robots[0]` after each direction's restore call — the script does this inside the direction loop.
