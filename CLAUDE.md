# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Pipeline: Four Scripts in Order

The full pipeline runs as four sequential stages:

### 1. Record reference trajectories
```bash
python record_demo.py --camera sideview --n_demos 1 --steps 200 --video --seed 0
```
Outputs `reference_demos_<camera>.h5` (HDF5) plus optional MP4 and per-frame PNGs.

### 2. Generate OOD state
```bash
python create_ood_state.py --camera sideview --drift 0.05 --drift_dir x --ood_frac 0.3
```
Reads the HDF5, displaces the EEF via Jacobian IK, outputs `ood_state_<camera>.npz` (perturbed) and `ood_state_<camera>_ref.npz` (reference at same timestep), plus PNG renderings.

Drift direction accepts axis presets (`x`, `-x`, `y`, `-y`, `z`, `-z`) or a comma-separated vector (`"1,0.5,0"`). Use `=` syntax for negative presets: `--drift_dir=-x`.

### 3. Generate candidate actions
```bash
python generate_candidate_actions.py --ood ood_state_sideview.npz --n 16 --seed 0
```
Outputs `candidate_actions_<camera>.npz` with `n` candidate OSC_POSE actions. Each action is a unit-vector translation (`[dx, dy, dz]` normalized from `N(0,1)³`), zero rotation, and gripper fixed to `-1.0` (open).

### 4. Render candidate actions
```bash
python render_candidate_actions.py --camera sideview --n 16
```
Restores the OOD scene and steps each candidate action `--steps` times (default: 10), saving `action_XXXX.png` + `action_XXXX_seg.npy` + `action_XXXX_seg.png` into `candidate_images_<camera>/`. Also saves `ood_state.png` and `ood_state_seg.npy` as the baseline. The `ood_state_<camera>.npz` **must** contain a `sim_state` key — re-run `create_ood_state.py` if using older files.

### 5. VLM evaluation (optional)
Three backends for detecting the red cube and robot gripper in candidate images:

```bash
# Grounded-SAM (Grounding DINO + SAM via HuggingFace)
python grounded_sam_segment.py --img_dir candidate_images_frontview

# Qwen-VL GGUF (requires llama-cpp-python + downloaded model files)
python qwen_vl_detect.py --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
    --img_dir candidate_images_frontview

# InternVL2-4B (downloaded automatically from HuggingFace, ~8 GB)
python intern_vl_detect.py --img_dir candidate_images_frontview
```

### 6. Rank candidate actions
```bash
python compute_centroid_distance.py --dir candidate_images_frontview
```
Uses robosuite's ground-truth instance segmentation masks (`*_seg.npy`) to compute the cube-to-gripper centroid L2 distance for each candidate. Actions are ranked by delta vs. the OOD baseline — a negative delta means the gripper moved closer to the cube (better). Results saved to `<dir>/centroid_distances.txt`.

## Architecture

**Data flow:** `record_demo.py` → `.h5` → `create_ood_state.py` → `.npz` pair → `generate_candidate_actions.py` → `.npz` → `render_candidate_actions.py` → `.png` + `_seg.npy` images → `compute_centroid_distance.py` → ranked results.

**`record_demo.py`** runs robosuite's `Lift` task with a 4-phase scripted OSC_POSE policy (hover → descend → close gripper → lift). Records per-timestep MuJoCo flat states (`qpos + qvel`) alongside RGB frames and robot state. These `sim_states` are what allow later scripts to restore the exact scene including cube position.

**`create_ood_state.py`** implements Jacobian-based IK: loads `q_ref` at `ood_frac * T`, computes a 3×7 translational Jacobian via finite differences, pseudo-inverts it (`J⁺ · Δx`), adds `Δq` to joints, clamps to Panda limits, re-renders. Both the perturbed and unperturbed states at that timestep are saved for VLM comparison.

**`generate_candidate_actions.py`** samples `n` unit-vector translations from `N(0,1)³` (normalized), with rotation fixed to zero and gripper fixed to `-1.0` (open).

**`render_candidate_actions.py`** restores the OOD scene via `env.sim.set_state_from_flattened(ood_sim_state)` before every action to ensure each candidate starts from the same state. Uses `env._get_observations(force_update=True)` to bypass the cached observation after manual state restoration. Saves RGB frames and instance segmentation masks (`camera_segmentations="instance"`).

**`compute_centroid_distance.py`** loads the `*_seg.npy` masks, identifies the red-cube label (yellow in colorized PNG) and gripper label (pink) by their rendered color, computes pixel centroids, and ranks candidates by L2 distance delta vs. the OOD baseline.

**VLM scripts** (`grounded_sam_segment.py`, `qwen_vl_detect.py`, `intern_vl_detect.py`) operate independently on the rendered PNGs in `candidate_images_<camera>/`. They skip `_seg.png`, `_vis.png`, and `_raw.txt` files automatically.

**Camera handling:** `frontview` and `sideview` images are vertically flipped (their raw output is inverted). The constant `CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}` is defined in each script that renders images. All scripts default to `--camera sideview`.

**File naming convention:** All output files are named `<type>_<camera>.<ext>` (e.g., `ood_state_agentview.npz`, `candidate_actions_frontview.npz`). Scripts auto-derive filenames from `--camera` when explicit paths are not given.

**Action space:** OSC_POSE, 7-dimensional `[dx, dy, dz, dax, day, daz, gripper]` where gripper is `-1` (open) or `+1` (close).
