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
Outputs `candidate_actions_<camera>.npz` with `n` uniformly-sampled OSC_POSE actions.

**Known issue (see `comments.txt`):** gripper dimension is currently sampled randomly. The gripper state should be fixed to match the OOD state's gripper rather than sampled freely.

### 4. Render candidate actions
```bash
python render_candidate_actions.py --camera sideview --n 16
```
Restores the OOD scene and steps each candidate action once, saving `action_XXXX.png` images plus `ood_state.png` into `candidate_images_<camera>/`. The `ood_state_<camera>.npz` **must** contain a `sim_state` key — re-run `create_ood_state.py` if using older files.

## Architecture

**Data flow:** `record_demo.py` → `.h5` → `create_ood_state.py` → `.npz` pair → `generate_candidate_actions.py` → `.npz` → `render_candidate_actions.py` → `.png` images.

**`record_demo.py`** runs robosuite's `Lift` task with a 4-phase scripted OSC_POSE policy (hover → descend → close gripper → lift). Records per-timestep MuJoCo flat states (`qpos + qvel`) alongside RGB frames and robot state. These `sim_states` are what allow later scripts to restore the exact scene including cube position.

**`create_ood_state.py`** implements Jacobian-based IK: loads `q_ref` at `ood_frac * T`, computes a 3×7 translational Jacobian via finite differences, pseudo-inverts it (`J⁺ · Δx`), adds `Δq` to joints, clamps to Panda limits, re-renders. Both the perturbed and unperturbed states at that timestep are saved for VLM comparison.

**`render_candidate_actions.py`** restores the OOD scene via `env.sim.set_state_from_flattened(ood_sim_state)` before every action to ensure each candidate starts from the same state. Uses `env._get_observations(force_update=True)` to bypass the cached observation after manual state restoration.

**Camera handling:** `frontview` and `sideview` images are vertically flipped (their raw output is inverted). The constant `CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}` is defined in each script that renders images. All scripts default to `--camera sideview`.

**File naming convention:** All output files are named `<type>_<camera>.<ext>` (e.g., `ood_state_agentview.npz`, `candidate_actions_frontview.npz`). Scripts auto-derive filenames from `--camera` when explicit paths are not given.

**Action space:** OSC_POSE, 7-dimensional `[dx, dy, dz, dax, day, daz, gripper]` where gripper is `-1` (open) or `+1` (close).
