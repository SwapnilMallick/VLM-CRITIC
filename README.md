# VLM-JUDGE

A toolkit for generating out-of-distribution (OOD) robot states to evaluate Vision Language Models (VLMs) on robotic manipulation tasks. It records reference trajectories of a Franka Panda arm performing a Lift task in [robosuite](https://robosuite.ai/), then systematically perturbs the end-effector position using Jacobian-based inverse kinematics to produce paired (reference, OOD) frames for downstream VLM evaluation.

---

## Overview

The workflow is two stages:

1. **Record** scripted reference trajectories as HDF5 files (multi-camera RGB frames + robot state)
2. **Perturb** a chosen timestep using IK to generate an OOD state with the same scene but a displaced end-effector

The resulting paired images can be fed to a VLM to assess whether it can detect the distributional shift.

---

## Directory Structure

```
VLM_JUDGE/
├── record_demo.py          # Stage 1: record reference trajectories
├── create_ood_state.py     # Stage 2: generate OOD perturbations via Jacobian IK
├── commands.txt            # Quick-reference command examples
├── ref_ood_states/         # Pre-generated reference states for sanity checks
│   ├── reference_demos.h5
│   ├── ood_state.npz / ood_state_ref.npz
│   └── *.png / *.mp4
└── vlm_judge_env/          # Python 3.11 virtual environment
```

Generated outputs land in the repo root:

| File | Description |
|------|-------------|
| `reference_demos_<camera>.h5` | HDF5 trajectory file |
| `demo_0_<camera>.mp4` | Recorded demo video |
| `demo_0_<camera>_frames/` | Per-frame PNGs |
| `ood_state_<camera>.npz` | OOD robot state + image |
| `ood_state_<camera>_ref.npz` | Reference state at same timestep |
| `ood_image_<camera>.png` | Re-rendered OOD frame |
| `ref_image_<camera>.png` | Original reference frame |

---

## Installation

Requires **Python 3.11** and MuJoCo.

```bash
python3.11 -m venv vlm_judge_env
source vlm_judge_env/bin/activate
pip install robosuite h5py opencv-python mujoco
```

The venv is already present in the repo — activate it directly:

```bash
source vlm_judge_env/bin/activate
```

---

## Usage

### Stage 1 — Record Reference Trajectories

```bash
python record_demo.py [--camera CAMERA] [--n_demos N] [--steps STEPS] [--seed SEED] [--video]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `agentview` | Camera viewpoint (`agentview`, `frontview`, `sideview`, `birdview`) |
| `--n_demos` | `1` | Number of demos to record |
| `--steps` | `200` | Timesteps per demo |
| `--seed` | `0` | RNG seed for deterministic cube placement |
| `--video` | off | Save MP4 video and per-frame PNGs |

**Examples:**

```bash
# Record 1 demo from the sideview camera with video output
python record_demo.py --camera sideview --n_demos 1 --steps 200 --video --seed 0

# Record 3 demos from multiple cameras
python record_demo.py --camera frontview --n_demos 3 --steps 200 --video
python record_demo.py --camera agentview --n_demos 3 --steps 200 --video
```

The scripted policy uses OSC_POSE control and executes a 4-phase Lift:
1. Hover 12 cm above the cube
2. Descend to 1 cm above the cube
3. Close the gripper
4. Lift straight up

**Output HDF5 schema:**

```
demo_0/
  ├── images            (T × 256 × 256 × 3, uint8)
  ├── robot0_eef_pos    (T × 3, float32)
  ├── robot0_joint_pos  (T × 7, float32)
  ├── robot0_joint_vel  (T × 7, float32)
  ├── robot0_gripper_qpos (T × 2, float32)
  ├── rewards           (T,)
  └── sim_states        (T × state_dim, float64)
```

---

### Stage 2 — Generate OOD States

```bash
python create_ood_state.py [--camera CAMERA] [--drift METERS] [--drift_dir DIR] [--ood_frac FRAC]
                           [--ref REF_H5] [--out OUT_NPZ]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `agentview` | Camera to render the output frame |
| `--drift` | `0.05` | End-effector displacement magnitude in meters |
| `--drift_dir` | `x` | Direction: `x`, `-x`, `y`, `-y`, `z`, `-z`, or `"a,b,c"` vector |
| `--ood_frac` | `0.3` | Fraction into the trajectory to perturb (0–1) |
| `--ref` | auto-detected | Path to reference HDF5 file |
| `--out` | auto-named | Path to output OOD NPZ file |

**Examples:**

```bash
# Default: 5 cm drift in +X at 30% through trajectory
python create_ood_state.py --camera sideview

# 8 cm drift in -X at 40% into trajectory
python create_ood_state.py --camera sideview --drift 0.08 --drift_dir "-x" --ood_frac 0.4

# Custom drift vector (normalized internally)
python create_ood_state.py --camera frontview --drift 0.1 --drift_dir "1,0.5,0"

# Explicit file paths
python create_ood_state.py --ref reference_demos_agentview.h5 --out ood_state_agentview.npz
```

**Output NPZ schema:**

```python
{
  'image':      (256, 256, 3)  # uint8 RGB
  'eef_pos':    (3,)           # end-effector position
  'joint_pos':  (7,)           # Franka joint positions
  'joint_vel':  (7,)           # Franka joint velocities
  'gripper':    (2,)           # gripper qpos
}
```

Both `ood_state_<camera>.npz` (perturbed) and `ood_state_<camera>_ref.npz` (unperturbed reference at the same timestep) are saved, along with PNG renderings for quick visual inspection.

---

## Method

### Jacobian-Based IK Perturbation

The OOD displacement is specified in Cartesian space (meters), making it interpretable regardless of the robot configuration. The algorithm:

1. Load reference joint configuration `q_ref` at the chosen timestep
2. Compute the 3×7 translational Jacobian `J` via finite differences
3. Solve for joint perturbation: `Δq = J⁺ · Δx` (pseudo-inverse least-norm solution)
4. Apply: `q_ood = clip(q_ref + Δq, joint_limits)`
5. Restore the full MuJoCo sim state (preserving cube position) and re-render

The script reports the achieved vs. requested EEF displacement so you can verify IK accuracy.

### Deterministic Scene Reconstruction

Full MuJoCo flat states (`qpos + qvel`) are recorded at each timestep. When generating OOD frames, the object positions are restored from these states before applying the joint perturbation, ensuring the cube is in the correct place.

---

## Cameras

| Camera | Notes |
|--------|-------|
| `agentview` | First-person wrist-mounted view |
| `frontview` | Front-facing; vertically flipped in storage |
| `sideview` | Side-facing; vertically flipped in storage |
| `birdview` | Top-down overhead view |

Vertical flip correction for `frontview` and `sideview` is applied automatically during recording.

---

## Dependencies

| Package | Version |
|---------|---------|
| robosuite | 1.5.2 |
| mujoco | 3.6.0 |
| numpy | 2.4.4 |
| h5py | 3.16.0 |
| opencv-python | 4.11.0.86 |
| Python | 3.11 |
