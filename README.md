# VLM-JUDGE

A toolkit for evaluating Vision Language Models (VLMs) on robotic manipulation tasks. The pipeline records reference trajectories of a Franka Panda arm performing a Lift task in [robosuite](https://robosuite.ai/), perturbs the end-effector position via Jacobian-based IK to produce an OOD state, generates and renders candidate corrective actions, and then uses VLMs to identify which candidate brings the gripper closest to the cube.

---

## Overview

The full pipeline has four core stages, plus optional VLM evaluation tools:

```
record_demo.py  →  create_ood_state.py  →  generate_candidate_actions.py  →  render_candidate_actions.py
                                                                                          ↓
                                         [grounded_sam_segment.py | qwen_vl_detect.py | intern_vl_detect.py]
                                                                                          ↓
                                                                         compute_centroid_distance.py
```

1. **Record** scripted reference trajectories (HDF5 + optional video)
2. **Perturb** a chosen timestep via IK to create a paired (reference, OOD) frame
3. **Generate** candidate corrective actions as random unit-vector translations
4. **Render** each candidate from the OOD state and save RGB + segmentation masks
5. **Evaluate** rendered images with a VLM (Grounded-SAM, Qwen-VL, or InternVL2)
6. **Rank** candidates by cube-to-gripper centroid distance from robosuite's instance segmentation

---

## Directory Structure

```
VLM_JUDGE/
├── record_demo.py                  # Stage 1: record reference trajectories
├── create_ood_state.py             # Stage 2: generate OOD perturbations via Jacobian IK
├── generate_candidate_actions.py   # Stage 3: sample candidate corrective actions
├── render_candidate_actions.py     # Stage 4: render each candidate from the OOD state
├── grounded_sam_segment.py         # VLM eval: Grounding DINO + SAM segmentation
├── qwen_vl_detect.py               # VLM eval: Qwen-VL GGUF detection
├── intern_vl_detect.py             # VLM eval: InternVL2 detection
├── compute_centroid_distance.py    # Rank actions by cube-gripper centroid L2 distance
├── comments.txt                    # Design notes and known issues
├── ref_ood_states/                 # Pre-generated states for sanity checks
└── vlm_judge_env/                  # Python 3.11 virtual environment
```

Generated outputs land in the repo root:

| File | Description |
|------|-------------|
| `reference_demos_<camera>.h5` | HDF5 trajectory file |
| `demo_0_<camera>.mp4` | Recorded demo video |
| `demo_0_<camera>_frames/` | Per-frame PNGs |
| `ood_state_<camera>.npz` | OOD robot state + image + sim state |
| `ood_state_<camera>_ref.npz` | Reference (unperturbed) state at same timestep |
| `ood_image_<camera>.png` | Re-rendered OOD frame |
| `ref_image_<camera>.png` | Original reference frame |
| `candidate_actions_<camera>.npz` | Sampled candidate corrective actions |
| `candidate_images_<camera>/` | RGB frames + segmentation masks per candidate |

---

## Installation

Requires **Python 3.11** and MuJoCo. A venv is already present — activate it directly:

```bash
source vlm_judge_env/bin/activate
```

To recreate from scratch:

```bash
python3.11 -m venv vlm_judge_env
source vlm_judge_env/bin/activate
pip install robosuite h5py opencv-python mujoco
```

Pinned versions: `robosuite==1.5.2`, `mujoco==3.6.0`, `numpy==2.4.4`, `h5py==3.16.0`, `opencv-python==4.11.0.86`

For VLM evaluation, additional packages are needed (see each script's docstring).

---

## Core Pipeline

### Stage 1 — Record Reference Trajectories

```bash
python record_demo.py [--camera CAMERA] [--n_demos N] [--steps STEPS] [--seed SEED] [--video]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `sideview` | Camera viewpoint (`agentview`, `frontview`, `sideview`, `birdview`) |
| `--n_demos` | `1` | Number of demos to record |
| `--steps` | `200` | Timesteps per demo |
| `--seed` | `0` | RNG seed for deterministic cube placement |
| `--video` | off | Save MP4 video and per-frame PNGs |

```bash
python record_demo.py --camera sideview --n_demos 1 --steps 200 --video --seed 0
```

The scripted policy uses OSC_POSE control and executes a 4-phase Lift:
1. Hover 12 cm above the cube
2. Descend to 1 cm above the cube
3. Close the gripper
4. Lift straight up

**Output HDF5 schema:**

```
demo_0/
  ├── images              (T × 256 × 256 × 3, uint8)
  ├── robot0_eef_pos      (T × 3, float32)
  ├── robot0_joint_pos    (T × 7, float32)
  ├── robot0_joint_vel    (T × 7, float32)
  ├── robot0_gripper_qpos (T × 2, float32)
  ├── rewards             (T,)
  └── sim_states          (T × state_dim, float64)
```

---

### Stage 2 — Generate OOD States

```bash
python create_ood_state.py [--camera CAMERA] [--drift METERS] [--drift_dir DIR] [--ood_frac FRAC]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `sideview` | Camera to render the output frame |
| `--drift` | `0.05` | End-effector displacement magnitude in meters |
| `--drift_dir` | `x` | Direction: `x`, `-x`, `y`, `-y`, `z`, `-z`, or `"a,b,c"` vector |
| `--ood_frac` | `0.3` | Fraction into the trajectory to perturb (0–1) |
| `--ref` | auto-detected | Path to reference HDF5 file |
| `--out` | auto-named | Path to output OOD NPZ file |

```bash
# Default: 5 cm drift in +X at 30% through trajectory
python create_ood_state.py --camera sideview

# 8 cm drift in -X at 40% into trajectory
python create_ood_state.py --camera sideview --drift 0.08 --drift_dir=-x --ood_frac 0.4

# Custom drift vector (normalized internally)
python create_ood_state.py --camera frontview --drift 0.1 --drift_dir "1,0.5,0"
```

Outputs both `ood_state_<camera>.npz` (perturbed) and `ood_state_<camera>_ref.npz` (unperturbed reference at the same timestep), plus PNG renderings for visual inspection.

**Output NPZ schema:**

```python
{
  'image':     (256, 256, 3)       # uint8 RGB
  'eef_pos':   (3,)                # end-effector position
  'joint_pos': (7,)                # Franka joint positions
  'joint_vel': (7,)                # Franka joint velocities
  'gripper':   (2,)                # gripper qpos
  'sim_state': (state_dim,)        # full MuJoCo flat state (qpos + qvel)
}
```

---

### Stage 3 — Generate Candidate Actions

```bash
python generate_candidate_actions.py [--ood OOD_NPZ] [--camera CAMERA] [--n N] [--seed SEED] [--out OUT]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ood` | `ood_state_<camera>.npz` | Path to OOD state NPZ |
| `--camera` | `sideview` | Camera name for auto-resolving file paths |
| `--n` | `16` | Number of candidate actions to generate |
| `--seed` | `0` | RNG seed for reproducibility |
| `--out` | `candidate_actions_<camera>.npz` | Output NPZ path |

```bash
python generate_candidate_actions.py --ood ood_state_sideview.npz --n 16 --seed 0
python generate_candidate_actions.py --ood ood_state_frontview.npz --n 32 --seed 7
```

Each candidate action is a 7-dim OSC_POSE vector `[dx, dy, dz, dax, day, daz, gripper]`:
- **Translation** `[dx, dy, dz]`: random unit vectors (uniform random direction, magnitude 1.0), giving maximum displacement in distinct directions
- **Rotation** `[dax, day, daz]`: fixed to zero
- **Gripper**: fixed to `-1.0` (open)

---

### Stage 4 — Render Candidate Actions

```bash
python render_candidate_actions.py [--camera CAMERA] [--ood OOD_NPZ] [--actions ACTIONS_NPZ]
                                   [--steps STEPS] [--out_dir OUT_DIR]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `sideview` | Camera to render (must match Stage 2) |
| `--ood` | `ood_state_<camera>.npz` | OOD state (must contain `sim_state` key) |
| `--actions` | `candidate_actions_<camera>.npz` | Candidate actions NPZ |
| `--steps` | `10` | Times each action is applied before rendering |
| `--out_dir` | `candidate_images_<camera>/` | Output directory |

```bash
python render_candidate_actions.py --camera sideview --n 16
python render_candidate_actions.py --camera frontview --steps 5
```

For each candidate, the script restores the exact OOD scene, applies the action `--steps` times, and saves:
- `action_XXXX.png` — RGB frame
- `action_XXXX_seg.npy` — instance segmentation mask (raw label IDs)
- `action_XXXX_seg.png` — colorized segmentation mask

Also saves `ood_state.png` and `ood_state_seg.npy` as the baseline reference.

> **Note:** `ood_state_<camera>.npz` must contain a `sim_state` key. Re-run Stage 2 if using older files.

---

## VLM Evaluation

Three VLM backends are available for detecting the red cube and robot gripper in candidate images. Results can be compared against the ground-truth robosuite segmentation using `compute_centroid_distance.py`.

### Grounded-SAM (Grounding DINO + SAM)

Uses HuggingFace `transformers`. Models are downloaded automatically on first run (~1–2 GB).

```bash
pip install torch transformers pillow matplotlib

python grounded_sam_segment.py [--img_dir DIR] [--box_threshold T] [--text_threshold T] [--device DEVICE]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_dir` | `candidate_images_agentview` | Directory with candidate PNGs |
| `--box_threshold` | `0.30` | Grounding DINO box confidence threshold |
| `--text_threshold` | `0.25` | Grounding DINO text similarity threshold |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |

```bash
python grounded_sam_segment.py --img_dir candidate_images_frontview
```

Outputs per image: `<name>_gsam_vis.png` (overlay) and `<name>_gsam_masks.npz` (masks, boxes, scores, labels).

---

### Qwen-VL (GGUF, via llama-cpp-python)

Runs a quantised Qwen2.5-VL or Qwen3-VL model locally.

**Setup:**
```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# Download model + vision projector from HuggingFace:
#   bartowski/Qwen2.5-VL-7B-Instruct-GGUF   (lighter)
#   bartowski/Qwen3-VL-30B-A3B-Instruct-GGUF (30B MoE)
# You need two files: *-Q4_K_M.gguf and mmproj-*-f16.gguf
```

```bash
python qwen_vl_detect.py \
    --model  /path/to/model-Q4_K_M.gguf \
    --mmproj /path/to/mmproj-f16.gguf \
    --img_dir candidate_images_frontview
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to quantised model GGUF |
| `--mmproj` | required | Path to vision projector GGUF |
| `--img_dir` | `candidate_images_agentview` | Directory with candidate PNGs |
| `--max_tokens` | `256` | Max tokens to generate per image |
| `--n_gpu_layers` | `-1` | Layers to offload to GPU; `-1` = all |
| `--n_ctx` | `4096` | Context window size |

Outputs per image: `<name>_qwen_vis.png` and `<name>_qwen_raw.txt`. Summary saved to `<img_dir>/qwen_detections.json`.

---

### InternVL2 (via HuggingFace)

Uses `OpenGVLab/InternVL2-4B` (downloaded automatically, ~8 GB).

**Setup:**
```bash
pip install transformers torch torchvision pillow sentencepiece
```

```bash
python intern_vl_detect.py [--img_dir DIR] [--model_name MODEL] [--max_new_tokens N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_dir` | `candidate_images_agentview` | Directory with candidate PNGs |
| `--model_name` | `OpenGVLab/InternVL2-4B` | HuggingFace model ID |
| `--max_new_tokens` | `256` | Max tokens to generate per image |

```bash
python intern_vl_detect.py --img_dir candidate_images_frontview
python intern_vl_detect.py --model_name OpenGVLab/InternVL2-2B --img_dir candidate_images_sideview
```

Outputs per image: `<name>_internvl_vis.png` and `<name>_internvl_raw.txt`. Summary saved to `<img_dir>/internvl_detections.json`.

---

## Ranking Candidates

`compute_centroid_distance.py` uses robosuite's ground-truth instance segmentation masks (the `*_seg.npy` files from Stage 4) to rank candidate actions. It computes the L2 distance between the gripper and cube centroids in each post-action frame and compares to the OOD baseline.

```bash
python compute_centroid_distance.py [--dir DIR] [--out OUT]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dir` | `candidate_images_frontview` | Directory with `*_seg.npy` and `*_seg.png` files |
| `--out` | `<dir>/centroid_distances.txt` | Output text file |

```bash
python compute_centroid_distance.py --dir candidate_images_frontview
```

**Ranking logic:** a negative delta (action L2 − OOD L2) means the candidate moved the gripper closer to the cube — which is the desired corrective behavior. Actions are ranked by this delta, smallest first.

Example output:
```
Rank  File                    L2 dist   Difference (action - OOD)
   1  action_0004_seg.npy      42.0724                     -6.8602
   2  action_0002_seg.npy      42.7396                     -6.1930
   ...
Best action: action_0004_seg.npy  (L2 = 42.0724,  difference = -6.8602)
```

---

## Method

### Jacobian-Based IK Perturbation

The OOD displacement is specified in Cartesian space (meters). The algorithm:

1. Load reference joint configuration `q_ref` at the chosen timestep
2. Compute the 3×7 translational Jacobian `J` via finite differences
3. Solve for joint perturbation: `Δq = J⁺ · Δx` (pseudo-inverse least-norm solution)
4. Apply: `q_ood = clip(q_ref + Δq, joint_limits)`
5. Restore the full MuJoCo sim state (preserving cube position) and re-render

The script reports achieved vs. requested EEF displacement for IK accuracy verification.

### Deterministic Scene Reconstruction

Full MuJoCo flat states (`qpos + qvel`) are recorded at each timestep and stored as `sim_state` in the NPZ files. When rendering candidates, each action starts from the identical OOD scene via `env.sim.set_state_from_flattened(ood_sim_state)`, ensuring cube position is preserved across all candidates.

### Action Sampling Strategy

Candidates are unit-vector translations: the 3D translation component is drawn from `N(0,1)³` then L2-normalized, giving uniform random directions on the unit sphere at maximum OSC_POSE magnitude. Rotation and gripper are fixed (zero and open respectively), isolating the effect of translational correction.

---

## Cameras

| Camera | Notes |
|--------|-------|
| `agentview` | First-person wrist-mounted view |
| `frontview` | Front-facing; raw output is vertically flipped |
| `sideview` | Side-facing; raw output is vertically flipped |
| `birdview` | Top-down overhead view |

Vertical flip correction for `frontview` and `sideview` is applied automatically in all scripts via the constant `CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}`.

---

## Known Limitations

- **Gripper state in candidate actions:** The gripper dimension is currently fixed to `-1.0` (open) for all candidates, regardless of the OOD state's actual gripper position. This is intentional for the current evaluation but may need to be matched to the OOD gripper state for tasks where gripper closure matters.
- **Centroid ranking uses ground-truth segmentation:** The `compute_centroid_distance.py` script uses robosuite's instance segmentation, not VLM output. VLM evaluation (Grounded-SAM, Qwen-VL, InternVL2) produces bounding boxes / masks separately and is not yet wired into the ranking pipeline.

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.11 |
| robosuite | 1.5.2 |
| mujoco | 3.6.0 |
| numpy | 2.4.4 |
| h5py | 3.16.0 |
| opencv-python | 4.11.0.86 |

Additional packages for VLM evaluation: `torch`, `transformers`, `pillow`, `sentencepiece` (InternVL2 / Grounded-SAM); `llama-cpp-python` (Qwen-VL).
