"""
Writes directional_robot_states.txt with three sections:
  1. Reference trajectory robot state at timestep 60 (from reference_demos_agentview.h5)
  2. OOD robot state (from ood_state_agentview.npz)
  3. Robot state after each of the 8 directional actions (from directional_actions_agentview/)
"""

import h5py
import numpy as np
from pathlib import Path

H5_FILE = Path("reference_demos_agentview.h5")
REF_TIMESTEP = 60

OOD_FILE = Path("ood_state_agentview.npz")

BASE_DIR = Path("directional_actions_agentview")
DIRECTIONS = ["east", "west", "north", "south",
              "northeast", "northwest", "southeast", "southwest"]
OUTPUT_FILE = Path("directional_robot_states.txt")


def fmt_array(arr):
    return "[" + ", ".join(f"{v:.6f}" for v in arr.flatten()) + "]"


with OUTPUT_FILE.open("w") as out:
    # ------------------------------------------------------------------
    # Section 1: Reference state at timestep 60
    # ------------------------------------------------------------------
    out.write("Reference Trajectory Robot State\n")
    out.write("=" * 60 + "\n")
    out.write(f"Source : {H5_FILE}  (demo_0, timestep {REF_TIMESTEP})\n")
    out.write("-" * 40 + "\n")

    with h5py.File(H5_FILE, "r") as f:
        g = f["demo_0"]
        out.write(f"  EEF position     : {fmt_array(g['robot0_eef_pos'][REF_TIMESTEP])}\n")
        out.write(f"  Joint positions  : {fmt_array(g['robot0_joint_pos'][REF_TIMESTEP])}\n")
        out.write(f"  Joint velocities : {fmt_array(g['robot0_joint_vel'][REF_TIMESTEP])}\n")
        out.write(f"  Gripper qpos     : {fmt_array(g['robot0_gripper_qpos'][REF_TIMESTEP])}\n")
        out.write(f"  Sim state        : {fmt_array(g['sim_states'][REF_TIMESTEP])}\n")

    out.write("\n\n")

    # ------------------------------------------------------------------
    # Section 2: OOD state
    # ------------------------------------------------------------------
    out.write("OOD Robot State\n")
    out.write("=" * 60 + "\n")
    out.write(f"Source : {OOD_FILE}\n")
    out.write("-" * 40 + "\n")

    ood = np.load(OOD_FILE, allow_pickle=True)
    out.write(f"  EEF position     : {fmt_array(ood['eef_pos'])}\n")
    out.write(f"  Joint positions  : {fmt_array(ood['joint_pos'])}\n")
    out.write(f"  Joint velocities : {fmt_array(ood['joint_vel'])}\n")
    out.write(f"  Gripper qpos     : {fmt_array(ood['gripper'])}\n")
    out.write(f"  Sim state        : {fmt_array(ood['sim_state'])}\n")

    out.write("\n\n")

    # ------------------------------------------------------------------
    # Section 3: Directional action states
    # ------------------------------------------------------------------
    out.write("Directional Action Robot States\n")
    out.write("=" * 60 + "\n\n")

    for direction in DIRECTIONS:
        npz_path = BASE_DIR / direction / "state.npz"
        if not npz_path.exists():
            out.write(f"Direction: {direction.upper()}\n")
            out.write(f"  [state.npz not found at {npz_path}]\n\n")
            continue

        d = np.load(npz_path, allow_pickle=True)

        out.write(f"Direction: {direction.upper()}\n")
        out.write("-" * 40 + "\n")
        out.write(f"  Direction vector : {fmt_array(d['direction'])}\n")
        out.write(f"  Distance (m)     : {float(d['dist_m']):.4f}\n")
        out.write(f"  EEF position     : {fmt_array(d['eef_pos'])}\n")
        out.write(f"  Joint positions  : {fmt_array(d['joint_pos'])}\n")
        out.write(f"  Joint velocities : {fmt_array(d['joint_vel'])}\n")
        out.write(f"  Gripper qpos     : {fmt_array(d['gripper'])}\n")
        out.write(f"  Sim state        : {fmt_array(d['sim_state'])}\n")
        out.write("\n")

print(f"Written to {OUTPUT_FILE}")
