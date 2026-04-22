"""
Create an out-of-distribution (OOD) state by displacing the end-effector
position at a chosen timestep via Jacobian-based inverse kinematics, simulating
the arm drifting off the reference path.

The perturbation is specified in Cartesian space (metres) rather than joint
space, which makes the magnitude more interpretable: --drift 0.05 always means
"shift the EEF 5 cm in --drift_dir regardless of the robot's configuration".

A numerical translational Jacobian (finite-difference, 3 × 7) is computed at
the reference configuration, then pseudo-inverted to find the joint-space delta
that best achieves the requested EEF displacement.  The resulting joint angles
are clamped to Franka Panda limits before re-rendering.

Usage:
    python create_ood_state.py
    python create_ood_state.py --demo demo_0 --ood_frac 0.3 --drift 0.05 --drift_dir x
    python create_ood_state.py --drift 0.08 --drift_dir "-1,0,0"   # custom direction
    python create_ood_state.py --ref reference_demos.h5 --out ood_state.npz

Outputs:
    ood_state.npz   — dict with keys: image, eef_pos, joint_pos, joint_vel, gripper
    ood_state_ref.npz — dict with keys: image, eef_pos, joint_pos, joint_vel, gripper
      (reference state at the same timestep, for VLM comparison)
    ood_image.png   — the re-rendered OOD frame (for quick visual inspection)
    ref_image.png   — the original reference frame at the same timestep
"""

import argparse
import os

import cv2
import h5py
import numpy as np
import robosuite as suite


# Cameras whose raw output is vertically inverted due to their mount position.
CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}

# Franka Panda joint limits (radians)
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Direction presets (unit vectors)
_DIR_PRESETS: dict[str, np.ndarray] = {
    "x":  np.array([ 1.,  0.,  0.]),
    "-x": np.array([-1.,  0.,  0.]),
    "y":  np.array([ 0.,  1.,  0.]),
    "-y": np.array([ 0., -1.,  0.]),
    "z":  np.array([ 0.,  0.,  1.]),
    "-z": np.array([ 0.,  0., -1.]),
}


def parse_drift_dir(drift_dir: str) -> np.ndarray:
    """Return a normalised (3,) unit vector from a preset key or 'x,y,z' string."""
    if drift_dir in _DIR_PRESETS:
        return _DIR_PRESETS[drift_dir]
    parts = drift_dir.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"--drift_dir must be one of {list(_DIR_PRESETS)} or a comma-separated "
            f"'x,y,z' vector (e.g. '1,0,0'), got: {drift_dir!r}"
        )
    vec = np.array([float(p) for p in parts])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("--drift_dir vector must be non-zero.")
    return vec / norm


def load_reference(h5_path: str, demo_key: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        if demo_key not in f:
            raise KeyError(f"Demo '{demo_key}' not found. Available: {list(f.keys())}")
        print("Available keys:", list(f[demo_key].keys()))
        data = {
            "images":       f[f"{demo_key}/images"][:],
            "eef_pos":      f[f"{demo_key}/robot0_eef_pos"][:],
            "joint_pos":    f[f"{demo_key}/robot0_joint_pos"][:],
            "joint_vel":    f[f"{demo_key}/robot0_joint_vel"][:],
            "gripper_qpos": f[f"{demo_key}/robot0_gripper_qpos"][:],
        }
        if f"{demo_key}/sim_states" in f:
            data["sim_states"] = f[f"{demo_key}/sim_states"][:]
        else:
            print("WARNING: 'sim_states' not found in HDF5. Re-record with the updated "
                  "record_demo.py so that object positions are restored correctly. "
                  "Falling back to env.reset(), which fixes the cube at its default spawn.")
            data["sim_states"] = None
        return data


def _compute_jacobian(env, site_id: int, n_joints: int = 7, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the 3 × n_joints translational Jacobian at the EEF site via
    finite differences.  qpos is restored to its original value after the
    computation so the function has no lasting side effects.

    Args:
        env:       Live robosuite environment (already set to the desired state).
        site_id:   MuJoCo site index for the EEF (robot.eef_site_id["right"]).
        n_joints:  Number of actuated joints to differentiate (7 for Panda).
        eps:       Finite-difference step size in radians.

    Returns:
        J: (3, n_joints) translational Jacobian  [∂eef_pos / ∂q_i].
    """
    q0 = env.sim.data.qpos[:n_joints].copy()
    eef0 = np.array(env.sim.data.site_xpos[site_id])

    J = np.zeros((3, n_joints))
    for i in range(n_joints):
        q_plus = q0.copy()
        q_plus[i] += eps
        env.sim.data.qpos[:n_joints] = q_plus
        env.sim.forward()
        eef_plus = np.array(env.sim.data.site_xpos[site_id])
        J[:, i] = (eef_plus - eef0) / eps

    # Restore original configuration
    env.sim.data.qpos[:n_joints] = q0
    env.sim.forward()
    return J


def render_ood_image(
    delta_eef: np.ndarray,
    ref_sim_state: np.ndarray | None,
    camera: str = "sideview",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject OOD by displacing the end-effector via Jacobian pseudo-inverse IK.

    Steps:
      1. Restore the full MuJoCo scene state recorded at ood_t (robot + objects).
      2. Compute the translational Jacobian J at the reference configuration.
      3. Solve Δq = J⁺ · Δx (least-norm solution for the desired EEF shift).
      4. Apply q_ood = q_ref + Δq, clamped to Panda joint limits.
      5. Run forward kinematics and render.

    Args:
        delta_eef:      Desired EEF displacement (3,) in metres.
        ref_sim_state:  Flattened MuJoCo state (qpos + qvel) from record_demo.py.
                        If None, falls back to env.reset() (cube at default spawn).

    Returns:
        (image_rgb, ood_eef_pos, ood_joint_pos)
          image_rgb    — rendered RGB frame (256, 256, 3) uint8
          ood_eef_pos  — actual EEF position after FK (3,) float64
          ood_joint_pos — perturbed joint positions (7,) float64
    """
    # Seed before reset so the cube's random fallback position is deterministic
    # (only matters when ref_sim_state is None and state restoration must be skipped).
    np.random.seed(0)

    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera,
        camera_heights=256,
        camera_widths=256,
    )
    env.reset()

    if ref_sim_state is not None:
        # Restore the full scene state (robot + object positions/velocities)
        # recorded at ood_t, matching the approach used in robosuite's own
        # playback_demonstrations_from_hdf5.py script.
        env.sim.set_state_from_flattened(ref_sim_state)
        env.sim.forward()   # update site positions / derived quantities

        # Diagnostic: if the restored render doesn't match the recorded frame,
        # set_state_from_flattened is silently failing (format mismatch).
        # Uncomment the block below to write a side-by-side check image:
        #   restored = env._get_observations()["agentview_image"]
        #   cv2.imwrite("debug_restored.png",
        #               cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]

    # --- Jacobian IK: find the joint-space delta for the desired EEF shift ---
    J = _compute_jacobian(env, site_id)          # (3, 7)
    delta_q = np.linalg.pinv(J) @ delta_eef      # (7,) least-norm solution

    q_ref = env.sim.data.qpos[:7].copy()
    q_ood = np.clip(q_ref + delta_q, JOINT_LOWER, JOINT_UPPER)

    # Apply and propagate
    env.sim.data.qpos[:7] = q_ood
    env.sim.forward()

    # True EEF position after FK (consistent with robot0_eef_pos observations)
    ood_eef_pos = np.array(robot.sim.data.site_xpos[site_id])

    # force_update=True re-captures the scene from the current sim state.
    # Without it, _get_observations() returns the cached frame from env.reset(),
    # which shows the cube at the random reset position, not the restored one.
    obs = env._get_observations(force_update=True)
    image = obs[f"{camera}_image"]   # RGB (H, W, 3) uint8
    if camera in CAMERAS_NEEDING_VFLIP:
        image = image[::-1]

    env.close()
    return image, ood_eef_pos, q_ood


def save_image(image_rgb: np.ndarray, path: str) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def build_ood_state(args):
    ref = load_reference(args.ref, args.demo)

    T = len(ref["eef_pos"])
    ood_t = int(T * args.ood_frac)
    print(f"Trajectory length: {T}  |  OOD injection timestep: {ood_t}  ({args.ood_frac*100:.0f}%)")

    # --- Build EEF displacement vector ---
    drift_dir = parse_drift_dir(args.drift_dir)
    delta_eef = drift_dir * args.drift
    print(f"Requested EEF drift: {delta_eef}  (magnitude {args.drift:.4f} m, "
          f"direction '{args.drift_dir}')")

    # --- Jacobian IK + re-render ---
    ref_sim_state = ref["sim_states"][ood_t] if ref["sim_states"] is not None else None
    print("Computing Jacobian and re-rendering OOD image …")
    ood_image, ood_eef_pos, ood_joint_pos = render_ood_image(delta_eef, ref_sim_state, args.camera)

    ref_eef = ref["eef_pos"][ood_t]
    achieved_delta = ood_eef_pos - ref_eef
    print(f"Requested EEF delta (L2): {np.linalg.norm(delta_eef):.4f} m")
    print(f"Achieved  EEF delta (L2): {np.linalg.norm(achieved_delta):.4f} m  "
          f"(IK residual: {np.linalg.norm(achieved_delta - delta_eef):.4f} m)")
    print(f"Joint pos delta    (L2):  "
          f"{np.linalg.norm(ood_joint_pos - ref['joint_pos'][ood_t]):.4f} rad")

    # --- Assemble state dicts ---
    ood_state = {
        "image":     ood_image,
        "eef_pos":   ood_eef_pos,
        "joint_pos": ood_joint_pos,
        "joint_vel": ref["joint_vel"][ood_t],
        "gripper":   ref["gripper_qpos"][ood_t],
    }

    ref_state = {
        "image":     ref["images"][ood_t],
        "eef_pos":   ref_eef,
        "joint_pos": ref["joint_pos"][ood_t],
        "joint_vel": ref["joint_vel"][ood_t],
        "gripper":   ref["gripper_qpos"][ood_t],
    }

    # --- Persist ---
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."

    np.savez(args.out, **ood_state)
    ref_out = (args.out.replace(".npz", "_ref.npz")
               if args.out.endswith(".npz") else args.out + "_ref.npz")
    np.savez(ref_out, **ref_state)

    ood_img_path = os.path.join(out_dir, f"ood_image_{args.camera}.png")
    ref_img_path = os.path.join(out_dir, f"ref_image_{args.camera}.png")
    save_image(ood_image,            ood_img_path)
    save_image(ref["images"][ood_t], ref_img_path)

    print(f"\nOOD state  → {args.out}")
    print(f"Ref state  → {ref_out}")
    print(f"OOD image  → {ood_img_path}")
    print(f"Ref image  → {ref_img_path}")

    return ood_state, ref_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an OOD state by drifting the EEF off the reference path."
    )
    parser.add_argument("--camera",    default="sideview",
                        help="Camera to render (default: sideview). "
                             "Must match the camera used in record_demo.py. "
                             "Choices: agentview, sideview, frontview, birdview, "
                             "robot0_robotview, robot0_eye_in_hand")
    parser.add_argument("--ref",       default=None,
                        help="Path to reference HDF5 file "
                             "(default: reference_demos_<camera>.h5)")
    parser.add_argument("--demo",      default="demo_0",
                        help="Demo key inside the HDF5 file")
    parser.add_argument("--out",       default=None,
                        help="Output path for OOD state (.npz) "
                             "(default: ood_state_<camera>.npz)")
    parser.add_argument("--ood_frac",  type=float, default=0.3,
                        help="Fraction into trajectory to inject OOD (default 0.3)")
    parser.add_argument("--drift",     type=float, default=0.05,
                        help="EEF displacement magnitude in metres (default 0.05)")
    parser.add_argument("--drift_dir", type=str,   default="x",
                        help="Drift direction: preset (x/-x/y/-y/z/-z) or "
                             "comma-separated vector e.g. '1,0,0' (default 'x'). "
                             "Use = syntax for negative presets: --drift_dir=-x")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="RNG seed (unused; kept for CLI compatibility)")
    args = parser.parse_args()
    if args.ref is None:
        args.ref = f"reference_demos_{args.camera}.h5"
    if args.out is None:
        args.out = f"ood_state_{args.camera}.npz"

    build_ood_state(args)
