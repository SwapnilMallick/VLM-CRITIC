"""
Render 8 compass-direction actions from the OOD state and save each result.

For each of the 8 cardinal/intercardinal directions (N, NE, E, SE, S, SW, W, NW)
the script uses Jacobian pseudo-inverse IK to move the EEF exactly --dist metres
in the x-y plane (no vertical component), then renders the resulting scene.

Each direction is saved to its own subfolder inside --out_dir:
    <out_dir>/
        ood_state.png          # baseline OOD frame for comparison
        north/
            image.png          # RGB frame after displacement
            image_seg.npy      # instance segmentation mask
            image_seg.png      # colorized segmentation
            state.npz          # eef_pos, joint_pos, joint_vel, gripper, sim_state
        northeast/
            ...

Compass convention (robot world frame, viewed from above):
    North  = +y    Northeast = +x+y    East  = +x    Southeast = +x-y
    South  = -y    Southwest = -x-y    West  = -x    Northwest  = -x+y

Usage:
    python render_directional_actions.py
    python render_directional_actions.py --camera agentview --dist 0.025
    python render_directional_actions.py --ood ood_state_agentview.npz --dist 0.05
"""

import argparse
import os

import cv2
import numpy as np
import robosuite as suite

# Cameras whose raw output is vertically inverted due to mount position.
CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}

# Franka Panda joint limits (radians)
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

_S2 = 1.0 / np.sqrt(2)

# 8 compass directions as (label, x-y unit vector) in robot world frame.
# z-component is zero — all displacements are in the horizontal plane.
DIRECTIONS = [
    ("north",     np.array([ 0.,   1.,  0.])),
    ("northeast", np.array([ _S2,  _S2, 0.])),
    ("east",      np.array([ 1.,   0.,  0.])),
    ("southeast", np.array([ _S2, -_S2, 0.])),
    ("south",     np.array([ 0.,  -1.,  0.])),
    ("southwest", np.array([-_S2, -_S2, 0.])),
    ("west",      np.array([-1.,   0.,  0.])),
    ("northwest", np.array([-_S2,  _S2, 0.])),
]


def _compute_jacobian(env, site_id: int, n_joints: int = 7, eps: float = 1e-5) -> np.ndarray:
    """3 × n_joints translational Jacobian via finite differences. Restores qpos after."""
    q0   = env.sim.data.qpos[:n_joints].copy()
    eef0 = np.array(env.sim.data.site_xpos[site_id])

    J = np.zeros((3, n_joints))
    for i in range(n_joints):
        q_plus = q0.copy()
        q_plus[i] += eps
        env.sim.data.qpos[:n_joints] = q_plus
        env.sim.forward()
        J[:, i] = (np.array(env.sim.data.site_xpos[site_id]) - eef0) / eps

    env.sim.data.qpos[:n_joints] = q0
    env.sim.forward()
    return J


def seg_to_png(seg: np.ndarray) -> np.ndarray:
    """Convert instance-ID segmentation array to a colorized BGR image."""
    ids = seg.squeeze()
    color_map = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for uid in np.unique(ids):
        hue = int((uid * 37) % 180)
        bgr = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
        color_map[ids == uid] = bgr
    return color_map


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main(args) -> None:
    ood_data = load_npz(args.ood)
    if "sim_state" not in ood_data:
        raise KeyError(
            f"'sim_state' not found in '{args.ood}'. "
            "Re-run create_ood_state.py to regenerate."
        )

    ood_sim_state = ood_data["sim_state"]
    ood_joint_pos = ood_data["joint_pos"]
    ood_joint_vel = ood_data["joint_vel"]
    ood_gripper   = ood_data["gripper"]

    os.makedirs(args.out_dir, exist_ok=True)

    vflip   = args.camera in CAMERAS_NEEDING_VFLIP
    obs_key = f"{args.camera}_image"
    seg_key = f"{args.camera}_segmentation_instance"

    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
        camera_segmentations="instance",
    )

    def restore_ood(env):
        env.reset()
        env.sim.set_state_from_flattened(ood_sim_state)
        env.sim.forward()

    # Save the OOD baseline frame for comparison
    restore_ood(env)
    ood_obs   = env._get_observations(force_update=True)
    ood_frame = ood_obs[obs_key][::-1] if vflip else ood_obs[obs_key]
    ood_seg   = ood_obs[seg_key][::-1] if vflip else ood_obs[seg_key]
    cv2.imwrite(
        os.path.join(args.out_dir, "ood_state.png"),
        cv2.cvtColor(ood_frame, cv2.COLOR_RGB2BGR),
    )
    np.save(os.path.join(args.out_dir, "ood_state_seg.npy"), ood_seg)
    cv2.imwrite(os.path.join(args.out_dir, "ood_state_seg.png"), seg_to_png(ood_seg))

    ood_eef_pos = ood_data["eef_pos"]
    print(f"OOD EEF position: {ood_eef_pos}")
    print(f"Displacement per direction: {args.dist * 100:.1f} cm\n")
    print(f"{'Direction':<12}  {'Target EEF':<30}  {'Achieved EEF':<30}  {'Error (m)'}")
    print("-" * 90)

    for label, unit_vec in DIRECTIONS:
        # Restore fresh OOD state for every direction.
        # env.reset() inside restore_ood() creates a new env.sim object, so robot
        # and site_id must be re-fetched from env.robots[0] after each restore.
        restore_ood(env)
        robot   = env.robots[0]
        site_id = robot.eef_site_id["right"]

        # Jacobian IK: compute joint delta for the desired Cartesian displacement
        J       = _compute_jacobian(env, site_id)  # (3, 7)
        delta_x = unit_vec * args.dist             # (3,) metres
        delta_q = np.linalg.pinv(J) @ delta_x     # (7,) least-norm solution

        q_new = np.clip(env.sim.data.qpos[:7] + delta_q, JOINT_LOWER, JOINT_UPPER)
        env.sim.data.qpos[:7] = q_new
        env.sim.forward()

        # Use env.sim directly — robot.sim may be stale if env.reset() swapped the sim.
        new_eef_pos = np.array(env.sim.data.site_xpos[site_id])
        achieved_delta = new_eef_pos - ood_eef_pos
        error = np.linalg.norm(achieved_delta - delta_x)

        target_eef = ood_eef_pos + delta_x
        print(f"{label:<12}  {str(np.round(target_eef, 4)):<30}  "
              f"{str(np.round(new_eef_pos, 4)):<30}  {error:.5f}")

        # Render after applying the IK displacement
        obs   = env._get_observations(force_update=True)
        frame = obs[obs_key][::-1] if vflip else obs[obs_key]
        seg   = obs[seg_key][::-1] if vflip else obs[seg_key]

        # Capture the full sim state at this configuration
        new_sim_state = env.sim.get_state().flatten()

        # Save to per-direction subfolder
        sub = os.path.join(args.out_dir, label)
        os.makedirs(sub, exist_ok=True)

        cv2.imwrite(os.path.join(sub, "image.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(sub, "image_seg.npy"), seg)
        cv2.imwrite(os.path.join(sub, "image_seg.png"), seg_to_png(seg))
        np.savez(
            os.path.join(sub, "state.npz"),
            eef_pos   = new_eef_pos,
            joint_pos = q_new,
            joint_vel = ood_joint_vel,   # velocity unchanged (IK only moves qpos)
            gripper   = ood_gripper,
            sim_state = new_sim_state,
            direction = unit_vec,
            dist_m    = np.float64(args.dist),
        )

    env.close()
    print(f"\nResults saved to '{args.out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render 8 compass-direction actions from the OOD state."
    )
    parser.add_argument("--camera",  default="agentview",
                        help="Camera to render (default: agentview).")
    parser.add_argument("--ood",     default=None,
                        help="Path to OOD state .npz (default: ood_state_<camera>.npz). "
                             "Must contain a 'sim_state' key.")
    parser.add_argument("--dist",    type=float, default=0.025,
                        help="EEF displacement magnitude in metres (default: 0.025 = 2.5 cm).")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: directional_actions_<camera>/).")
    args = parser.parse_args()

    if args.ood     is None: args.ood     = f"ood_state_{args.camera}.npz"
    if args.out_dir is None: args.out_dir = f"directional_actions_{args.camera}"

    main(args)
