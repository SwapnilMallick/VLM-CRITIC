"""
Apply each candidate action from the OOD state and save the resulting images.

For every action in candidate_actions_<camera>.npz the script:
  1. Restores the exact OOD scene (robot pose + cube position) from the sim
     state saved by create_ood_state.py.
  2. Steps the simulator once with that action.
  3. Saves the rendered frame as action_XXXX.png.

A copy of the pre-action OOD frame is also saved as ood_state.png so you can
visually compare starting state vs. result for each candidate.

Note: ood_state_<camera>.npz must contain a 'sim_state' key.  Re-generate it
with the updated create_ood_state.py if your file pre-dates this change.

Usage:
    python render_candidate_actions.py
    python render_candidate_actions.py --camera sideview --n 16
    python render_candidate_actions.py --actions candidate_actions_sideview.npz \\
                                        --ood ood_state_sideview.npz \\
                                        --out_dir candidate_images_sideview/
"""

import argparse
import os

import cv2
import numpy as np
import robosuite as suite

CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def restore_ood_state(env, ood_sim_state: np.ndarray) -> None:
    """Reset env bookkeeping then restore the exact OOD scene."""
    env.reset()
    env.sim.set_state_from_flattened(ood_sim_state)
    env.sim.forward()


def main(args) -> None:
    actions_data = load_npz(args.actions)
    ood_data     = load_npz(args.ood)

    if "sim_state" not in ood_data:
        raise KeyError(
            f"'sim_state' key not found in '{args.ood}'. "
            "Re-generate the OOD state with the updated create_ood_state.py."
        )

    actions       = actions_data["actions"]     # (N, 7)
    ood_sim_state = ood_data["sim_state"]       # flattened MuJoCo qpos+qvel

    N = len(actions)
    print(f"Loaded {N} candidate actions  from '{args.actions}'")
    print(f"Loaded OOD sim state          from '{args.ood}'")

    os.makedirs(args.out_dir, exist_ok=True)

    vflip   = args.camera in CAMERAS_NEEDING_VFLIP
    obs_key = f"{args.camera}_image"

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
    )

    # Save the OOD frame itself for side-by-side comparison
    restore_ood_state(env, ood_sim_state)
    ood_frame = env._get_observations(force_update=True)[obs_key]
    if vflip:
        ood_frame = ood_frame[::-1]
    cv2.imwrite(
        os.path.join(args.out_dir, "ood_state.png"),
        cv2.cvtColor(ood_frame, cv2.COLOR_RGB2BGR),
    )
    print(f"Saved OOD reference frame → {args.out_dir}/ood_state.png\n")

    width = len(str(N))
    for i, action in enumerate(actions):
        restore_ood_state(env, ood_sim_state)
        obs, _, _, _ = env.step(action)

        frame = obs[obs_key][::-1] if vflip else obs[obs_key]
        out_path = os.path.join(args.out_dir, f"action_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  [{i+1:>{width}}/{N}]  action_{i:04d}.png  "
              f"action={np.round(action, 2)}")

    env.close()
    print(f"\nSaved {N} images + 1 reference → {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the result of each candidate action from the OOD state."
    )
    parser.add_argument("--camera",   default="sideview",
                        help="Camera to render (default: sideview). "
                             "Must match the camera used in create_ood_state.py.")
    parser.add_argument("--ood",      default=None,
                        help="Path to OOD state .npz (default: ood_state_<camera>.npz). "
                             "Must contain a 'sim_state' key.")
    parser.add_argument("--actions",  default=None,
                        help="Path to candidate actions .npz "
                             "(default: candidate_actions_<camera>.npz)")
    parser.add_argument("--out_dir",  default=None,
                        help="Output directory for rendered images "
                             "(default: candidate_images_<camera>/)")
    args = parser.parse_args()

    if args.ood     is None: args.ood     = f"ood_state_{args.camera}.npz"
    if args.actions is None: args.actions = f"candidate_actions_{args.camera}.npz"
    if args.out_dir is None: args.out_dir = f"candidate_images_{args.camera}"

    main(args)
