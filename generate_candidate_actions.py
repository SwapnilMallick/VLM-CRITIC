"""
Generate a set of candidate actions for a given OOD state.

The Panda with OSC_POSE has a 7-dim action space:
    [dx, dy, dz, dax, day, daz, gripper]
where each translational/rotational delta is in [-1, 1] and
gripper = -1 (open) / +1 (close).

Translational dims [dx, dy, dz] are sampled as random unit vectors (uniformly
random direction, magnitude 1.0) so every candidate makes the maximum possible
displacement in a distinct direction.
Rotational dims [dax, day, daz] are fixed to 0.
Gripper is fixed to -1.0 (open) for all candidate actions.

Usage:
    python generate_candidate_actions.py
    python generate_candidate_actions.py --ood ood_state_sideview.npz --n 32
    python generate_candidate_actions.py --ood ood_state_frontview.npz --n 16 --seed 7 --out candidates.npz
"""

import argparse
import os

import numpy as np

ACTION_DIM = 7   # OSC_POSE: [dx, dy, dz, dax, day, daz, gripper]
ACTION_LABELS = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]

GRIPPER_OPEN = -1.0


def load_ood_state(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def generate_actions(
    n: int,
    rng: np.random.Generator,
    gripper_action: float,
) -> np.ndarray:
    """Sample n candidate actions: unit-vector translation, zero rotation, fixed gripper."""
    raw         = rng.normal(0.0, 1.0, size=(n, 3))
    translation = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    rotation    = np.zeros((n, 3))
    gripper_col = np.full((n, 1), gripper_action)
    return np.concatenate([translation, rotation, gripper_col], axis=1)


def print_summary(actions: np.ndarray, gripper_action: float) -> None:
    print(f"\nCandidate actions  shape: {actions.shape}")
    print(f"  gripper fixed to: {gripper_action:+.1f} "
          f"({'open' if gripper_action < 0 else 'close'})")
    print(f"\n{'dim':<10} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print("-" * 46)
    for i, label in enumerate(ACTION_LABELS):
        col = actions[:, i]
        print(f"{label:<10} {col.min():>8.4f} {col.max():>8.4f} "
              f"{col.mean():>8.4f} {col.std():>8.4f}")


def main(args) -> None:
    ood = load_ood_state(args.ood)
    print(f"Loaded OOD state from '{args.ood}'")
    print(f"  eef_pos   : {ood['eef_pos']}")
    print(f"  joint_pos : {ood['joint_pos']}")
    print(f"  gripper   : open (fixed, gripper_action=-1.0)")

    rng = np.random.default_rng(args.seed)
    actions = generate_actions(args.n, rng, GRIPPER_OPEN)

    print_summary(actions, GRIPPER_OPEN)

    np.savez(
        args.out,
        actions=actions,
        ood_eef_pos=ood["eef_pos"],
        ood_joint_pos=ood["joint_pos"],
        gripper_action=np.array(GRIPPER_OPEN),
        seed=np.array(args.seed),
    )
    print(f"\nSaved {args.n} candidate actions → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate candidate actions for an OOD state via Gaussian sampling."
    )
    parser.add_argument("--ood",  default=None,
                        help="Path to OOD state .npz file "
                             "(default: ood_state_<camera>.npz, guessed from --camera)")
    parser.add_argument("--camera", default="sideview",
                        help="Camera name used to auto-resolve default file paths "
                             "(default: sideview)")
    parser.add_argument("--n",    type=int, default=16,
                        help="Number of candidate actions to generate (default: 16)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility (default: 0)")
    parser.add_argument("--out",  default=None,
                        help="Output .npz path "
                             "(default: candidate_actions_<camera>.npz)")
    args = parser.parse_args()

    if args.ood is None:
        args.ood = f"ood_state_{args.camera}.npz"
    if args.out is None:
        # Derive output name from the ood filename so that
        # --ood ood_state_frontview.npz → candidate_actions_frontview.npz
        ood_basename = os.path.basename(args.ood)
        args.out = ood_basename.replace("ood_state_", "candidate_actions_", 1)

    main(args)
