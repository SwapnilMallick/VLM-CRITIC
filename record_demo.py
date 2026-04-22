"""
Record a small reference trajectory from robosuite and save it as an HDF5 file.
Uses a scripted policy: move gripper to cube → close gripper → lift up.

Usage:
    python record_demo.py                                        # 1 demo, 200 steps, HDF5 only
    python record_demo.py --video                                # also save mp4 and frames/
    python record_demo.py --camera agentview --video            # use a different camera
    python record_demo.py --n_demos 5 --steps 200 --video
"""

import argparse
import os

import cv2
import h5py
import numpy as np

import robosuite as suite


def scripted_lift_action(obs: dict, t: int) -> np.ndarray:
    """
    Four-phase scripted policy for the Lift task.

    The Panda with default OSC_POSE controller has a 7-dim action space:
        [dx, dy, dz, dax, day, daz, gripper]
    where gripper = -1 open, +1 close.
    (PandaGripper.format_action maps +1 → closed, -1 → open.)

    `gripper_to_cube_pos` = cube_pos - eef_pos (vector FROM EEF TO cube).
    Setting action[i] = (cube[i] + offset) * gain drives the EEF so that
    cube[i] converges to -offset (i.e. EEF ends up `offset` above the cube
    in z, or on top of it in x/y).

    Phase 0 (t <  60): fly to pre-grasp hover 12 cm above cube, gripper open
    Phase 1 (t < 120): descend to grasp height (~1 cm above cube center), open
    Phase 2 (t < 160): actively hold grasp position while closing gripper
    Phase 3 (t >= 160): lift straight up, gripper closed
    """
    cube = obs["gripper_to_cube_pos"]   # (3,) cube_pos - eef_pos

    # Lower gain avoids overshoot/oscillation on the descent; 3.0 gives
    # smooth convergence without bouncing past the target.
    GAIN = 3.0
    action = np.zeros(7)

    if t < 60:
        # Phase 0 — approach hover: align XY, settle 12 cm above cube
        action[0] = np.clip(cube[0] * GAIN, -1, 1)
        action[1] = np.clip(cube[1] * GAIN, -1, 1)
        action[2] = np.clip((cube[2] + 0.12) * GAIN, -1, 1)  # target eef_z = cube_z + 0.12
        action[6] = -1.0                                       # open gripper

    elif t < 120:
        # Phase 1 — descend to grasp height with gripper open so fingers
        # straddle the cube; target eef_z = cube_z + 0.01 (1 cm above
        # cube center) to avoid overshooting through the cube
        action[0] = np.clip(cube[0] * GAIN, -1, 1)
        action[1] = np.clip(cube[1] * GAIN, -1, 1)
        action[2] = np.clip((cube[2] + 0.01) * GAIN, -1, 1)
        action[6] = -1.0                                       # keep gripper open

    elif t < 160:
        # Phase 2 — actively servo to grasp position while closing gripper;
        # continuing to servo (rather than zeroing commands) keeps the arm
        # centred over the cube as the fingers close
        action[0] = np.clip(cube[0] * GAIN, -1, 1)
        action[1] = np.clip(cube[1] * GAIN, -1, 1)
        action[2] = np.clip((cube[2] + 0.01) * GAIN, -1, 1)
        action[6] = 1.0                                        # close gripper

    else:
        # Phase 3 — lift straight up with gripper firmly closed
        action[2] = 1.0
        action[6] = 1.0                                        # keep gripper closed

    return action


# Cameras whose raw output is vertically inverted due to their mount position.
CAMERAS_NEEDING_VFLIP = {"sideview", "frontview"}


def make_video_writer(path: str, fps: int = 20) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(path, fourcc, fps, (256, 256))


def record_demos(n_demos: int, steps_per_demo: int, out_path: str, save_video: bool,
                 camera: str, seed: int = 0) -> None:
    # Fix numpy random seed so env.reset() places the cube at the same position
    # on every run.  Pass --seed -1 to disable seeding (non-deterministic).
    if seed >= 0:
        np.random.seed(seed)
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )

    vflip = camera in CAMERAS_NEEDING_VFLIP
    obs_key = f"{camera}_image"
    out_dir = os.path.dirname(os.path.abspath(out_path))

    with h5py.File(out_path, "w") as f:
        for demo_idx in range(n_demos):
            obs = env.reset()

            images = np.zeros((steps_per_demo, 256, 256, 3), dtype=np.uint8)
            eef_pos = np.zeros((steps_per_demo, 3), dtype=np.float32)
            joint_pos = np.zeros((steps_per_demo, 7), dtype=np.float32)
            joint_vel = np.zeros((steps_per_demo, 7), dtype=np.float32)
            gripper_qpos = np.zeros((steps_per_demo, 2), dtype=np.float32)
            rewards = np.zeros(steps_per_demo, dtype=np.float32)
            # Full flattened MuJoCo state (qpos + qvel for robot AND objects).
            # Needed by create_ood_state.py to restore the exact scene before
            # injecting a joint perturbation, so the cube is in the right place.
            state_dim = len(env.sim.get_state().flatten())
            sim_states = np.zeros((steps_per_demo, state_dim), dtype=np.float64)

            video_path = os.path.join(out_dir, f"demo_{demo_idx}_{camera}.mp4")
            writer = make_video_writer(video_path) if save_video else None

            # One sub-directory per demo for its individual frames
            frames_dir = os.path.join(out_dir, f"demo_{demo_idx}_{camera}_frames")
            if save_video:
                os.makedirs(frames_dir, exist_ok=True)

            for t in range(steps_per_demo):
                action = scripted_lift_action(obs, t)
                obs, reward, done, info = env.step(action)

                frame = obs[obs_key][::-1] if vflip else obs[obs_key]   # RGB
                images[t] = frame
                eef_pos[t] = obs["robot0_eef_pos"]
                joint_pos[t] = obs["robot0_joint_pos"]
                joint_vel[t] = obs["robot0_joint_vel"]
                gripper_qpos[t] = obs["robot0_gripper_qpos"]
                rewards[t] = reward
                sim_states[t] = env.sim.get_state().flatten()

                if writer is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{t:04d}.png"), bgr)

            if writer is not None:
                writer.release()
                print(f"  video  → {video_path}")
                print(f"  frames → {frames_dir}/frame_XXXX.png ({steps_per_demo} files)")

            grp = f.create_group(f"demo_{demo_idx}")
            grp.create_dataset("images", data=images, compression="gzip", compression_opts=4)
            grp.create_dataset("robot0_eef_pos", data=eef_pos)
            grp.create_dataset("robot0_joint_pos", data=joint_pos)
            grp.create_dataset("robot0_joint_vel", data=joint_vel)
            grp.create_dataset("robot0_gripper_qpos", data=gripper_qpos)
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("sim_states", data=sim_states)

            print(f"demo_{demo_idx}: {steps_per_demo} steps, "
                  f"total reward={rewards.sum():.3f}, "
                  f"successes={int((rewards > 0).sum())}")

    env.close()
    print(f"\nSaved {n_demos} demo(s) → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=1, help="Number of demos to record")
    parser.add_argument("--steps", type=int, default=200, help="Steps per demo")
    parser.add_argument("--out", type=str, default=None,
                        help="Output HDF5 path (default: reference_demos_<camera>.h5)")
    parser.add_argument("--video", action="store_true", help="Save each demo as an MP4 alongside the HDF5")
    parser.add_argument("--camera", type=str, default="sideview",
                        help="Camera to render (default: sideview). "
                             "Choices: agentview, sideview, frontview, birdview, "
                             "robot0_robotview, robot0_eye_in_hand")
    parser.add_argument("--seed", type=int, default=0,
                        help="Numpy random seed for reproducible cube placement (default 0; -1 = random)")
    args = parser.parse_args()

    out_path = args.out if args.out is not None else f"reference_demos_{args.camera}.h5"
    record_demos(args.n_demos, args.steps, out_path, args.video, args.camera, args.seed)
