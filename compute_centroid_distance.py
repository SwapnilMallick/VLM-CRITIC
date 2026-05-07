"""
For every segmentation mask matching "action_XXXX_seg.npy" or "ood_state_seg.npy"
in a directory, identify the red-cube label (yellow in the colour-coded PNG) and
the gripper label (pink), compute their centroids, and report the L2 distance.
Actions are ranked by their L2 distance relative to ood_state_seg.npy: a negative
delta means the gripper moved closer to the cube compared to the OOD state (better).
Results are printed to stdout and saved to a text file.

Usage:
    python compute_centroid_distance.py
    python compute_centroid_distance.py --dir candidate_images_frontview
    python compute_centroid_distance.py --dir candidate_images_sideview --out distances.txt
"""

import argparse
import os
import re

import numpy as np
from PIL import Image

SEG_PATTERN = re.compile(r"^(action_\d{4}|ood_state)_seg\.npy$")


def load_seg(path: str) -> np.ndarray:
    return np.load(path).squeeze()  # (H, W, 1) → (H, W)


def load_seg_png(seg_path: str) -> np.ndarray:
    png_path = seg_path.replace("_seg.npy", "_seg.png")
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Expected colour-coded PNG at '{png_path}'")
    return np.array(Image.open(png_path).convert("RGB"))


def label_colors(seg: np.ndarray, seg_png: np.ndarray) -> dict[int, np.ndarray]:
    """Return {label: median_RGB} sampled from the colour-coded PNG."""
    colors = {}
    for label in np.unique(seg):
        ys, xs = np.where(seg == label)
        colors[label] = np.median(seg_png[ys, xs], axis=0)
    return colors


def is_yellow(rgb: np.ndarray) -> bool:
    r, g, b = rgb
    return r > 120 and g > 120 and b < 100 and abs(float(r) - float(g)) < 80


def is_pink(rgb: np.ndarray) -> bool:
    r, g, b = rgb
    return r > 120 and b > 120 and g < 100


def find_label(colors: dict[int, np.ndarray], predicate) -> int | None:
    for label, rgb in colors.items():
        if predicate(rgb):
            return label
    return None


def centroid(seg: np.ndarray, label: int) -> tuple[float, float]:
    ys, xs = np.where(seg == label)
    return float(ys.mean()), float(xs.mean())


def process_file(seg_path: str) -> dict:
    seg = load_seg(seg_path)
    seg_png = load_seg_png(seg_path)
    colors = label_colors(seg, seg_png)

    cube_label = find_label(colors, is_yellow)
    gripper_label = find_label(colors, is_pink)

    if cube_label is None:
        raise RuntimeError(f"{seg_path}: could not identify red-cube label (yellow).")
    if gripper_label is None:
        raise RuntimeError(f"{seg_path}: could not identify gripper label (pink).")

    cube_cy, cube_cx = centroid(seg, cube_label)
    grip_cy, grip_cx = centroid(seg, gripper_label)
    l2 = float(np.sqrt((cube_cy - grip_cy) ** 2 + (cube_cx - grip_cx) ** 2))

    return {
        "file": os.path.basename(seg_path),
        "cube_label": cube_label,
        "cube_px": int((seg == cube_label).sum()),
        "cube_centroid": (cube_cy, cube_cx),
        "gripper_label": gripper_label,
        "gripper_px": int((seg == gripper_label).sum()),
        "gripper_centroid": (grip_cy, grip_cx),
        "l2": l2,
    }


def main(args) -> None:
    seg_files = sorted(
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if SEG_PATTERN.match(f)
    )

    if not seg_files:
        raise SystemExit(f"No matching seg files found in '{args.dir}'.")

    # Process all files
    results = {}
    for seg_path in seg_files:
        try:
            results[seg_path] = process_file(seg_path)
        except RuntimeError as e:
            print(f"WARNING: {e}")
            results[seg_path] = None

    # ── Details table ───────────────────────────────────────────────────────
    detail_lines = [
        f"{'File':<30}  {'Cube lbl':>8}  {'Cube px':>8}  {'Cube centroid':>22}"
        f"  {'Grip lbl':>8}  {'Grip px':>8}  {'Grip centroid':>22}  {'L2 dist':>10}",
        "-" * 115,
    ]

    for seg_path in seg_files:
        r = results[seg_path]
        if r is None:
            detail_lines.append(f"{os.path.basename(seg_path):<30}  ERROR")
            continue
        cy1, cx1 = r["cube_centroid"]
        cy2, cx2 = r["gripper_centroid"]
        detail_lines.append(
            f"{r['file']:<30}  {r['cube_label']:>8}  {r['cube_px']:>8}"
            f"  {f'({cy1:.2f}, {cx1:.2f})':>22}"
            f"  {r['gripper_label']:>8}  {r['gripper_px']:>8}"
            f"  {f'({cy2:.2f}, {cx2:.2f})':>22}  {r['l2']:>10.4f}"
        )

    # ── Ranking table ───────────────────────────────────────────────────────
    ood_path = os.path.join(args.dir, "ood_state_seg.npy")
    if ood_path not in results or results[ood_path] is None:
        raise SystemExit("ood_state_seg.npy not found or failed — cannot rank actions.")

    ood_l2 = results[ood_path]["l2"]

    action_results = [
        r for path, r in results.items()
        if r is not None and r["file"] != "ood_state_seg.npy"
    ]
    action_results.sort(key=lambda r: r["l2"] - ood_l2)

    best = action_results[0]
    best_delta = best["l2"] - ood_l2

    rank_lines = [
        f"\nOOD reference  —  ood_state_seg.npy  L2 = {ood_l2:.4f}",
        "",
        f"{'Rank':>6}  {'File':<30}  {'L2 dist':>10}  {'Difference (action - OOD)':>26}",
        "-" * 80,
    ]
    for rank, r in enumerate(action_results, start=1):
        delta = r["l2"] - ood_l2
        rank_lines.append(
            f"{rank:>6}  {r['file']:<30}  {r['l2']:>10.4f}  {delta:>+26.4f}"
        )

    rank_lines += [
        "",
        f"Best action: {best['file']}  (L2 = {best['l2']:.4f},  difference = {best_delta:+.4f})",
    ]

    # ── Print & save ────────────────────────────────────────────────────────
    for line in detail_lines:
        print(line)
    print()
    for line in rank_lines:
        print(line)

    out_path = args.out or os.path.join(args.dir, "centroid_distances.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(detail_lines) + "\n")
        f.write("\n".join(rank_lines) + "\n")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cube-to-gripper centroid L2 distances for all seg files in a directory."
    )
    parser.add_argument(
        "--dir",
        default="candidate_images_frontview",
        help="Directory containing *_seg.npy and *_seg.png files (default: candidate_images_frontview)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output text file path (default: <dir>/centroid_distances.txt)",
    )
    args = parser.parse_args()
    main(args)
