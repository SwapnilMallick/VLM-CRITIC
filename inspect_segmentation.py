"""
Load a segmentation mask (.npy) and the paired RGB image (.png) and identify
which integer ID corresponds to the cube, robot gripper, and background.

Strategy:
  - Background: the ID covering the most pixels (typically 0)
  - Red cube:   the ID whose masked pixels have the highest mean red-to-green ratio
  - Gripper:    the remaining prominent ID(s)

A colour-coded overlay is saved alongside a per-ID summary.

Usage:
    python inspect_segmentation.py
    python inspect_segmentation.py --seg candidate_images_agentview/ood_state_seg.npy
    python inspect_segmentation.py --seg candidate_images_agentview/action_0000_seg.npy
"""

import argparse
import os

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Colours used in the overlay plot (one per ID, up to 10 IDs)
OVERLAY_COLORS = [
    (0.5, 0.5, 0.5),   # ID 0 — grey (background)
    (1.0, 0.2, 0.2),   # ID 1 — red
    (0.2, 0.8, 0.2),   # ID 2 — green
    (0.2, 0.4, 1.0),   # ID 3 — blue
    (1.0, 0.8, 0.0),   # ID 4 — yellow
    (0.8, 0.2, 0.8),   # ID 5 — magenta
    (0.0, 0.8, 0.8),   # ID 6 — cyan
    (1.0, 0.5, 0.0),   # ID 7 — orange
    (0.5, 0.0, 0.5),   # ID 8 — purple
    (0.0, 0.5, 0.0),   # ID 9 — dark green
]


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_seg(path: str) -> np.ndarray:
    seg = np.load(path)
    return seg.squeeze()   # (H, W, 1) → (H, W)


def mean_color_per_id(rgb: np.ndarray, seg: np.ndarray) -> dict:
    """Return {id: mean_rgb_array} for every unique ID in seg."""
    result = {}
    for uid in np.unique(seg):
        mask = seg == uid
        result[uid] = rgb[mask].mean(axis=0)   # (3,) float
    return result


def guess_label(uid: int, pixel_count: int, total_pixels: int,
                mean_rgb: np.ndarray, all_counts: dict) -> str:
    """Heuristic label for an ID based on pixel count and mean colour."""
    r, g, b = mean_rgb
    frac = pixel_count / total_pixels

    # Background: largest region
    if uid == max(all_counts, key=all_counts.get):
        return "background"

    # Red cube: red channel dominant over green and blue
    if r > 100 and r > 1.5 * g and r > 1.5 * b:
        return "red cube"

    # Gripper / robot arm: grey-ish (channels close to each other)
    if abs(float(r) - float(g)) < 40 and abs(float(r) - float(b)) < 40:
        return "gripper / arm"

    return "unknown"


def build_overlay(rgb: np.ndarray, seg: np.ndarray,
                  id_info: list[dict]) -> np.ndarray:
    """
    Blend a semi-transparent colour per ID onto the RGB image.
    id_info: list of dicts with keys 'id', 'color', 'label'.
    """
    overlay = rgb.astype(float) / 255.0
    colored = np.zeros_like(overlay)

    for info in id_info:
        mask = (seg == info["id"])
        for c, val in enumerate(info["color"]):
            colored[:, :, c][mask] = val

    blended = 0.55 * overlay + 0.45 * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def main(args) -> None:
    seg_path = args.seg

    # Derive the paired RGB path: <dir>/action_XXXX_seg.npy → <dir>/action_XXXX.png
    rgb_path = seg_path.replace("_seg.npy", ".png")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(
            f"Expected paired RGB image at '{rgb_path}' — not found. "
            "Pass --rgb explicitly if the path differs."
        )
    if args.rgb:
        rgb_path = args.rgb

    rgb = load_rgb(rgb_path)
    seg = load_seg(seg_path)

    unique_ids    = np.unique(seg)
    total_pixels  = seg.size
    mean_colors   = mean_color_per_id(rgb, seg)
    pixel_counts  = {uid: int((seg == uid).sum()) for uid in unique_ids}

    print(f"\nSegmentation mask : {seg_path}")
    print(f"Paired RGB image  : {rgb_path}")
    print(f"Mask shape        : {seg.shape}  dtype={seg.dtype}")
    print(f"Unique IDs        : {unique_ids.tolist()}")
    print()
    print(f"{'ID':>4}  {'pixels':>8}  {'%':>6}  {'mean R':>7} {'mean G':>7} {'mean B':>7}  label")
    print("-" * 70)

    id_info = []
    for uid in unique_ids:
        count    = pixel_counts[uid]
        mean_rgb = mean_colors[uid]
        label    = guess_label(uid, count, total_pixels, mean_rgb, pixel_counts)
        color    = OVERLAY_COLORS[int(uid) % len(OVERLAY_COLORS)]
        id_info.append({"id": uid, "label": label, "color": color,
                        "count": count, "mean_rgb": mean_rgb})
        print(f"{uid:>4}  {count:>8}  {100*count/total_pixels:>5.1f}%"
              f"  {mean_rgb[0]:>7.1f} {mean_rgb[1]:>7.1f} {mean_rgb[2]:>7.1f}"
              f"  {label}")

    # --- Visualisation ---
    overlay = build_overlay(rgb, seg, id_info)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(rgb);       axes[0].set_title("RGB image");       axes[0].axis("off")
    axes[1].imshow(seg, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Segmentation mask (raw IDs)"); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title("Overlay");         axes[2].axis("off")

    patches = [
        mpatches.Patch(color=info["color"],
                       label=f"ID {info['id']}: {info['label']}  ({info['count']} px)")
        for info in id_info
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    out_path = seg_path.replace("_seg.npy", "_seg_inspect.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\nVisualization saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify which segmentation IDs correspond to cube, gripper, background."
    )
    parser.add_argument(
        "--seg", default="candidate_images_agentview/ood_state_seg.npy",
        help="Path to segmentation mask .npy file (default: candidate_images_agentview/ood_state_seg.npy)"
    )
    parser.add_argument(
        "--rgb", default=None,
        help="Path to paired RGB image (default: auto-derived from --seg path)"
    )
    args = parser.parse_args()
    main(args)
