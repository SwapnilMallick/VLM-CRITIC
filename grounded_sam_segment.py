"""
Segment the red cube and robot gripper in every candidate image using
Grounded-SAM (Grounding DINO + SAM) via HuggingFace transformers.

Pipeline per image:
  1. Grounding DINO: text prompt → bounding boxes + confidence scores
  2. SAM: bounding boxes → binary masks (best of 3 candidates selected by IoU score)

Outputs written next to each source image:
  <name>_gsam_vis.png   — RGB overlay with coloured masks and labelled boxes
  <name>_gsam_masks.npz — raw data: masks (bool), boxes (xyxy), labels, scores

Usage:
    python grounded_sam_segment.py
    python grounded_sam_segment.py --img_dir candidate_images_sideview
    python grounded_sam_segment.py --box_threshold 0.25 --text_threshold 0.20
    python grounded_sam_segment.py --device cpu
"""

import argparse
import os
import glob

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

# HuggingFace model IDs
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID   = "facebook/sam-vit-base"

# Text prompt: period-separated phrases, all lowercase (Grounding DINO requirement)
TEXT_PROMPT = "red cube. robot end-effector."

# Mask overlay colours (RGBA, values in [0, 1])
LABEL_COLORS = {
    "red cube":      (1.0, 0.15, 0.15, 0.50),
    "robot end-effector": (0.15, 0.40, 1.00, 0.50),
    "default":       (0.10, 0.90, 0.10, 0.50),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: str):
    print(f"Loading Grounding DINO  ({GDINO_MODEL_ID}) …")
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)

    print(f"Loading SAM             ({SAM_MODEL_ID}) …")
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model     = SamModel.from_pretrained(SAM_MODEL_ID).to(device)

    return gdino_processor, gdino_model, sam_processor, sam_model


# ---------------------------------------------------------------------------
# Detection (Grounding DINO)
# ---------------------------------------------------------------------------

def detect(image_pil: Image.Image,
           gdino_processor, gdino_model,
           box_threshold: float,
           text_threshold: float,
           device: str) -> tuple[list, list, list]:
    """
    Returns (boxes_xyxy, scores, labels) — all Python lists.
    boxes_xyxy: each element is [x1, y1, x2, y2] in pixel coords.
    """
    inputs  = gdino_processor(images=image_pil, text=TEXT_PROMPT,
                               return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    W, H = image_pil.size
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )[0]

    boxes  = results["boxes"].cpu().tolist()    # [[x1,y1,x2,y2], ...]
    scores = results["scores"].cpu().tolist()
    labels = results["text_labels"]             # list[str]  (renamed in transformers≥4.51)
    return boxes, scores, labels


# ---------------------------------------------------------------------------
# Segmentation (SAM)
# ---------------------------------------------------------------------------

def segment(image_pil: Image.Image,
            boxes_xyxy: list,
            sam_processor, sam_model,
            device: str) -> np.ndarray:
    """
    Returns bool array of shape (N, H, W) — one mask per detected box.
    Selects the candidate mask with the highest IoU score from SAM's 3 outputs.
    SAM always runs on CPU: MPS does not support the float64 ops SAM requires.
    """
    if not boxes_xyxy:
        return np.zeros((0, image_pil.size[1], image_pil.size[0]), dtype=bool)

    sam_device = "cpu"

    # SAM processor expects input_boxes as [[[x1,y1,x2,y2], ...]] (batch × boxes × 4)
    sam_inputs = sam_processor(
        images=image_pil,
        input_boxes=[[boxes_xyxy]],
        return_tensors="pt",
    ).to(sam_device)
    sam_model.to(sam_device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    # post_process_masks returns list[Tensor(N, 3, H, W)]
    masks_3 = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]                                  # (N, 3, H, W) bool

    iou_scores = sam_outputs.iou_scores[0].cpu()    # (N, 3)
    best_idx   = iou_scores.argmax(dim=-1)           # (N,)
    best_masks = torch.stack(
        [masks_3[i, best_idx[i]] for i in range(len(best_idx))]
    ).numpy()                             # (N, H, W) bool
    return best_masks


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(image_pil: Image.Image,
              boxes: list, scores: list, labels: list,
              masks: np.ndarray,
              out_path: str) -> None:

    img_np = np.array(image_pil)
    overlay = img_np.astype(float) / 255.0

    # Apply each mask as a coloured tint
    for mask, label in zip(masks, labels):
        color = LABEL_COLORS.get(label, LABEL_COLORS["default"])
        for c in range(3):
            overlay[:, :, c][mask] = (
                (1 - color[3]) * overlay[:, :, c][mask] + color[3] * color[c]
            )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: original
    axes[0].imshow(img_np)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # Right: overlay with boxes
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Grounded-SAM detections")
    axes[1].axis("off")

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        color = LABEL_COLORS.get(label, LABEL_COLORS["default"])[:3]
        rect  = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(y1 - 4, 0), f"{label}  {score:.2f}",
            color="white", fontsize=7,
            bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"),
        )

    # Legend
    patches = [
        mpatches.Patch(
            color=LABEL_COLORS.get(lbl, LABEL_COLORS["default"])[:3],
            label=f"{lbl}  ({score:.2f})",
        )
        for lbl, score in zip(labels, scores)
    ]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(img_path: str,
                  gdino_processor, gdino_model,
                  sam_processor, sam_model,
                  box_threshold: float,
                  text_threshold: float,
                  device: str) -> None:

    image_pil = Image.open(img_path).convert("RGB")
    stem      = os.path.splitext(img_path)[0]

    # --- detect ---
    boxes, scores, labels = detect(
        image_pil, gdino_processor, gdino_model,
        box_threshold, text_threshold, device,
    )

    if not boxes:
        print(f"  [{os.path.basename(img_path)}]  no detections above threshold")
        # Still write an empty results file so callers don't need to special-case
        np.savez(f"{stem}_gsam_masks.npz",
                 masks=np.array([], dtype=bool),
                 boxes=np.array([], dtype=float),
                 scores=np.array([], dtype=float),
                 labels=np.array([], dtype=str))
        return

    # --- segment ---
    masks = segment(image_pil, boxes, sam_processor, sam_model, device)

    # --- save masks ---
    npz_path = f"{stem}_gsam_masks.npz"
    np.savez(
        npz_path,
        masks=masks,
        boxes=np.array(boxes),
        scores=np.array(scores),
        labels=np.array(labels),
    )

    # --- save visualisation ---
    vis_path = f"{stem}_gsam_vis.png"
    visualise(image_pil, boxes, scores, labels, masks, vis_path)

    det_str = ", ".join(f"{lbl} ({sc:.2f})" for lbl, sc in zip(labels, scores))
    print(f"  [{os.path.basename(img_path)}]  {det_str}")
    print(f"    masks → {os.path.basename(npz_path)}")
    print(f"    vis   → {os.path.basename(vis_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    # Pick device
    if args.device == "auto":
        device = ("mps"  if torch.backends.mps.is_available()  else
                  "cuda" if torch.cuda.is_available()           else "cpu")
    else:
        device = args.device
    print(f"Device: {device}\n")

    gdino_processor, gdino_model, sam_processor, sam_model = load_models(device)
    print()

    # Collect images: skip segmentation masks and inspection outputs
    all_pngs = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    images   = [p for p in all_pngs
                if not any(p.endswith(s) for s in
                           ("_qwen_vis.png.", "_gsam_vis.png", "_seg_inspect.png"))]

    if not images:
        print(f"No PNG images found in '{args.img_dir}'.")
        return

    print(f"Processing {len(images)} image(s) in '{args.img_dir}' …\n")
    for img_path in images:
        process_image(
            img_path,
            gdino_processor, gdino_model,
            sam_processor, sam_model,
            args.box_threshold,
            args.text_threshold,
            device,
        )

    print(f"\nDone. Results written alongside source images in '{args.img_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment red cube and robot gripper with Grounded-SAM."
    )
    parser.add_argument("--img_dir",        default="candidate_images_agentview",
                        help="Directory containing candidate PNG images "
                             "(default: candidate_images_agentview)")
    parser.add_argument("--box_threshold",  type=float, default=0.30,
                        help="Grounding DINO box confidence threshold (default: 0.30)")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Grounding DINO text similarity threshold (default: 0.25)")
    parser.add_argument("--device",         default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Inference device (default: auto)")
    args = parser.parse_args()
    main(args)
