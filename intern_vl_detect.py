"""
Detect the red cube and robot gripper in candidate images using InternVL2-4B
via HuggingFace transformers.

InternVL2 natively supports visual grounding and may output bounding boxes as:
    <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
where coordinates are normalized to [0, 1000]. A fallback regex parser also
handles free-form coordinate responses.

Outputs written alongside each source image:
  <name>_internvl_vis.png   – annotated image with drawn boxes and labels
  <name>_internvl_raw.txt   – raw model response (for debugging)
A combined JSON summary is written to <img_dir>/internvl_detections.json.

Setup:
    pip install transformers torch torchvision Pillow sentencepiece

    # The model (~8 GB) is downloaded automatically from HuggingFace on first run.
    # Default model: OpenGVLab/InternVL2-4B

Usage:
    python internvl_detect.py

    python internvl_detect.py \\
        --img_dir candidate_images_sideview

    python internvl_detect.py \\
        --model_name OpenGVLab/InternVL2-2B \\
        --img_dir candidate_images_sideview \\
        --max_new_tokens 512
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

PROMPT = (
    "Detect the red cube and the robot gripper in this image.\n"
    "For each object provide a bounding box using this exact format:\n"
    "Red cube -> <box> (x1,y1,x2,y2) </box>\n"
    "Robot gripper -> <box> (x1,y1,x2,y2) </box>"
)

COLORS = {
    "red cube":      (30,  30, 220),   # red (BGR)
    "robot gripper": (220, 80,  30),   # blue (BGR)
}
DEFAULT_COLOR = (30, 200, 30)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing — InternVL2 dynamic-resolution tiling
# ---------------------------------------------------------------------------

def _build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best_ratio = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size ** 2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=6,
                         image_size=448, use_thumbnail=True) -> list[Image.Image]:
    w, h = image.size
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tr = _closest_aspect_ratio(w / h, target_ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = [
        resized.crop((
            (idx % cols) * image_size,       (idx // cols) * image_size,
            (idx % cols + 1) * image_size,   (idx // cols + 1) * image_size,
        ))
        for idx in range(tr[0] * tr[1])
    ]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image_tensor(img_path: str, input_size=448, max_num=6,
                       dtype=torch.float16, device="cpu") -> torch.Tensor:
    image = Image.open(img_path).convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([transform(t) for t in tiles]).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    # MPS (Apple Silicon) works reliably with float16; bfloat16 on CUDA
    dtype = torch.float16 if device == "mps" else torch.bfloat16

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    print(f"Loading model:     {model_name}  (dtype={dtype}, device={device})")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,       # Flash Attention not available on MPS/CPU
        trust_remote_code=True,
    ).eval().to(device)

    return model, tokenizer, dtype


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, pixel_values: torch.Tensor,
                  max_new_tokens: int) -> str:
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    response = model.chat(tokenizer, pixel_values, f"<image>\n{PROMPT}", gen_cfg)
    return response


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _to_pixels(val: float, img_dim: int) -> int:
    """Convert a coordinate to pixels. InternVL2 normalises to [0, 1000]."""
    if val > img_dim:
        return int(val / 1000.0 * img_dim)
    return int(val)


def parse_boxes(text: str, img_w: int, img_h: int) -> dict[str, list[int] | None]:
    results: dict[str, list[int] | None] = {"red cube": None, "robot gripper": None}

    # Pattern 1: InternVL2 native grounding
    #   <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
    native = re.findall(
        r"<ref>(.*?)</ref>\s*<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>",
        text, re.IGNORECASE,
    )
    for label_raw, x1, y1, x2, y2 in native:
        label_raw = label_raw.strip().lower()
        for key in results:
            if key in label_raw or any(w in label_raw for w in key.split()):
                results[key] = [
                    _to_pixels(float(x1), img_w), _to_pixels(float(y1), img_h),
                    _to_pixels(float(x2), img_w), _to_pixels(float(y2), img_h),
                ]

    # Pattern 2: prompted format and bare coords
    #   Red cube -> <box> (x1,y1,x2,y2) </box>
    label_patterns = {
        "red cube":      r"(?:red\s+cube|cube)[^\d]{0,30}?(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)",
        "robot gripper": r"(?:robot\s+gripper|gripper)[^\d]{0,30}?(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)",
    }
    for label, pattern in label_patterns.items():
        if results[label] is not None:
            continue
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            x1, y1, x2, y2 = [float(v) for v in m.groups()]
            results[label] = [
                _to_pixels(x1, img_w), _to_pixels(y1, img_h),
                _to_pixels(x2, img_w), _to_pixels(y2, img_h),
            ]

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_boxes(img_path: str, detections: dict[str, list[int] | None],
               out_path: str) -> None:
    img = cv2.imread(img_path)
    for label, box in detections.items():
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(model, tokenizer, img_path: str, dtype, device: str,
                  max_new_tokens: int) -> dict:
    img = Image.open(img_path)
    W, H = img.size

    pixel_values = load_image_tensor(img_path, dtype=dtype, device=device)
    raw_response  = run_inference(model, tokenizer, pixel_values, max_new_tokens)
    detections    = parse_boxes(raw_response, W, H)

    stem     = os.path.splitext(img_path)[0]
    vis_path = f"{stem}_internvl_vis.png"
    raw_path = f"{stem}_internvl_raw.txt"

    draw_boxes(img_path, detections, vis_path)
    with open(raw_path, "w") as f:
        f.write(raw_response)

    print(f"  [{os.path.basename(img_path)}]")
    for label, box in detections.items():
        print(f"    {label:<20} {str(box) if box else 'NOT DETECTED'}")
    missing = [k for k, v in detections.items() if v is None]
    if missing:
        print(f"    (not detected: {', '.join(missing)} — try adjusting the prompt)")

    return {
        "image":      os.path.basename(img_path),
        "detections": detections,
        "raw":        raw_response,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer, dtype = load_model(args.model_name, device)
    print(f"\nModel loaded. Running on: {device.upper()}\n")
    print(f"Prompt:\n{PROMPT}\n")

    all_pngs = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    images = [
        p for p in all_pngs
        if not any(p.endswith(s) for s in
                   ("_internvl_vis.png", "_qwen_vis.png",
                    "_gsam_vis.png", "_seg_inspect.png"))
    ]

    if not images:
        print(f"No PNG images found in '{args.img_dir}'.")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) in '{args.img_dir}' …\n")

    all_results = []
    for img_path in images:
        result = process_image(model, tokenizer, img_path, dtype, device,
                               args.max_new_tokens)
        all_results.append(result)

    summary = [
        {"image": r["image"], "detections": r["detections"]}
        for r in all_results
    ]
    json_path = os.path.join(args.img_dir, "internvl_detections.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.")
    print(f"  Visualisations and raw responses written alongside source images.")
    print(f"  Detection summary → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect red cube and robot gripper with InternVL2-4B."
    )
    parser.add_argument("--model_name",     default="OpenGVLab/InternVL2-4B",
                        help="HuggingFace model ID (default: OpenGVLab/InternVL2-4B)")
    parser.add_argument("--img_dir",        default="candidate_images_agentview",
                        help="Directory containing candidate PNG images "
                             "(default: candidate_images_agentview)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate per image (default: 256)")
    args = parser.parse_args()
    main(args)
