"""
Detect the red cube and robot gripper in candidate images using a Qwen-VL
GGUF model (Qwen2.5-VL or Qwen3-VL) via llama-cpp-python.

The model is prompted to return bounding boxes in the format:
    Red cube    -> <box> (x1,y1,x2,y2) </box>
    Robot gripper -> <box> (x1,y1,x2,y2) </box>

Two coordinate conventions are handled automatically:
  • Pixel coordinates  – values clearly within image dimensions (0–256)
  • Normalised [0-1000] – Qwen-VL's native grounding format; converted to pixels

Outputs written alongside each source image:
  <name>_qwen_vis.png   – annotated image with drawn boxes and labels
  <name>_qwen_raw.txt   – raw model response (for debugging)
A combined JSON summary is written to <img_dir>/qwen_detections.json.

Setup:
    # Install llama-cpp-python with Metal (Apple Silicon):
    pip install llama-cpp-python \\
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

    # Download model + mmproj from HuggingFace, e.g.:
    #   bartowski/Qwen2.5-VL-7B-Instruct-GGUF   (lighter, good for testing)
    #   bartowski/Qwen3-VL-30B-A3B-Instruct-GGUF (the 30B MoE variant)
    #
    # You need TWO files from the repo:
    #   *-Q4_K_M.gguf      (the quantised language model)
    #   mmproj-*-f16.gguf  (the vision projector)

Usage:
    python qwen_vl_detect.py \\
        --model  /path/to/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf \\
        --mmproj /path/to/mmproj-Qwen3-VL-30B-A3B-f16.gguf

    python qwen_vl_detect.py \\
        --model  /path/to/model.gguf \\
        --mmproj /path/to/mmproj.gguf \\
        --img_dir candidate_images_sideview \\
        --n_gpu_layers 40
"""

import argparse
import base64
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

PROMPT = (
    "Where is the red cube and the robot gripper in the image?\n"
    "Provide the output as bounding boxes -\n"
    "Red cube -> <box> (x1,y1,x2,y2) </box>\n"
    "Robot gripper -> <box> (x1,y1,x2,y2) </box>"
)

# Colours for drawing: BGR for OpenCV
COLORS = {
    "red cube":       (30,  30, 220),   # red
    "robot gripper":  (220, 80,  30),   # blue
}
DEFAULT_COLOR = (30, 200, 30)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, mmproj_path: str, n_gpu_layers: int, n_ctx: int):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

    print(f"Loading vision projector: {mmproj_path}")
    chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)

    print(f"Loading model:            {model_path}")
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        verbose=False,
    )
    return llm


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def encode_image(img_path: str) -> str:
    """Return a data-URI string for the image."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def run_inference(llm, img_path: str, max_tokens: int) -> str:
    data_uri = encode_image(img_path)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text",      "text": PROMPT},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _to_pixels(val: float, img_dim: int) -> int:
    """
    Convert a single coordinate to pixels.
    Qwen-VL native grounding uses normalised [0, 1000]; values > img_dim
    indicate normalised coords and are rescaled. Otherwise treated as pixels.
    """
    if val > img_dim:
        return int(val / 1000.0 * img_dim)
    return int(val)


def parse_boxes(text: str, img_w: int, img_h: int) -> dict[str, list[int] | None]:
    """
    Extract bounding boxes for 'red cube' and 'robot gripper' from model output.

    Handles three common response patterns:
      1. User-prompted format:  Red cube -> <box> (x1,y1,x2,y2) </box>
      2. Qwen native grounding: <tool_call>...</tool_call>(x1,y1),(x2,y2)</tool_call>
      3. Bare coords after label: Red cube: x1,y1,x2,y2
    """
    results: dict[str, list[int] | None] = {"red cube": None, "robot gripper": None}

    # Pattern 1 & 3 — explicit label before a box token or comma-separated coords
    label_patterns = {
        "red cube":      r"(?:red\s+cube|cube)[^\d]{0,30}?(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)",
        "robot gripper": r"(?:robot\s+gripper|gripper)[^\d]{0,30}?(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)",
    }
    for label, pattern in label_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            x1, y1, x2, y2 = [float(v) for v in m.groups()]
            results[label] = [
                _to_pixels(x1, img_w), _to_pixels(y1, img_h),
                _to_pixels(x2, img_w), _to_pixels(y2, img_h),
            ]

    # Pattern 2 — Qwen native: <tool_call>LABEL</tool_call><tool_call>(x1,y1),(x2,y2)</tool_call>
    native = re.findall(
        r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>",
        text,
        re.IGNORECASE,
    )
    for label_raw, x1, y1, x2, y2 in native:
        label_raw = label_raw.strip().lower()
        for key in results:
            if key in label_raw or any(w in label_raw for w in key.split()):
                results[key] = [
                    _to_pixels(float(x1), img_w), _to_pixels(float(y1), img_h),
                    _to_pixels(float(x2), img_w), _to_pixels(float(y2), img_h),
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
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(llm, img_path: str, max_tokens: int) -> dict:
    img = Image.open(img_path)
    W, H = img.size

    raw_response = run_inference(llm, img_path, max_tokens)
    detections   = parse_boxes(raw_response, W, H)

    stem     = os.path.splitext(img_path)[0]
    vis_path = f"{stem}_qwen_vis.png"
    raw_path = f"{stem}_qwen_raw.txt"

    draw_boxes(img_path, detections, vis_path)
    with open(raw_path, "w") as f:
        f.write(raw_response)

    found = [k for k, v in detections.items() if v is not None]
    missing = [k for k, v in detections.items() if v is None]

    print(f"  [{os.path.basename(img_path)}]")
    for label, box in detections.items():
        status = str(box) if box else "NOT DETECTED"
        print(f"    {label:<20} {status}")
    if missing:
        print(f"    (not detected: {', '.join(missing)} — try lowering --box_threshold"
              " or adjusting the prompt)")

    return {
        "image":      os.path.basename(img_path),
        "detections": {k: v for k, v in detections.items()},
        "raw":        raw_response,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    llm = load_model(args.model, args.mmproj, args.n_gpu_layers, args.n_ctx)
    print(f"\nModel loaded. Running on: {'GPU (Metal)' if args.n_gpu_layers != 0 else 'CPU'}\n")
    print(f"Prompt:\n{PROMPT}\n")

    # Only process action_XXXX.png and ood_state.png; skip vis/raw outputs
    all_pngs = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    images   = [
        p for p in all_pngs
        if not any(p.endswith(s) for s in
                   ("_internvl_vis.png", "_qwen_vis.png", "_gsam_vis.png", "_seg_inspect.png"))
    ]

    if not images:
        print(f"No PNG images found in '{args.img_dir}'.")
        sys.exit(1)

    print(f"Processing {len(images)} image(s) in '{args.img_dir}' …\n")

    all_results = []
    for img_path in images:
        result = process_image(llm, img_path, args.max_tokens)
        all_results.append(result)

    # Save combined JSON (without raw text to keep it readable)
    summary = [
        {"image": r["image"], "detections": r["detections"]}
        for r in all_results
    ]
    json_path = os.path.join(args.img_dir, "qwen_detections.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.")
    print(f"  Visualisations and raw responses written alongside source images.")
    print(f"  Detection summary → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect red cube and robot gripper with Qwen-VL GGUF."
    )
    parser.add_argument("--model",  required=True,
                        help="Path to the quantised model GGUF file "
                             "(e.g. Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf)")
    parser.add_argument("--mmproj", required=True,
                        help="Path to the vision projector GGUF file "
                             "(e.g. mmproj-Qwen3-VL-30B-A3B-f16.gguf)")
    parser.add_argument("--img_dir",      default="candidate_images_agentview",
                        help="Directory containing candidate PNG images "
                             "(default: candidate_images_agentview)")
    parser.add_argument("--max_tokens",   type=int, default=256,
                        help="Maximum tokens to generate per image (default: 256)")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="Layers to offload to GPU; -1 = all (default: -1). "
                             "Set to 0 to run entirely on CPU.")
    parser.add_argument("--n_ctx",        type=int, default=4096,
                        help="Context window size (default: 4096)")
    args = parser.parse_args()
    main(args)
