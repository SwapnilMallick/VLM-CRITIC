import re
import sys
from pathlib import Path
import torch

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMAGES_DIR = Path("images")
IMAGE_ORDER = ["ood_state.png", "north.png", "south.png", "east.png", "west.png"]
OUTPUT_FILE = "qwen_response.txt"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """You are evaluating a robot manipulation task. The robot must grasp the red cube.

You are given five images in this order:
- ood_state.png: The current OOD state of the robot
- north.png: State after taking action NORTH
- south.png: State after taking action SOUTH
- east.png: State after taking action EAST
- west.png: State after taking action WEST

For each of the four actions (NORTH, SOUTH, EAST, WEST):
1. Describe what you observe about the gripper's position relative to the red cube in that action's image.
2. Compare it to ood_state.png — did the gripper move closer or further from the red cube?
3. Output a verdict: KEEP if the gripper moved closer or maintained distance, DISCARD if the gripper moved further away.

Format your response exactly as follows:
REASONING_NORTH: <your observation for NORTH>
REASONING_SOUTH: <your observation for SOUTH>
REASONING_EAST: <your observation for EAST>
REASONING_WEST: <your observation for WEST>
NORTH: <KEEP or DISCARD>
SOUTH: <KEEP or DISCARD>
EAST: <KEEP or DISCARD>
WEST: <KEEP or DISCARD>"""


def load_images():
    images = {}
    for name in IMAGE_ORDER:
        path = IMAGES_DIR / name
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        images[name] = Image.open(path).convert("RGB")
    return images


def build_messages(images):
    content = []
    for name in IMAGE_ORDER:
        content.append({"type": "image", "image": images[name]})
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


def parse_verdicts(response_text):
    verdicts = {}
    for action in ("NORTH", "SOUTH", "EAST", "WEST"):
        # Match lines like "NORTH: KEEP" or "NORTH: DISCARD" (not REASONING_NORTH)
        pattern = rf"^{action}:\s*(KEEP|DISCARD)"
        match = re.search(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        verdicts[action] = match.group(1).upper() if match else "UNKNOWN"
    return verdicts


def main():
    print("Loading images...")
    images = load_images()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model {MODEL_ID} on {device} (this may take a while)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = build_messages(images)

    print("Preparing inputs...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)
    print("Running inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    # Strip prompt tokens from output
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    with open(OUTPUT_FILE, "w") as f:
        f.write(response)
    print(f"Raw response saved to {OUTPUT_FILE}")

    verdicts = parse_verdicts(response)
    print("\n--- Action Verdicts ---")
    for action in ("NORTH", "SOUTH", "EAST", "WEST"):
        print(f"{action}: {verdicts[action]}")


if __name__ == "__main__":
    main()
