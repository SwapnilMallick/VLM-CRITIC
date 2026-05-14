import sys
from pathlib import Path

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

IMAGES_DIR = Path("images")
OUTPUT_FILE = "qwen_response_pairwise.txt"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

PAIRWISE_PROMPT = """You are given two images:
- ood_state.png: The current state of the robot
- {action}.png: The state after taking action {ACTION}

The task is to grasp the red cube.

Did the gripper move closer to or further from the red cube in {action}.png
compared to ood_state.png?

Answer with a single word only: CLOSER or FURTHER"""


def load_image(name: str) -> Image.Image:
    path = IMAGES_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    return Image.open(path).convert("RGB")


def build_messages(ood_image: Image.Image, action_image: Image.Image, action: str) -> list:
    prompt = PAIRWISE_PROMPT.format(action=action.lower(), ACTION=action.upper())
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ood_image},
                {"type": "image", "image": action_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_inference(model, processor, messages: list) -> str:
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
    generated_ids = model.generate(**inputs, max_new_tokens=16)
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_verdict(response: str) -> str:
    upper = response.upper()
    if "CLOSER" in upper:
        return "KEEP"
    if "FURTHER" in upper:
        return "DISCARD"
    return "UNKNOWN"


def main():
    print("Loading model on CPU (this may take a while)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading OOD state image...")
    ood_image = load_image("ood_state.png")

    raw_responses = {}
    for action in ACTIONS:
        print(f"Running inference for {action}...")
        action_image = load_image(f"{action.lower()}.png")
        messages = build_messages(ood_image, action_image, action)
        raw_responses[action] = run_inference(model, processor, messages)

    with open(OUTPUT_FILE, "w") as f:
        for action in ACTIONS:
            f.write(f"{action}: {raw_responses[action]}\n")
    print(f"\nRaw responses saved to {OUTPUT_FILE}")

    print("\n--- Action Verdicts ---")
    for action in ACTIONS:
        verdict = parse_verdict(raw_responses[action])
        print(f"{action}: {verdict}")


if __name__ == "__main__":
    main()
