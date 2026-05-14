import base64
import re
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

IMAGES_DIR = Path("images")
IMAGE_ORDER = ["ood_state.png", "north.png", "south.png", "east.png", "west.png"]
OUTPUT_FILE = "qwen_response_7b_quantized.txt"

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


def image_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_verdicts(response: str) -> dict[str, str]:
    verdicts = {}
    for action in ("NORTH", "SOUTH", "EAST", "WEST"):
        match = re.search(rf"^{action}:\s*(KEEP|DISCARD)", response, re.MULTILINE | re.IGNORECASE)
        if match:
            verdicts[action] = match.group(1).upper()
        else:
            verdicts[action] = "UNKNOWN"
    return verdicts


def main():
    print("Loading model...")
    chat_handler = Qwen25VLChatHandler(clip_model_path="/Users/swapnilmallick/models/qwen25-vl-7b/mmproj-model-f16.gguf")
    model = Llama(
        model_path="/Users/swapnilmallick/models/qwen25-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        chat_handler=chat_handler,
        n_ctx=8192,
        n_threads=8,
        verbose=False,
    )

    print("Loading images...")
    content = []
    for filename in IMAGE_ORDER:
        img_path = IMAGES_DIR / filename
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(img_path)}})

    content.append({"type": "text", "text": PROMPT})

    print("Running inference...")
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.0,
    )

    raw_text = response["choices"][0]["message"]["content"]

    with open(OUTPUT_FILE, "w") as f:
        f.write(raw_text)
    print(f"Raw response saved to {OUTPUT_FILE}")

    verdicts = parse_verdicts(raw_text)
    print()
    for action in ("NORTH", "SOUTH", "EAST", "WEST"):
        print(f"{action}: {verdicts[action]}")


if __name__ == "__main__":
    main()
