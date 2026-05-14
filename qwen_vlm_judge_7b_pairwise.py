import base64
import re
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

IMAGES_DIR = Path("images")
OUTPUT_FILE = "qwen_response_7b_quantized_pairwise.txt"
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

PAIRWISE_PROMPT = """You are given two images:
- ood_state.png: The current state of the robot
- {action}.png: The state after taking action {ACTION}

The task is to grasp the red cube.

First, describe what you observe about the gripper's position relative to the red cube in both images. Then, determine whether the gripper moved closer to or further from the red cube in {action}.png compared to ood_state.png.

Format your response exactly as follows:
REASONING: <your observation comparing the gripper position in both images>
VERDICT: <CLOSER or FURTHER>"""


def image_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_response(text: str) -> tuple[str, str]:
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=VERDICT:|$)", text, re.DOTALL | re.IGNORECASE)
    verdict_match = re.search(r"VERDICT:\s*(CLOSER|FURTHER)", text, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
    return reasoning, verdict


chat_handler = Qwen25VLChatHandler(clip_model_path="/Users/swapnilmallick/models/qwen25-vl-7b/mmproj-model-f16.gguf")
model = Llama(
    model_path="/Users/swapnilmallick/models/qwen25-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    chat_handler=chat_handler,
    n_ctx=8192,
    n_threads=8,
    verbose=False,
)

ood_uri = image_to_data_uri(IMAGES_DIR / "ood_state.png")

results: dict[str, tuple[str, str]] = {}

for action in ACTIONS:
    action_lower = action.lower()
    action_uri = image_to_data_uri(IMAGES_DIR / f"{action_lower}.png")
    prompt = PAIRWISE_PROMPT.format(action=action_lower, ACTION=action)

    response = model.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ood_uri}},
                    {"type": "image_url", "image_url": {"url": action_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )

    raw_text = response["choices"][0]["message"]["content"]
    reasoning, verdict = parse_response(raw_text)
    results[action] = (reasoning, verdict)
    print(f"  [{action}] verdict={verdict}")

with open(OUTPUT_FILE, "w") as f:
    for action in ACTIONS:
        reasoning, verdict = results[action]
        f.write(f"{action}:\n")
        f.write(f"  REASONING: {reasoning}\n")
        f.write(f"  VERDICT: {verdict}\n")

print(f"\nResponses saved to {OUTPUT_FILE}\n")

for action in ACTIONS:
    _, verdict = results[action]
    decision = "KEEP" if verdict == "CLOSER" else "DISCARD"
    print(f"{action}: {decision}")
