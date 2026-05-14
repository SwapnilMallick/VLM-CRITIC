import base64
import json
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

IMAGES_DIR = Path("images")
EEF_FILE = Path("eef_positions.txt")
OUTPUT_FILE = Path("qwen_response_7b_quantized_pairwise_eef.txt")
ACTION_STEMS = ["action_0", "action_1", "action_2", "action_3"]

PROMPT_TEMPLATE = """\
You are given two images and their corresponding gripper positions:

- ood_state.png: The current state of the robot
  Gripper position: {eef_ood}

- {action_file}: The state after taking meta-action {action_index}
  Gripper position: {eef_action}

The task is to grasp the red cube.

Step 1 - Using the gripper positions provided, note how the gripper \
moved from {eef_ood} to {eef_action} and in which direction.

Step 2 - Look at ood_state.png and {action_file}. Identify where the \
red cube is in the images relative to the gripper.

Step 3 - Based on the images provided in Step 2, identify the approximate location of the red cube in pixel space. \

Step 4 - Based on the gripper movement direction from Step 1 and the \
red cube location from Step 3, did the gripper move closer to or \
further from the red cube?

Format your response exactly as follows:
REASONING: <your observations from Steps 1, 2 and 3>
VERDICT: <CLOSER or FURTHER>"""


def load_image_as_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_eef_positions(path: Path) -> dict:
    positions = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, value = line.split(" : ", 1)
        positions[key.strip()] = json.loads(value.strip())
    return positions


def parse_response(raw: str) -> str:
    upper = raw.upper()
    if "FURTHER" in upper:
        return "DISCARD"
    if "CLOSER" in upper:
        return "KEEP"
    return "UNKNOWN"


def split_reasoning_verdict(raw: str) -> tuple[str, str]:
    reasoning = ""
    verdict = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("REASONING:"):
            reasoning = stripped[len("REASONING:"):].strip()
        elif stripped.upper().startswith("VERDICT:"):
            verdict = stripped[len("VERDICT:"):].strip()
    return reasoning, verdict


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

    eef_positions = parse_eef_positions(EEF_FILE)
    print(f"Parsed EEF positions: {eef_positions}")

    ood_uri = load_image_as_data_uri(IMAGES_DIR / "ood_state.png")
    eef_ood = eef_positions["ood"]

    results = {}

    for stem in ACTION_STEMS:
        action_file = f"{stem}.png"
        action_index = stem.split("_")[1]
        eef_action = eef_positions[stem]
        action_uri = load_image_as_data_uri(IMAGES_DIR / action_file)

        prompt = PROMPT_TEMPLATE.format(
            action_file=action_file,
            action_index=action_index,
            eef_ood=eef_ood,
            eef_action=eef_action,
        )

        print(f"Running VLM for {action_file}...")
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
        )

        raw = response["choices"][0]["message"]["content"].strip()
        results[action_file] = {
            "eef_ood": eef_ood,
            "eef_action": eef_action,
            "raw": raw,
        }
        print(f"  Done. Response: {raw[:80]}...")

    with open(OUTPUT_FILE, "w") as f:
        for stem in ACTION_STEMS:
            action_file = f"{stem}.png"
            r = results[action_file]
            reasoning, verdict = split_reasoning_verdict(r["raw"])
            f.write(f"{action_file}:\n")
            f.write(f"  EEF_OOD: {r['eef_ood']}\n")
            f.write(f"  EEF_ACTION: {r['eef_action']}\n")
            f.write(f"  REASONING: {reasoning}\n")
            f.write(f"  VERDICT: {verdict}\n")

    print(f"\nResponses saved to {OUTPUT_FILE}\n")
    print("Summary:")
    for stem in ACTION_STEMS:
        action_file = f"{stem}.png"
        decision = parse_response(results[action_file]["raw"])
        print(f"  {action_file}: {decision}")


if __name__ == "__main__":
    main()
