"""
Score candidate actions using Claude as a visual judge (Direct Scoring / Option A).

For each candidate action, three images are sent to Claude:
  1. REFERENCE  — what the robot should look like at this trajectory timestep
  2. OOD STATE  — where the robot actually is after drifting off the reference path
  3. RESULT     — robot state after applying the candidate action

Claude scores each action 1–10 on how well it corrects the drift toward the
reference, along with a one-sentence reason.  Results are sorted by score and
saved as a JSON file.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=<your key>

Usage:
    python score_candidate_actions.py --camera frontview
    python score_candidate_actions.py --camera agentview --model claude-haiku-4-5-20251001
    python score_candidate_actions.py \\
        --ref_image  ref_image_frontview.png \\
        --images_dir candidate_images_frontview/ \\
        --actions    candidate_actions_frontview.npz \\
        --out        scores_frontview.json
"""

import argparse
import base64
import json
import os
import re
import time

import anthropic
import numpy as np


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert robotics evaluator assessing corrective actions for a "
    "Franka Panda robot arm performing a block-lifting task."
)

USER_PROMPT = """\
A Franka Panda robot is performing a block-lifting task but has drifted off its \
reference trajectory (out-of-distribution state). You must judge how well a \
candidate action corrects this drift.

You are shown three images in this order:
  Image 1 — REFERENCE : the correct robot state at this point in the trajectory
  Image 2 — OOD STATE : the robot's actual (drifted) state before the action
  Image 3 — RESULT    : the robot state after applying the candidate action

Score the candidate action from 1 to 10:
  10 — result closely matches the reference; drift is fully corrected
   5 — neutral; no meaningful change toward or away from reference
   1 — result moves further from the reference; drift is worsened

Respond with valid JSON only, no extra text:
{"score": <integer 1-10>, "reason": "<one concise sentence>"}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_block(path: str) -> dict:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image", "source": {"type": "base64",
                                        "media_type": "image/png",
                                        "data": data}}


def score_action(
    client: anthropic.Anthropic,
    ref_path: str,
    ood_path: str,
    result_path: str,
    model: str,
) -> tuple[int, str]:
    """Call Claude with three images; return (score, reason)."""
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    _img_block(ref_path),
                    _img_block(ood_path),
                    _img_block(result_path),
                ],
            }
        ],
    )
    text = response.content[0].text.strip()
    # Use regex to find the first {...} block — handles raw JSON,
    # markdown fences, and any surrounding commentary.
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in Claude response: {text!r}")
    parsed = json.loads(match.group())
    score  = int(parsed.get("score",  0))
    reason = str(parsed.get("reason", ""))
    return score, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client  = anthropic.Anthropic()
    actions = np.load(args.actions, allow_pickle=True)["actions"]   # (N, 7)
    N       = min(len(actions), args.n) if args.n else len(actions)

    ood_path = os.path.join(args.images_dir, "ood_state.png")

    print(f"Model      : {args.model}")
    print(f"Reference  : {args.ref_image}")
    print(f"OOD state  : {ood_path}")
    print(f"Images dir : {args.images_dir}")
    print(f"Actions    : {args.actions}  ({N} to score)\n")

    results = []
    width   = len(str(N))

    for i in range(N):
        result_path = os.path.join(args.images_dir, f"action_{i:04d}.png")
        if not os.path.exists(result_path):
            print(f"  [{i+1:>{width}}/{N}]  SKIP — {result_path} not found")
            continue

        score, reason = score_action(client, args.ref_image, ood_path, result_path, args.model)

        results.append({
            "rank":       None,           # filled in after sorting
            "action_idx": i,
            "score":      score,
            "reason":     reason,
            "action":     actions[i].tolist(),
        })
        print(f"  [{i+1:>{width}}/{N}]  action_{i:04d}  score={score:>2}  {reason}")

        if args.delay > 0:
            time.sleep(args.delay)

    # Sort descending by score and assign ranks
    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # Save
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"{'Rank':<6} {'Action':<14} {'Score':<7} Reason")
    print(f"{'─'*60}")
    for r in results:
        print(f"  #{r['rank']:<4} action_{r['action_idx']:04d}   {r['score']:>2}/10   {r['reason']}")
    print(f"{'─'*60}")
    print(f"\nSaved scores → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score candidate actions with Claude as a visual judge."
    )
    parser.add_argument("--camera",     default="frontview",
                        help="Camera name used to resolve default file paths "
                             "(default: frontview)")
    parser.add_argument("--ref_image",  default=None,
                        help="Reference image path (default: ref_image_<camera>.png)")
    parser.add_argument("--images_dir", default=None,
                        help="Directory of rendered candidate images "
                             "(default: candidate_images_<camera>/)")
    parser.add_argument("--actions",    default=None,
                        help="Candidate actions .npz "
                             "(default: candidate_actions_<camera>.npz)")
    parser.add_argument("--out",        default=None,
                        help="Output JSON path (default: scores_<camera>.json)")
    parser.add_argument("--model",      default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6). "
                             "Use claude-haiku-4-5-20251001 for a faster/cheaper option.")
    parser.add_argument("--n",          type=int, default=None,
                        help="Score only the first N actions (default: all)")
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds to wait between API calls to avoid "
                             "rate-limiting (default: 0.5)")
    args = parser.parse_args()

    if args.ref_image  is None: args.ref_image  = f"ref_image_{args.camera}.png"
    if args.images_dir is None: args.images_dir = f"candidate_images_{args.camera}"
    if args.actions    is None: args.actions    = f"candidate_actions_{args.camera}.npz"
    if args.out        is None: args.out        = f"scores_{args.camera}.json"

    main(args)
