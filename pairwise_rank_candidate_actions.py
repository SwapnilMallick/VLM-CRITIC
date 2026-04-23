"""
Score candidate actions using Claude as a visual judge (Pairwise Ranking / Option B).

For each pair of candidate actions (A, B), four images are sent to Claude:
  1. REFERENCE  — what the robot should look like at this trajectory timestep
  2. OOD STATE  — where the robot actually is after drifting off the reference path
  3. RESULT A   — robot state after applying candidate action A
  4. RESULT B   — robot state after applying candidate action B

Claude picks which result better corrects the drift (A, B, or tie).
Outcomes are aggregated via Elo rating into a final ranking.

Comparison count: N*(N-1)/2 round-robin pairs.
  N=10  →   45 comparisons
  N=20  →  190 comparisons
  N=50  → 1225 comparisons

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=<your key>

Usage:
    python pairwise_rank_candidate_actions.py --camera frontview
    python pairwise_rank_candidate_actions.py --camera agentview --model claude-haiku-4-5-20251001
    python pairwise_rank_candidate_actions.py \\
        --ref_image  ref_image_frontview.png \\
        --images_dir candidate_images_frontview/ \\
        --actions    candidate_actions_frontview.npz \\
        --out        pairwise_ranks_frontview.json \\
        --n          10
"""

import argparse
import base64
import json
import os
import re
import time
from itertools import combinations

import anthropic
import numpy as np


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert robotics evaluator assessing corrective actions for a "
    "Franka Panda robot arm performing a block-lifting task."
)

USER_PROMPT = """\
A Franka Panda robot is performing a block-lifting task but has drifted off its \
reference trajectory (out-of-distribution state). You must compare two candidate \
actions and decide which one better corrects this drift.

You are shown four images in this order:
  Image 1 — REFERENCE : the correct robot state at this point in the trajectory
  Image 2 — OOD STATE : the robot's actual (drifted) state before either action
  Image 3 — RESULT A  : the robot state after applying candidate action A
  Image 4 — RESULT B  : the robot state after applying candidate action B

Choose the action whose result is closer to the REFERENCE state.

Respond with valid JSON only, no extra text:
{"winner": "<A|B|tie>", "reason": "<one concise sentence>"}"""


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

ELO_K = 32
ELO_START = 1000.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo(
    ratings: dict[int, float], idx_a: int, idx_b: int, score_a: float
) -> None:
    """Update ratings in-place. score_a: 1.0=A wins, 0.5=tie, 0.0=B wins."""
    ea = _expected(ratings[idx_a], ratings[idx_b])
    ratings[idx_a] += ELO_K * (score_a - ea)
    ratings[idx_b] += ELO_K * ((1.0 - score_a) - (1.0 - ea))


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def _img_block(path: str) -> dict:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


# ---------------------------------------------------------------------------
# Single pairwise comparison
# ---------------------------------------------------------------------------

def compare_actions(
    client: anthropic.Anthropic,
    ref_path: str,
    ood_path: str,
    result_a: str,
    result_b: str,
    model: str,
) -> tuple[str, str]:
    """Call Claude with four images; return (winner, reason).

    winner is one of 'A', 'B', or 'tie'.
    """
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    _img_block(ref_path),
                    _img_block(ood_path),
                    _img_block(result_a),
                    _img_block(result_b),
                ],
            }
        ],
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in Claude response: {text!r}")
    parsed = json.loads(match.group())
    winner = str(parsed.get("winner", "tie")).upper().strip()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    reason = str(parsed.get("reason", ""))
    return winner, reason


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

    # Collect only indices for which an image actually exists
    valid_indices = [
        i for i in range(N)
        if os.path.exists(os.path.join(args.images_dir, f"action_{i:04d}.png"))
    ]
    skipped = N - len(valid_indices)
    if skipped:
        print(f"  WARNING: {skipped} image(s) not found and will be skipped.")

    n_valid    = len(valid_indices)
    n_pairs    = n_valid * (n_valid - 1) // 2

    print(f"Model      : {args.model}")
    print(f"Reference  : {args.ref_image}")
    print(f"OOD state  : {ood_path}")
    print(f"Images dir : {args.images_dir}")
    print(f"Actions    : {args.actions}  ({n_valid} valid, {n_pairs} pairs)\n")

    # Elo ratings and match record
    ratings: dict[int, float] = {i: ELO_START for i in valid_indices}
    wins:    dict[int, int]   = {i: 0         for i in valid_indices}
    losses:  dict[int, int]   = {i: 0         for i in valid_indices}
    ties:    dict[int, int]   = {i: 0         for i in valid_indices}
    comparisons = []

    pair_num = 0
    width    = len(str(n_pairs))

    for idx_a, idx_b in combinations(valid_indices, 2):
        pair_num += 1
        path_a = os.path.join(args.images_dir, f"action_{idx_a:04d}.png")
        path_b = os.path.join(args.images_dir, f"action_{idx_b:04d}.png")

        winner, reason = compare_actions(
            client, args.ref_image, ood_path, path_a, path_b, args.model
        )

        if winner == "A":
            score_a = 1.0
            wins[idx_a]   += 1
            losses[idx_b] += 1
        elif winner == "B":
            score_a = 0.0
            losses[idx_a] += 1
            wins[idx_b]   += 1
        else:
            score_a = 0.5
            ties[idx_a] += 1
            ties[idx_b] += 1

        _update_elo(ratings, idx_a, idx_b, score_a)

        comparisons.append({
            "action_a":  idx_a,
            "action_b":  idx_b,
            "winner":    winner,
            "reason":    reason,
            "elo_a_after": round(ratings[idx_a], 2),
            "elo_b_after": round(ratings[idx_b], 2),
        })

        print(
            f"  [{pair_num:>{width}}/{n_pairs}]"
            f"  action_{idx_a:04d} vs action_{idx_b:04d}"
            f"  → winner={winner}  {reason}"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # Build final results sorted by Elo descending
    results = []
    for rank, idx in enumerate(
        sorted(valid_indices, key=lambda i: ratings[i], reverse=True), 1
    ):
        results.append({
            "rank":       rank,
            "action_idx": idx,
            "elo":        round(ratings[idx], 2),
            "wins":       wins[idx],
            "losses":     losses[idx],
            "ties":       ties[idx],
            "action":     actions[idx].tolist(),
        })

    output = {
        "results":     results,
        "comparisons": comparisons,
        "meta": {
            "model":      args.model,
            "n_actions":  n_valid,
            "n_pairs":    n_pairs,
            "elo_k":      ELO_K,
            "elo_start":  ELO_START,
        },
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"{'Rank':<6} {'Action':<14} {'Elo':>7}  {'W':>4} {'L':>4} {'T':>4}")
    print(f"{'─'*70}")
    for r in results:
        print(
            f"  #{r['rank']:<4} action_{r['action_idx']:04d}"
            f"   {r['elo']:>7.1f}"
            f"  {r['wins']:>4} {r['losses']:>4} {r['ties']:>4}"
        )
    print(f"{'─'*70}")
    print(f"\nSaved rankings → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank candidate actions with Claude via pairwise comparisons."
    )
    parser.add_argument("--camera",     default="frontview",
                        help="Camera name used to resolve default file paths "
                             "(default: frontview)")
    parser.add_argument("--ref_image",  default=None,
                        help="Reference image path "
                             "(default: ref_image_<camera>.png)")
    parser.add_argument("--images_dir", default=None,
                        help="Directory of rendered candidate images "
                             "(default: candidate_images_<camera>/)")
    parser.add_argument("--actions",    default=None,
                        help="Candidate actions .npz "
                             "(default: candidate_actions_<camera>.npz)")
    parser.add_argument("--out",        default=None,
                        help="Output JSON path "
                             "(default: pairwise_ranks_<camera>.json)")
    parser.add_argument("--model",      default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6). "
                             "Use claude-haiku-4-5-20251001 for a faster/cheaper option.")
    parser.add_argument("--n",          type=int, default=None,
                        help="Rank only the first N actions (default: all). "
                             "Note: N actions → N*(N-1)/2 API calls.")
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds between API calls to avoid rate-limiting "
                             "(default: 0.5)")
    args = parser.parse_args()

    if args.ref_image  is None: args.ref_image  = f"ref_image_{args.camera}.png"
    if args.images_dir is None: args.images_dir = f"candidate_images_{args.camera}"
    if args.actions    is None: args.actions    = f"candidate_actions_{args.camera}.npz"
    if args.out        is None: args.out        = f"pairwise_ranks_{args.camera}.json"

    main(args)
