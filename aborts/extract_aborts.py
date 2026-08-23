"""
Extract and categorize abort reasons per game from Playpen/CLEMbench results.

Usage:
    python extract_aborts.py results/Qwen3.5-9B-KTO abort_summary.csv

Walks every <game>/<experiment>/<instance>/ folder under the given model
results directory, checks scores.json for "Aborted": 1, and if aborted,
pulls the reason out of interactions.json (the "invalid format" / "metadata"
events the GM logs right before aborting).

Outputs one row per aborted episode with a coarse category, plus a
per-game summary printed to the console.
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict


def categorize(reason_texts):
    """Bucket an abort based on the GM's logged reason strings."""
    joined = " ".join(reason_texts).lower()
    if "invalid format" in joined or "invalid after max attempts" in joined:
        return "format_violation"
    if "invalid move" in joined or "not a valid" in joined:
        return "protocol_violation"
    if "timeout" in joined or "max_new_tokens" in joined or "truncat" in joined:
        return "truncation_or_timeout"
    if not reason_texts:
        return "no_reason_logged"
    return "other"


def get_abort_reason(interactions_path: Path):
    """Pull GM-logged reason strings from an aborted episode's interactions.json."""
    try:
        data = json.loads(interactions_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    reasons = []
    turns = data.get("turns", [])
    for turn in turns:
        events = turn if isinstance(turn, list) else [turn]
        for e in events:
            if not isinstance(e, dict):
                continue
            action = e.get("action", {})
            atype = action.get("type", "")
            if atype in ("invalid format", "metadata", "invalid move", "error"):
                reasons.append(f"{atype}: {action.get('content', '')}")
    return reasons


def main(model_results_dir: str, out_csv: str):
    root = Path(model_results_dir)
    rows = []
    game_totals = defaultdict(lambda: {"total": 0, "aborted": 0})
    category_by_game = defaultdict(lambda: defaultdict(int))

    for scores_file in root.rglob("scores.json"):
        episode_dir = scores_file.parent
        game = episode_dir.parent.parent.name  # .../<game>/<experiment>/<instance>/
        experiment = episode_dir.parent.name
        instance = episode_dir.name

        try:
            scores = json.loads(scores_file.read_text())
        except json.JSONDecodeError:
            continue

        episode_scores = scores.get("episode scores", {})
        aborted = episode_scores.get("Aborted", 0) == 1

        game_totals[game]["total"] += 1
        if not aborted:
            continue
        game_totals[game]["aborted"] += 1

        interactions_file = episode_dir / "interactions.json"
        reasons = get_abort_reason(interactions_file)
        category = categorize(reasons)
        category_by_game[game][category] += 1

        rows.append({
            "game": game,
            "experiment": experiment,
            "instance": instance,
            "category": category,
            "reason_snippet": " | ".join(reasons)[:200],
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "experiment", "instance", "category", "reason_snippet"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} aborted episodes to {out_csv}\n")
    print(f"{'game':<20}{'total':>8}{'aborted':>10}{'abort%':>10}")
    print("-" * 48)
    for game, counts in sorted(game_totals.items(), key=lambda x: -x[1]["aborted"]):
        pct = 100 * counts["aborted"] / counts["total"] if counts["total"] else 0
        print(f"{game:<20}{counts['total']:>8}{counts['aborted']:>10}{pct:>9.1f}%")

    print(f"\n{'game':<20}{'category':<22}{'count':>8}")
    print("-" * 50)
    for game, cats in category_by_game.items():
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"{game:<20}{cat:<22}{n:>8}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
