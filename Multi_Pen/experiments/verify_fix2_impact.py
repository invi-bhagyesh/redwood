#!/usr/bin/env python3
"""
Deeper verification of Fix 2: Does the per-turn refusal budget reset
actually affect results, given that the analysis uses 'refusals mode'
which counts refusals as additional rounds?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("FIX 2 DEEP DIVE: Impact of per-turn refusal budget reset")
    print("=" * 70)

    # The key question: In the "refusals mode" analysis, refusals count as
    # additional rounds. So a conversation with 8 actual rounds + 31 refusals
    # gets 39 "effective rounds". But the plot only goes to 8 on x-axis.
    #
    # For multi-turn: get_max_score_by_round_with_refusals treats each refusal
    # as a round with score 0.0. So if there are many refusals early on,
    # the multi-turn conversation gets "pushed right" - its good scores
    # appear at higher round numbers.
    #
    # For single-turn: each single-turn attempt includes its refusals.
    # get_all_scores_including_refusals returns [0.0, 0.0, ..., final_score].
    # These are all treated as separate samples for expected_max.

    batch_paths = [
        "clean_results/final_runs/batch6A",
        "clean_results/final_runs/batch6B",
    ]

    from asr_analysis_utils import (
        get_max_score_by_round_with_refusals,
        get_max_score_by_round,
        read_json_file
    )

    print("\n--- Multi-turn: Comparing standard vs refusals-mode round counting ---")

    for batch_path in batch_paths:
        batch_name = Path(batch_path).name
        model = "Gemini" if "6A" in batch_name else "Claude"

        root = Path(batch_path)
        multi_files = sorted(root.rglob("*multi_turn*.jsonl"))

        total_effective_rounds = []
        total_actual_rounds = []

        print(f"\n{model} ({batch_name}):")
        print(f"{'Test Case':<40} | {'Actual Rounds':>13} | {'Effective Rounds':>15} | {'Refusals':>8}")
        print("-" * 85)

        for jsonl_file in multi_files:
            data = read_json_file(str(jsonl_file))
            if not data:
                continue

            # Standard: only count actual scored rounds
            standard = get_max_score_by_round(data, 8)
            actual_rounds = sum(1 for k, v in standard.items() if v is not None and 'max_score_by_' in k)

            # Refusals mode: count refusals as additional rounds
            refusals = get_max_score_by_round_with_refusals(data, 8)
            effective_rounds = len([k for k in refusals.keys() if 'max_score_by_' in k])

            # Count actual refusals from file
            lines = []
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        lines.append(json.loads(line))
                    except:
                        continue
            num_refusals = sum(1 for l in lines[1:] if l.get("score") == "refused")

            tc_name = jsonl_file.stem.replace("direct_request_", "").replace("_gemini-2.5-flash_multi_turn_sample1", "").replace("_claude-haiku-4.5_multi_turn_sample1", "")
            print(f"{tc_name:<40} | {actual_rounds:>13} | {effective_rounds:>15} | {num_refusals:>8}")

            total_effective_rounds.append(effective_rounds)
            total_actual_rounds.append(actual_rounds)

        print(f"\n  Mean actual rounds: {np.mean(total_actual_rounds):.1f}")
        print(f"  Mean effective rounds (with refusals): {np.mean(total_effective_rounds):.1f}")
        print(f"  Max effective rounds: {max(total_effective_rounds)}")

    # Now check if this matters for the Figure 2 plot
    print("\n\n--- Impact on Figure 2 ---")
    print("Figure 2 plots 'Number of Samples/Turns' from 1 to 8.")
    print("In refusals mode:")
    print("  - Multi-turn: refusals push scores to higher 'round' numbers")
    print("  - A conversation that gets a good score on round 3 but had 5 refusals")
    print("    before that would have the good score at effective round 8, not round 3")
    print("  - This means at x=3 on the plot, multi-turn shows LOWER scores")
    print("    (because the good scores are delayed by refusals)")
    print()
    print("Wait -- does this HELP or HURT multi-turn?")
    print("  If refusals push multi-turn scores to higher x values,")
    print("  then at any given x, multi-turn looks WORSE (lower).")
    print("  But multi-turn also gets MORE retries (up to 80 vs 10),")
    print("  which means more chances to get high scores overall.")
    print()
    print("Net effect: The extra retries from budget reset give multi-turn more")
    print("chances to break through, but the refusals-mode analysis counts those")
    print("extra retries as additional rounds. It's a complex interaction.")
    print()
    print("The budget reset is a real paper-code mismatch: the paper says 'cumulative limit'")
    print("but the code resets per turn. However, since both conditions are analyzed")
    print("in 'refusals mode' which accounts for the extra retries by counting them,")
    print("the net effect on Figure 2 is ambiguous.")
    print()
    print("Key question: Would enforcing a global budget of 10 total refusals across")
    print("all turns change the conversation dynamics enough to alter which scores are")
    print("achieved? This can only be verified by re-running generate_data.py (~$20).")


if __name__ == "__main__":
    main()
