"""
04_inspect.py — Parse the pilot results and produce a spreadsheet for hand review.

Inputs:
    pilot_results.csv         (from 03_run_pilot.py — has raw_output)

Outputs:
    pilot_inspection.csv      (parsed labels + correctness + an empty notes column)
    Console summary           (per-prompt accuracy, parse-failure rate, confusion matrix)

This is the most important script in the pilot. Open the CSV and look at every
row. The numbers don't matter yet — the *behavior* matters.
"""

import sys
from pathlib import Path
import pandas as pd
from collections import Counter

from parser import parse


def main():
    src = Path("pilot_results.csv")
    if not src.exists():
        print(f"{src} not found. Run 03_run_pilot.py first.")
        sys.exit(1)

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from {src}")

    # Parse every row
    parsed = df["raw_output"].apply(parse)
    df["parsed_label"] = parsed.apply(lambda x: x[0])
    df["parse_method"] = parsed.apply(lambda x: x[1])
    df["correct"] = (df["parsed_label"] == df["true_label"]).astype(int)
    df["notes"] = ""   # for hand annotation

    # Per-prompt summary
    print("\n" + "=" * 70)
    print(f"{'prompt':6} {'n':>4} {'parse_fail':>11} {'strict':>7} {'syn':>5} {'fb':>4} {'correct':>9}")
    print("-" * 70)
    for pid, sub in df.groupby("prompt_id"):
        n = len(sub)
        pf = (sub["parsed_label"] == "PARSE_FAIL").sum()
        strict = (sub["parse_method"] == "strict").sum()
        syn = (sub["parse_method"] == "synonym").sum()
        fb = (sub["parse_method"] == "fallback").sum()
        # Correctness only over successfully-parsed rows
        ok_rows = sub[sub["parsed_label"] != "PARSE_FAIL"]
        acc = ok_rows["correct"].mean() if len(ok_rows) else float("nan")
        print(f"{pid:6} {n:>4} {pf:>11} {strict:>7} {syn:>5} {fb:>4} "
              f"{acc:>9.1%}" if len(ok_rows) else f"{pid:6} {n:>4} {pf:>11} {strict:>7} {syn:>5} {fb:>4}    n/a")
    print("=" * 70)

    # Per-class breakdown for each prompt
    for pid, sub in df.groupby("prompt_id"):
        print(f"\n[{pid}] predicted vs true (rows = true, cols = predicted):")
        confusion = pd.crosstab(sub["true_label"], sub["parsed_label"], dropna=False)
        print(confusion)

    # Latency summary
    print("\nLatency (seconds):")
    print(df.groupby("prompt_id")["latency_s"].describe()[["mean", "min", "max"]])

    # Save inspection CSV with the columns you want to review by hand
    out_cols = ["image_id", "true_label", "prompt_id", "parsed_label",
                "parse_method", "correct", "latency_s", "notes", "raw_output"]
    df[out_cols].to_csv("pilot_inspection.csv", index=False)
    print(f"\nWrote pilot_inspection.csv ({len(df)} rows)")
    print("\nNext steps:")
    print("  1. Open pilot_inspection.csv in a spreadsheet")
    print("  2. Sort by prompt_id, then read every raw_output")
    print("  3. Note in the 'notes' column anything weird:")
    print("     - did CoT (P4) actually reason, or skip straight to the answer?")
    print("     - did P1 obey the format, or wrap the answer in extra prose?")
    print("     - any classes the model never predicts? always predicts?")
    print("     - any obvious image-handling failure (refusal, blank, error)?")
    print("  4. Decide what to fix in prompts/parser before scaling.")


if __name__ == "__main__":
    main()
