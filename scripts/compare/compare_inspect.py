"""
compare_inspect.py — Side-by-side comparison of multiple model results
across the 4 zero-shot prompt conditions (P1-P4).

Reads:
    results/pilot_medgemma.csv, results/pilot_qwen.csv, ... (any results/pilot_*.csv
    with the canonical schema: model, image_id, true_label, prompt_id, raw_output, latency_s)

Writes:
    results/compare_inspection.csv — combined parsed table
    Console: per-(model, prompt) summary, class-diversity per condition,
             interaction-style 2x2 view (role × CoT)
"""

import sys
from pathlib import Path
import pandas as pd

# Repo-anchored paths + put src/ on the import path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_DIR = REPO_ROOT / "results"

from parser import parse


PROMPT_AXES = {
    "P1": ("no-role", "direct"),
    "P2": ("no-role", "CoT"),
    "P3": ("role",    "direct"),
    "P4": ("role",    "CoT"),
}


REQUIRED_COLS = {"model", "image_id", "true_label", "prompt_id", "raw_output", "latency_s"}


def load_all():
    candidates = sorted(RESULTS_DIR.glob("pilot_*.csv"))
    files, dfs = [], []
    for f in candidates:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not REQUIRED_COLS.issubset(df.columns):
            continue
        df["raw_output"] = df["raw_output"].fillna("").astype(str)
        files.append(str(f))
        dfs.append(df)
    if not dfs:
        print("No model-result CSVs found (need columns: "
              + ", ".join(sorted(REQUIRED_COLS)) + ")")
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files: {files}")
    return df


def short_model_name(model_id: str) -> str:
    """Trim model_id for display: 'mlx-community/medgemma-4b-it-4bit' -> 'medgemma-4b'."""
    name = model_id.split("/")[-1].lower()
    for stem, short in [
        ("medgemma-4b", "medgemma-4b"),
        ("qwen2.5-vl-3b", "qwen2.5-vl-3b"),
        ("qwen2-vl-2b", "qwen2-vl-2b"),
        ("llava", "llava"),
        ("smolvlm", "smolvlm"),
    ]:
        if stem in name:
            return short
    return name[:24]


def main():
    df = load_all()
    df["model_short"] = df["model"].apply(short_model_name)

    parsed = df["raw_output"].apply(parse)
    df["parsed_label"] = parsed.apply(lambda x: x[0])
    df["parse_method"] = parsed.apply(lambda x: x[1])
    df["correct"] = ((df["parsed_label"] == df["true_label"])
                     & (df["parsed_label"] != "PARSE_FAIL")).astype(int)

    # ---- (1) Per-(model, prompt) headline ----
    print("\n" + "=" * 88)
    print("Headline — accuracy / parse-fail / class-diversity per (model × prompt)")
    print("=" * 88)
    print(f"{'model':14} {'prompt':6} {'role':8} {'reason':6} "
          f"{'n':>3} {'acc':>6} {'pfail':>6} {'#classes':>8} {'lat_s':>7}")
    print("-" * 88)
    rows = []
    for (m, p), sub in df.groupby(["model_short", "prompt_id"]):
        n = len(sub)
        acc = sub["correct"].mean()
        pfail = (sub["parsed_label"] == "PARSE_FAIL").sum()
        ncls = sub.loc[sub["parsed_label"] != "PARSE_FAIL", "parsed_label"].nunique()
        lat = sub["latency_s"].astype(float).mean()
        role, reason = PROMPT_AXES.get(p, ("?", "?"))
        rows.append((m, p, role, reason, n, acc, pfail, ncls, lat))
        print(f"{m:14} {p:6} {role:8} {reason:6} {n:>3} {acc:>6.1%} {pfail:>6} "
              f"{ncls:>8} {lat:>7.2f}")
    print("=" * 88)

    # ---- (2) Class diversity heatmap (#unique predicted classes per condition) ----
    print("\nClass diversity (# unique classes predicted out of 7 possible):")
    div = (df[df.parsed_label != "PARSE_FAIL"]
             .groupby(["model_short", "prompt_id"])["parsed_label"]
             .nunique().unstack().fillna(0).astype(int))
    print(div)

    # ---- (3) 2×2 main effects: role × CoT, accuracy per cell, per model ----
    print("\nRole × Reasoning interaction (accuracy):")
    for m, sub in df.groupby("model_short"):
        print(f"\n  [{m}]")
        sub = sub.copy()
        sub["role"]   = sub["prompt_id"].map(lambda p: PROMPT_AXES[p][0])
        sub["reason"] = sub["prompt_id"].map(lambda p: PROMPT_AXES[p][1])
        pivot = (sub.groupby(["role", "reason"])["correct"]
                    .mean().unstack().reindex(index=["no-role", "role"],
                                              columns=["direct", "CoT"]))
        print(pivot.map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a"))

    # ---- (4) Per-class confusion: did CoT unlock new classes? ----
    print("\nPredicted-class distribution per (model, prompt):")
    for (m, p), sub in df.groupby(["model_short", "prompt_id"]):
        counts = sub["parsed_label"].value_counts().to_dict()
        line = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  [{m} / {p}]  {line}")

    # ---- Save flat inspection ----
    out_cols = ["model_short", "image_id", "true_label", "prompt_id",
                "parsed_label", "parse_method", "correct", "latency_s",
                "raw_output"]
    out_path = RESULTS_DIR / "compare_inspection.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
