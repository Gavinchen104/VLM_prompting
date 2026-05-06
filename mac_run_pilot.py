"""
mac_run_pilot.py — Apple Silicon pilot runner using MLX.

Runs all 21 pilot images × the 2 pilot prompts (P1, P4) and writes
pilot_results.csv (same schema as the DCC version, so 04_inspect.py
works for both).

Run:
    source .venv/bin/activate
    python mac_run_pilot.py

Expected timing on M1/M2 16GB:
    ~10-30s per inference × 42 inferences ≈ 15-25 minutes total.
"""

import sys
import time
import csv
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Reuse loader + runner from the smoke test
from mac_smoke_test import load_model_mlx, run_one_mlx, MODEL_ID

# Reuse the prompt building blocks (same text as DCC pilots — only the wrapping differs)
from prompts import CLASSES_DESC, CLASSES_SHORT


# ---- Pilot prompts, in mlx-vlm-friendly form ----
# We reproduce P1 and P4 here (same wording) but as (system, instruction) tuples
# so the mlx-vlm runner can wrap them itself.

def p1_inputs():
    """Zero-shot, no role, direct."""
    system = None
    instruction = f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_DESC}

Respond in exactly this format and nothing else:
Final answer: <class_code>"""
    return system, instruction


def p4_inputs():
    """Zero-shot, dermatologist role, CoT."""
    system = ("You are an experienced dermatologist analyzing a dermoscopic image. "
              "Use the ABCD rule and standard dermoscopic criteria.")
    instruction = f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Reason step by step:
1. Asymmetry, Border, Color, Differential structures.
2. Top 2 candidate diagnoses with rationale.
3. Final commitment.

End your response with exactly this line:
Final answer: <class_code>"""
    return system, instruction


PILOT_PROMPTS = {
    "P1": p1_inputs,
    "P4": p4_inputs,
}


# Different prompts need different generation budgets — CoT is much longer
MAX_TOKENS_BY_PROMPT = {
    "P1": 60,    # short answer, plenty of room
    "P4": 500,   # CoT can be long
}


def main():
    manifest_path = Path("pilot_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    print(f"Pilot manifest: {len(manifest)} images, "
          f"{manifest['true_label'].nunique()} classes")
    print(f"Prompts: {list(PILOT_PROMPTS.keys())}")
    print(f"Total inferences: {len(manifest) * len(PILOT_PROMPTS)}")

    model, processor = load_model_mlx(MODEL_ID)

    out_path = "pilot_results.csv"
    fieldnames = ["image_id", "true_label", "prompt_id", "raw_output", "latency_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(manifest) * len(PILOT_PROMPTS)
        pbar = tqdm(total=total, desc="Inferences")

        for _, row in manifest.iterrows():
            image_path = row["image_path"]
            for prompt_id, prompt_fn in PILOT_PROMPTS.items():
                system, instruction = prompt_fn()
                t0 = time.time()
                try:
                    output = run_one_mlx(
                        model, processor, image_path, instruction,
                        system=system,
                        max_tokens=MAX_TOKENS_BY_PROMPT[prompt_id],
                    )
                except Exception:
                    output = f"<<EXCEPTION>>\n{traceback.format_exc()}"
                dt = time.time() - t0

                writer.writerow({
                    "image_id": row["image_id"],
                    "true_label": row["true_label"],
                    "prompt_id": prompt_id,
                    "raw_output": output,
                    "latency_s": f"{dt:.2f}",
                })
                f.flush()
                pbar.update(1)

        pbar.close()

    print(f"\nDone. Wrote {out_path}")
    print("Run `python 04_inspect.py` to parse and review.")


if __name__ == "__main__":
    main()
