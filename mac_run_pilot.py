"""
mac_run_pilot.py — Apple Silicon pilot runner using MLX.

Runs the pilot manifest × the configured prompts and writes a results CSV
(same schema as the DCC version, so 04_inspect.py works for both).

Usage:
    source .venv/bin/activate
    python mac_run_pilot.py                                  # default: full 21 images, P1-P4
    python mac_run_pilot.py pilot_manifest_mini.csv          # mini manifest
    python mac_run_pilot.py pilot_manifest_mini.csv out.csv  # custom output

Expected timing on M1/M2 16GB:
    direct prompts ~3-5s, CoT prompts ~10-30s per inference.
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

# Reuse the shared prompt blocks so this file stays in sync with prompts.py.
# mlx-vlm uses a different message shape than transformers, so we re-wrap here,
# but the actual instruction text is composed from the same building blocks.
from prompts import _build_instruction, ROLE_SYSTEM


# ---- Pilot prompts in mlx-vlm-friendly form: (system, instruction) tuples ----

def p1_inputs():
    """Zero-shot, no role, direct."""
    return None, _build_instruction(cot=False)


def p2_inputs():
    """Zero-shot, no role, CoT."""
    return None, _build_instruction(cot=True)


def p3_inputs():
    """Zero-shot, role, direct."""
    return ROLE_SYSTEM, _build_instruction(cot=False)


def p4_inputs():
    """Zero-shot, role, CoT."""
    return ROLE_SYSTEM, _build_instruction(cot=True)


PILOT_PROMPTS = {
    "P1": p1_inputs,
    "P2": p2_inputs,
    "P3": p3_inputs,
    "P4": p4_inputs,
}


# Different prompts need different generation budgets — CoT is much longer.
# The new CoT format asks for a 2-3 sentence reasoning so we can be tighter
# than the old open-ended ABCD prompt (was 500).
MAX_TOKENS_BY_PROMPT = {
    "P1": 60,
    "P2": 300,
    "P3": 60,
    "P4": 300,
}


def main():
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pilot_manifest.csv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pilot_results.csv"

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Manifest: {manifest_path} — {len(manifest)} images, "
          f"{manifest['true_label'].nunique()} classes")
    print(f"Prompts: {list(PILOT_PROMPTS.keys())}")
    print(f"Total inferences: {len(manifest) * len(PILOT_PROMPTS)}")
    print(f"Output: {out_path}")

    model, processor = load_model_mlx(MODEL_ID)

    fieldnames = ["model", "image_id", "true_label", "prompt_id", "raw_output", "latency_s"]
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
                    "model": MODEL_ID,
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
