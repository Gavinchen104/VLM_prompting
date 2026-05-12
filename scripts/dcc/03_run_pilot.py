"""
03_run_pilot.py — Run 21 images × 2 prompts (P1, P4) and save raw outputs.

This produces results/pilot_results.csv with columns:
  image_id, true_label, prompt_id, raw_output, latency_s

We do NOT parse here. Parsing happens in 04_inspect.py so we can iterate on
the parser without re-running the model.

Designed to be submitted via sbatch (see run_pilot.sh).
"""

import os
import sys
import time
import csv
import traceback
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# Repo-anchored paths + put src/ AND this script's dir on the import path.
# The latter is needed for import_module("02_smoke_test") below — numeric prefix
# means it's not a normal-importable name.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from prompts import PILOT_PROMPTS

# Reuse the smoke test's model loader to keep behavior identical
from importlib import import_module
smoke = import_module("02_smoke_test")  # numeric prefix → can't normal-import

OUTPUT_CSV = RESULTS_DIR / "pilot_results.csv"


def main():
    manifest = pd.read_csv(RESULTS_DIR / "pilot_manifest.csv")
    print(f"Pilot manifest: {len(manifest)} images, "
          f"{manifest['true_label'].nunique()} classes")

    model_id = smoke.pick_model()
    processor, model = smoke.load_model(model_id)

    # Open CSV and write header
    fieldnames = ["image_id", "true_label", "prompt_id", "raw_output", "latency_s"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(manifest) * len(PILOT_PROMPTS)
        pbar = tqdm(total=total, desc="Inferences")

        for _, row in manifest.iterrows():
            try:
                image = Image.open(row["image_path"]).convert("RGB")
            except Exception as e:
                print(f"Failed to open {row['image_path']}: {e}")
                continue

            for prompt_id, prompt_fn in PILOT_PROMPTS.items():
                messages = prompt_fn(image)
                t0 = time.time()
                try:
                    output = smoke.run_one(
                        processor, model, image, messages,
                        max_new_tokens=400,    # CoT needs more room than direct
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
                f.flush()   # so partial results survive a crash
                pbar.update(1)

        pbar.close()

    print(f"\nDone. Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
