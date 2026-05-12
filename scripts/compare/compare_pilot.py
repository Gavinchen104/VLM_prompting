"""
compare_pilot.py — Run all 4 zero-shot prompts (P1-P4) on a mini manifest
for a single model, write results to a model-tagged CSV under results/.

Usage:
    python scripts/compare/compare_pilot.py <model_id> <output_name> [--manifest <path>]

The output_name is interpreted relative to results/ unless it's an absolute path
or already contains a directory component.

Examples:
    python scripts/compare/compare_pilot.py mlx-community/medgemma-4b-it-4bit pilot_medgemma.csv
    python scripts/compare/compare_pilot.py mlx-community/Qwen2.5-VL-3B-Instruct-4bit pilot_qwen.csv

Designed to be run in separate Python invocations per model so memory
is freed between runs.
"""

import sys
import time
import csv
import argparse
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Repo-anchored paths + put src/ on the import path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from prompts import CLASSES_DESC, CLASSES_SHORT


# ---- 4 zero-shot prompts as (system, instruction) tuples ----

def p1():
    """zero-shot, no role, direct."""
    return None, f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_DESC}

Respond in exactly this format and nothing else:
Final answer: <class_code>"""


def p2():
    """zero-shot, no role, CoT."""
    return None, f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_SHORT}.

Reason step by step before answering:
1. Describe what you see: asymmetry, border irregularity, color variation,
   diameter cues, and any dermoscopic structures (pigment network, globules,
   streaks, vessels).
2. List the top 2 candidate classes and why each fits or doesn't.
3. Commit to one class.

End your response with exactly this line:
Final answer: <class_code>"""


def p3():
    """zero-shot, role, direct."""
    sys_msg = "You are an experienced dermatologist analyzing a dermoscopic image."
    return sys_msg, f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Respond in exactly this format and nothing else:
Final answer: <class_code>"""


def p4():
    """zero-shot, role, CoT."""
    sys_msg = ("You are an experienced dermatologist analyzing a dermoscopic image. "
               "Use the ABCD rule and standard dermoscopic criteria.")
    return sys_msg, f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Reason step by step:
1. Asymmetry, Border, Color, Differential structures.
2. Top 2 candidate diagnoses with rationale.
3. Final commitment.

End your response with exactly this line:
Final answer: <class_code>"""


PROMPTS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4}

# Direct prompts get tight token budgets; CoT needs room
MAX_TOKENS = {"P1": 60, "P2": 500, "P3": 60, "P4": 500}


def load_model_mlx(model_id):
    print(f"Loading {model_id} ...")
    t0 = time.time()
    from mlx_vlm import load
    model, processor = load(model_id)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def run_one_mlx(model, processor, image_path, instruction, system, max_tokens):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = []
    if system:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system}]})
    messages.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    })
    prompt = apply_chat_template(processor, model.config, messages, num_images=1)
    out = generate(model, processor, prompt,
                   image=[image_path],
                   max_tokens=max_tokens,
                   temperature=0.0,
                   verbose=False)
    return out.text if hasattr(out, "text") else str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("output_csv", help="Bare filename (goes under results/) or full path")
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "pilot_manifest_mini.csv"))
    args = ap.parse_args()

    # If output_csv is a bare filename, put it under results/.
    out_path = Path(args.output_csv)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = RESULTS_DIR / out_path

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {args.manifest} ({len(manifest)} images)")
    print(f"Model:    {args.model_id}")
    print(f"Prompts:  {list(PROMPTS.keys())}")
    total = len(manifest) * len(PROMPTS)
    print(f"Total inferences: {total}")

    model, processor = load_model_mlx(args.model_id)

    fieldnames = ["model", "image_id", "true_label", "prompt_id", "raw_output", "latency_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=total, desc="Inferences")
        for _, row in manifest.iterrows():
            for pid, pfn in PROMPTS.items():
                system, instr = pfn()
                t0 = time.time()
                try:
                    output = run_one_mlx(
                        model, processor, row["image_path"],
                        instr, system, MAX_TOKENS[pid],
                    )
                except Exception:
                    output = f"<<EXCEPTION>>\n{traceback.format_exc()}"
                dt = time.time() - t0

                writer.writerow({
                    "model": args.model_id,
                    "image_id": row["image_id"],
                    "true_label": row["true_label"],
                    "prompt_id": pid,
                    "raw_output": output,
                    "latency_s": f"{dt:.2f}",
                })
                f.flush()
                pbar.update(1)
        pbar.close()

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
