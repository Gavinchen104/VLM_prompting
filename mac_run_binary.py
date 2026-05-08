"""
mac_run_binary.py — One-off binary (mel vs not_mel) experiment.

Same 21 images, same model, same prompt structure (P1-P4) — but reframed as a
2-class task to test whether the 7-class collapse we observed is driven by
task difficulty (a hard 7-way distinction) or by the model's lack of visual
engagement.

Hypothesis:
- If task difficulty is the bottleneck → binary accuracy should jump (target 65%+).
- If lack of visual engagement is the bottleneck → the model will still collapse,
  just to whichever single class is its dominant prior.

Run:
    python mac_run_binary.py
"""

import sys
import time
import csv
import re
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mac_smoke_test import load_model_mlx, run_one_mlx, MODEL_ID
from prompts import CONSTRAINTS_BLOCK, COT_STEPS_BLOCK, ROLE_SYSTEM


BINARY_CLASSES_DESC = """- mel:     melanoma
- not_mel: any non-melanoma lesion (nevus, basal cell carcinoma, actinic keratosis,
           benign keratosis, dermatofibroma, or vascular lesion)"""


OUTPUT_DIRECT_BIN = """Output exactly this format and nothing else:

Final answer: <mel or not_mel>"""

OUTPUT_COT_BIN = """Output exactly this format and nothing else:

Reasoning: <your 2-3 sentence analysis>
Final answer: <mel or not_mel>"""


def _build_instruction(cot: bool) -> str:
    header = f"""Classify this dermoscopic image into exactly one of these two classes:

{BINARY_CLASSES_DESC}

{CONSTRAINTS_BLOCK}"""
    if cot:
        return f"{header}\n\n{COT_STEPS_BLOCK}\n\n{OUTPUT_COT_BIN}"
    return f"{header}\n\n{OUTPUT_DIRECT_BIN}"


PROMPTS = {
    "P1": (None,        _build_instruction(cot=False)),  # direct, no role
    "P2": (None,        _build_instruction(cot=True)),   # CoT, no role
    "P3": (ROLE_SYSTEM, _build_instruction(cot=False)),  # direct, role
    "P4": (ROLE_SYSTEM, _build_instruction(cot=True)),   # CoT, role
}

MAX_TOKENS = {"P1": 60, "P2": 300, "P3": 60, "P4": 300}


# ---------- Binary parser ----------
NOT_MEL_OLD_CODES = {"nv", "bcc", "akiec", "bkl", "df", "vasc"}
NOT_MEL_OLD_NAMES = {"nevus", "carcinoma", "keratosis", "dermatofibroma",
                     "vascular", "hemangioma", "angioma", "lentigo"}


def parse_binary(output: str) -> tuple[str, str]:
    if not output:
        return "PARSE_FAIL", "fail"
    text = output.lower()
    m = re.search(r"final answer\s*[:\-]\s*([^\n]+)", text)
    if m:
        s = m.group(1).strip().rstrip(".,;:")
        # not_mel variants first (more specific)
        if re.match(r"^(not[_\s\-]*mel|non[_\s\-]*mel|notmel|nonmelanoma)", s):
            return "not_mel", "strict"
        # mel / melanoma — must not be the start of "melanocytic"
        first = s.split()[0] if s.split() else ""
        first = first.rstrip(".,;:")
        if first in ("mel", "melanoma"):
            return "mel", "strict"
        # If model regressed to old 7-class codes
        if first in NOT_MEL_OLD_CODES:
            return "not_mel", "strict"
        for w in NOT_MEL_OLD_NAMES:
            if first.startswith(w):
                return "not_mel", "strict"
    # Tail fallback
    tail = text[-300:]
    if re.search(r"\bnot[_\s\-]*mel|non[_\s\-]*mel|nonmelanoma", tail):
        return "not_mel", "fallback"
    if re.search(r"\bmelanoma\b", tail):
        return "mel", "fallback"
    return "PARSE_FAIL", "fail"


def main():
    manifest = pd.read_csv("pilot_manifest.csv")
    # binarize true labels: mel → 'mel', everything else → 'not_mel'
    manifest["true_binary"] = manifest["true_label"].apply(
        lambda x: "mel" if x == "mel" else "not_mel"
    )
    print(f"Manifest: 21 images — {(manifest['true_binary']=='mel').sum()} mel, "
          f"{(manifest['true_binary']=='not_mel').sum()} not_mel")
    print(f"Prompts: {list(PROMPTS.keys())}  |  Total inferences: {21*4}")

    model, processor = load_model_mlx(MODEL_ID)

    out_path = "pilot_binary.csv"
    fieldnames = ["model", "image_id", "true_label", "true_binary",
                  "prompt_id", "raw_output", "latency_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=21 * 4, desc="Inferences")
        for _, row in manifest.iterrows():
            for prompt_id, (system, instruction) in PROMPTS.items():
                t0 = time.time()
                try:
                    output = run_one_mlx(
                        model, processor, row["image_path"], instruction,
                        system=system, max_tokens=MAX_TOKENS[prompt_id],
                    )
                except Exception:
                    output = f"<<EXCEPTION>>\n{traceback.format_exc()}"
                dt = time.time() - t0
                writer.writerow({
                    "model": MODEL_ID,
                    "image_id": row["image_id"],
                    "true_label": row["true_label"],
                    "true_binary": row["true_binary"],
                    "prompt_id": prompt_id,
                    "raw_output": output,
                    "latency_s": f"{dt:.2f}",
                })
                f.flush()
                pbar.update(1)
        pbar.close()

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
