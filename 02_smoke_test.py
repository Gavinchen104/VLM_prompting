"""
02_smoke_test.py — Prove the model works end-to-end on ONE image.

Run interactively on a GPU node:
    srun --partition=gpu-common --gres=gpu:1 --time=01:00:00 --mem=24G --pty bash
    conda activate medvlm
    cd $WORK
    python 02_smoke_test.py

If this prints sensible-looking output and saves smoke_test_output.txt, you're good.
DO NOT PROCEED to the pilot run until this works.
"""

import os
import sys
import json
import time
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------- Config ----------

# Prefer 1.5 if you have access; fall back to v1.
MODEL_CANDIDATES = [
    "google/medgemma-1.5-4b-it",
    "google/medgemma-4b-it",
]

# Set to True if your GPU has < 12GB VRAM (e.g. 8GB).
# bf16 takes ~9GB; 4-bit quantization takes ~3-4GB at a small accuracy cost.
USE_4BIT = False


def pick_model():
    """Try the candidates in order, return the first one that's accessible."""
    from huggingface_hub import HfApi, GatedRepoError
    api = HfApi()
    for model_id in MODEL_CANDIDATES:
        try:
            api.model_info(model_id)
            print(f"[OK] Will use model: {model_id}")
            return model_id
        except Exception as e:
            print(f"[--] Cannot access {model_id}: {type(e).__name__}")
    print("ERROR: no MedGemma model accessible. Did you accept the license?")
    sys.exit(1)


def load_model(model_id):
    print(f"Loading {model_id} (this takes ~30s) ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return processor, model


def run_one(processor, model, image, messages, max_new_tokens=300):
    """Apply chat template, generate, decode the *new* tokens only."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # deterministic — important for reproducibility
        )

    new_tokens = out[0][input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text


def main():
    # 1. Load manifest, take first image
    manifest = pd.read_csv("pilot_manifest.csv")
    if len(manifest) == 0:
        print("ERROR: pilot_manifest.csv is empty. Run 01_download_data.py first.")
        sys.exit(1)
    row = manifest.iloc[0]
    print(f"Using image: {row['image_id']}  (true label: {row['true_label']})")
    image = Image.open(row["image_path"]).convert("RGB")
    print(f"Image size: {image.size}")

    # 2. Pick + load model
    model_id = pick_model()
    processor, model = load_model(model_id)

    # 3. Build a minimal prompt — one of our pilot prompts
    from prompts import p1_zero_norole_direct
    messages = p1_zero_norole_direct(image)

    # 4. Generate
    print("\n--- Generating ---")
    t0 = time.time()
    output = run_one(processor, model, image, messages)
    dt = time.time() - t0
    print(f"Generated in {dt:.1f}s")
    print(f"Output:\n{output}")

    # 5. Parse
    from parser import parse
    parsed, method = parse(output)
    print(f"\nParsed: {parsed}  (method: {method})")
    print(f"True:   {row['true_label']}  (match: {parsed == row['true_label']})")

    # 6. Save artifacts
    with open("smoke_test_output.txt", "w") as f:
        f.write(f"model: {model_id}\n")
        f.write(f"image_id: {row['image_id']}\n")
        f.write(f"true_label: {row['true_label']}\n")
        f.write(f"latency_s: {dt:.2f}\n")
        f.write(f"parsed: {parsed} ({method})\n")
        f.write(f"\n--- raw output ---\n{output}\n")
    print("\nSaved smoke_test_output.txt")
    print("\n*** Smoke test PASSED. You can now run the pilot. ***")


if __name__ == "__main__":
    main()
