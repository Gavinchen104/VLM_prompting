"""
mac_smoke_test.py — Apple Silicon smoke test using MLX.

Uses the 4-bit MLX port of MedGemma (mlx-community/medgemma-4b-it-4bit, ~3GB),
which fits comfortably in 16GB unified memory.

Run:
    source .venv/bin/activate
    python mac_smoke_test.py

What success looks like:
    - Model loads in ~30s (first run downloads ~3GB; subsequent runs use cache)
    - One inference takes ~10-30s on M1/M2
    - smoke_test_output.txt is written with non-empty model output
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Use the 4-bit MLX quantization — community port, no gating
MODEL_ID = "mlx-community/medgemma-4b-it-4bit"

# Chat-template settings
MAX_TOKENS = 300


def load_model_mlx(model_id):
    """Load a model + processor via mlx-vlm."""
    print(f"Loading {model_id} ...")
    t0 = time.time()
    from mlx_vlm import load
    model, processor = load(model_id)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def build_messages_for_mlx(image_path: str, instruction: str, system: str | None = None):
    """
    mlx-vlm expects a messages list very similar to transformers, but the image
    is passed separately to generate(). The 'image' content blocks are placeholders.
    """
    messages = []
    if system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system}],
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ],
    })
    return messages


def run_one_mlx(model, processor, image_path: str, instruction: str,
                system: str | None = None, max_tokens: int = MAX_TOKENS):
    """
    One inference. Returns the generated text (output only, no prompt).
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = build_messages_for_mlx(image_path, instruction, system)
    prompt = apply_chat_template(processor, model.config, messages, num_images=1)

    out = generate(
        model, processor, prompt,
        image=[image_path],
        max_tokens=max_tokens,
        temperature=0.0,    # deterministic
        verbose=False,
    )
    # mlx-vlm's generate may return a string, or an object with .text — handle both
    if hasattr(out, "text"):
        return out.text
    return str(out)


def main():
    # 1. Load the manifest
    manifest_path = Path("pilot_manifest.csv")
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run 01_download_data.py first.")
        sys.exit(1)
    manifest = pd.read_csv(manifest_path)
    if len(manifest) == 0:
        print("ERROR: manifest is empty.")
        sys.exit(1)

    row = manifest.iloc[0]
    image_path = row["image_path"]
    print(f"Test image: {row['image_id']}  (true label: {row['true_label']})")
    print(f"Image path: {image_path}")

    # 2. Load model
    model, processor = load_model_mlx(MODEL_ID)

    # 3. Build the same P1 instruction we'll use later
    from prompts import CLASSES_DESC
    instruction = f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_DESC}

Respond in exactly this format and nothing else:
Final answer: <class_code>"""

    # 4. Generate
    print("\n--- Generating ---")
    t0 = time.time()
    output = run_one_mlx(model, processor, image_path, instruction)
    dt = time.time() - t0
    print(f"Generated in {dt:.1f}s")
    print(f"\n--- Raw output ---\n{output}\n--- End ---")

    # 5. Parse
    from parser import parse
    parsed, method = parse(output)
    print(f"Parsed: {parsed}  (method: {method})")
    print(f"True:   {row['true_label']}  (match: {parsed == row['true_label']})")

    # 6. Save
    with open("smoke_test_output.txt", "w") as f:
        f.write(f"backend: mlx-vlm\n")
        f.write(f"model: {MODEL_ID}\n")
        f.write(f"image_id: {row['image_id']}\n")
        f.write(f"true_label: {row['true_label']}\n")
        f.write(f"latency_s: {dt:.2f}\n")
        f.write(f"parsed: {parsed} ({method})\n")
        f.write(f"\n--- raw output ---\n{output}\n")

    print("\nSaved smoke_test_output.txt")
    print("\n*** Smoke test passed. You can now run mac_run_pilot.py ***")


if __name__ == "__main__":
    main()
