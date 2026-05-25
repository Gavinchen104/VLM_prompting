"""
run_rag.py — RAG-augmented pilot runner on Apple Silicon (mlx-vlm).

For each query image in the manifest:
  1. Retrieve K=3 demo (image, label) pairs from the FAISS index built by
     scripts/06_build_rag_index.py, under a chosen policy.
  2. Build a multi-image few-shot mlx-vlm message: demos + their labels,
     followed by the query image and the standard classification instruction.
  3. Run the model and persist the raw output plus retrieval provenance.

Phase 4 of RAG_DESIGN.md. The retrieval pool is the same 21-image pilot
manifest the index was built from, so we use exclude_id for leave-one-out
queries (so a query never retrieves itself as one of its own demos).

Output CSV columns (extends the pilot schema):
    model, image_id, true_label, prompt_id, raw_output, latency_s,
    policy, retrieved_ids, retrieved_labels

Usage:
    python scripts/mac/run_rag.py                          # default: P5R, class_balanced, K=3
    python scripts/mac/run_rag.py --dry-run                # show retrievals, don't load model
    python scripts/mac/run_rag.py --policy topk_diverse
    python scripts/mac/run_rag.py --k 2
    python scripts/mac/run_rag.py --manifest results/pilot_manifest_mini.csv
"""

from __future__ import annotations
import sys
import time
import csv
import argparse
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Repo-anchored paths + put src/ AND this script's dir on the import path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Reuse loader + the shared prompt blocks. Retrieval is a separate concern.
from smoke_test import load_model_mlx, MODEL_ID  # noqa: E402
from prompts import _build_instruction  # noqa: E402
import rag  # noqa: E402


# ---- mlx-vlm few-shot message construction ---------------------------------
# mlx-vlm takes images via generate(image=[paths]); messages just carry
# {"type": "image"} placeholders that the chat template expands to image
# tokens. We don't reuse prompts._few_shot_preamble because that one stuffs
# PIL images inline (transformers shape), which mlx-vlm doesn't want.

def _fewshot_messages(instruction: str, demo_labels: list[str]) -> list[dict]:
    """Build mlx-vlm-shape messages: K demos with labels, then the query.

    Caller passes the same number of image paths to generate() in the order
    [demo_1, demo_2, ..., demo_K, query].
    """
    content = [{"type": "text",
                "text": "Here are reference examples. Each image is followed by its correct label."}]
    for lab in demo_labels:
        content.append({"type": "image"})
        content.append({"type": "text", "text": f"Final answer: {lab}\n---"})
    content.append({"type": "text",
                    "text": f"Now classify the next image using the same format.\n\n{instruction}"})
    content.append({"type": "image"})   # the query image
    return [{"role": "user", "content": content}]


def _run_mlx_multi(model, processor, prompt_text, image_paths, max_tokens):
    """Run mlx-vlm with multiple images. Returns generated text."""
    from mlx_vlm import generate
    out = generate(
        model, processor, prompt_text,
        image=list(image_paths),
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return out.text if hasattr(out, "text") else str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "pilot_manifest.csv"))
    ap.add_argument("--output", default=None,
                    help="Output CSV (default: results/pilot_rag_<policy>.csv)")
    ap.add_argument("--policy", default="topk_class_balanced",
                    choices=["topk_pure", "topk_class_balanced", "topk_diverse"])
    ap.add_argument("--k", type=int, default=3, help="Number of demos to retrieve")
    ap.add_argument("--prompt-id", default="P5R",
                    help="Tag written into output CSV; the prompt itself is "
                         "the zero-shot, no-role, direct template with retrieved demos.")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run retrieval only, don't load the VLM or generate.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Manifest: {manifest_path}  ({len(manifest)} rows)")
    print(f"Policy:   {args.policy}   K={args.k}")
    print(f"Prompt:   {args.prompt_id}  (P5R = zero-shot, no role, direct + RAG demos)")
    print(f"Dry run:  {args.dry_run}\n")

    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"pilot_rag_{args.policy}.csv"
    )

    instruction = _build_instruction(cot=False)
    fieldnames = ["model", "image_id", "true_label", "prompt_id",
                  "raw_output", "latency_s", "policy",
                  "retrieved_ids", "retrieved_labels"]

    model = processor = None
    if not args.dry_run:
        model, processor = load_model_mlx(MODEL_ID)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=len(manifest), desc="Inferences")
        for _, row in manifest.iterrows():
            # 1. Retrieve K demos, excluding self (the query is in the index).
            hits = rag.retrieve(
                row["image_path"], k=args.k, policy=args.policy,
                exclude_id=row["image_id"],
            )
            demo_paths = [h[0] for h in hits]
            demo_labels = [h[1] for h in hits]
            demo_ids = [Path(p).stem for p in demo_paths]

            if args.dry_run:
                writer.writerow({
                    "model": MODEL_ID,
                    "image_id": row["image_id"],
                    "true_label": row["true_label"],
                    "prompt_id": args.prompt_id,
                    "raw_output": "<<DRY_RUN>>",
                    "latency_s": "0.00",
                    "policy": args.policy,
                    "retrieved_ids": ";".join(demo_ids),
                    "retrieved_labels": ";".join(demo_labels),
                })
                f.flush(); pbar.update(1)
                continue

            # 2. Build messages + apply chat template + generate.
            from mlx_vlm.prompt_utils import apply_chat_template
            messages = _fewshot_messages(instruction, demo_labels)
            prompt_text = apply_chat_template(
                processor, model.config, messages, num_images=len(demo_paths) + 1
            )
            image_paths = demo_paths + [row["image_path"]]

            t0 = time.time()
            try:
                output = _run_mlx_multi(model, processor, prompt_text, image_paths,
                                        args.max_tokens)
            except Exception:
                output = f"<<EXCEPTION>>\n{traceback.format_exc()}"
            dt = time.time() - t0

            writer.writerow({
                "model": MODEL_ID,
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "prompt_id": args.prompt_id,
                "raw_output": output,
                "latency_s": f"{dt:.2f}",
                "policy": args.policy,
                "retrieved_ids": ";".join(demo_ids),
                "retrieved_labels": ";".join(demo_labels),
            })
            f.flush(); pbar.update(1)

        pbar.close()

    print(f"\nDone. Wrote {out_path}")
    if not args.dry_run:
        print("Run `python scripts/04_inspect.py` after pointing it at the new CSV "
              "(or use compare_inspect for a side-by-side with the baseline).")


if __name__ == "__main__":
    main()
