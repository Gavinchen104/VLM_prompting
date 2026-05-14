"""
06_build_rag_index.py — Build a FAISS index over images in a manifest.

Phase 1 of RAG_DESIGN.md. For now we index the existing 21 pilot images
directly — the "first runnable artifact" from §12 — which is deliberately
leaky but proves the plumbing end-to-end. Once the train/eval split is
decided, swap the manifest for pilot_manifest_train.csv (Phase 1.5).

Output (under data/rag/, gitignored):
    index_biomedclip.faiss   — FAISS IndexFlatIP (cosine via dot, vectors are L2-normalized)
    index_meta.csv           — row-aligned: image_id, true_label, image_path
    embedder_config.json     — model id + dim + source manifest (version pin)

Usage:
    python scripts/06_build_rag_index.py
    python scripts/06_build_rag_index.py results/some_other_manifest.csv
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Repo-anchored paths + put src/ on the import path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_DIR = REPO_ROOT / "results"
RAG_DIR = REPO_ROOT / "data" / "rag"

from embedder import embed_batch, MODEL_ID, EMBEDDING_DIM


def build(manifest_path: Path, out_dir: Path) -> None:
    import faiss  # local import keeps the top of the file light

    df = pd.read_csv(manifest_path)
    print(f"Manifest: {manifest_path}  ({len(df)} rows, "
          f"{df['true_label'].nunique()} classes)")

    print("Embedding images...")
    vectors = embed_batch(df["image_path"].tolist())
    print(f"  embeddings: shape={vectors.shape}, dtype={vectors.dtype}")
    # Sanity: every vector should be unit-norm out of the embedder.
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"Non-unit embedding(s) — min={norms.min():.4f}, max={norms.max():.4f}"

    # IndexFlatIP + L2-normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    print(f"FAISS: ntotal={index.ntotal}, dim={EMBEDDING_DIM}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index_biomedclip.faiss"
    meta_path = out_dir / "index_meta.csv"
    config_path = out_dir / "embedder_config.json"

    faiss.write_index(index, str(index_path))
    df[["image_id", "true_label", "image_path"]].to_csv(meta_path, index=False)
    config_path.write_text(json.dumps({
        "model_id": MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "n_vectors": int(index.ntotal),
        "source_manifest": str(manifest_path),
    }, indent=2))

    print(f"\nWrote: {index_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {config_path}")

    # Smoke check: top-3 self-retrieval for the first image.
    # (Deliberately leaky — the query is in the index. We expect the top hit
    # to be itself with cosine ~1.0, and the rest to be genuinely similar.)
    print("\nSmoke check — top-3 neighbours of the first manifest image:")
    q = vectors[0:1]
    sims, idxs = index.search(q, 3)
    head = df.iloc[idxs[0]][["image_id", "true_label"]].reset_index(drop=True)
    head["cosine"] = [round(float(s), 4) for s in sims[0]]
    print(head.to_string(index=False))


if __name__ == "__main__":
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "pilot_manifest.csv"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found. Run scripts/01_download_data.py first.")
        sys.exit(1)
    build(manifest, RAG_DIR)
