"""
embedder.py — Load BiomedCLIP and embed images for RAG retrieval.

Uses microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 via open_clip.
Embeddings are 512-d, L2-normalized so cosine similarity == dot product.

The model is lazily loaded on first call. On Apple Silicon we run on MPS
when available, otherwise CPU.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512

_model = None
_preprocess = None
_device: str | None = None


def _ensure_loaded():
    global _model, _preprocess, _device
    if _model is not None:
        return
    from open_clip import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(MODEL_ID)
    model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model = model.to(_device)
    _preprocess = preprocess


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def embed_image(image) -> np.ndarray:
    """Return an L2-normalized 512-d float32 vector for one image.

    `image` may be a PIL Image, a path-like, or a string.
    """
    _ensure_loaded()
    x = _preprocess(_to_pil(image)).unsqueeze(0).to(_device)
    with torch.no_grad():
        v = _model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).to("cpu").numpy().astype(np.float32)


def embed_batch(images: Iterable) -> np.ndarray:
    """Embed a list of images. Returns (N, 512) float32, L2-normalized."""
    _ensure_loaded()
    xs = [_preprocess(_to_pil(img)) for img in images]
    batch = torch.stack(xs).to(_device)
    with torch.no_grad():
        v = _model.encode_image(batch)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.to("cpu").numpy().astype(np.float32)


# Self-test — mirrors parser.py's pattern.
if __name__ == "__main__":
    import sys
    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    manifest_path = REPO_ROOT / "results" / "pilot_manifest.csv"
    if not manifest_path.exists():
        print(f"FAIL  manifest missing: {manifest_path}")
        print("Run `python scripts/01_download_data.py` first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Loading BiomedCLIP and embedding {min(3, len(manifest))} pilot images...")

    # Single-image embed
    first = manifest.iloc[0]
    v = embed_image(first["image_path"])
    norm = float(np.linalg.norm(v))
    ok_shape = v.shape == (EMBEDDING_DIM,)
    ok_norm = abs(norm - 1.0) < 1e-4
    print(f"{'PASS' if ok_shape else 'FAIL'} embed_image shape: got {v.shape}")
    print(f"{'PASS' if ok_norm else 'FAIL'} embed_image norm:  got {norm:.6f}")

    # Batch embed (3 images)
    head = manifest.head(3)
    vs = embed_batch(head["image_path"].tolist())
    norms = np.linalg.norm(vs, axis=1)
    ok_batch_shape = vs.shape == (len(head), EMBEDDING_DIM)
    ok_batch_norm = np.allclose(norms, 1.0, atol=1e-4)
    print(f"{'PASS' if ok_batch_shape else 'FAIL'} embed_batch shape: got {vs.shape}")
    print(f"{'PASS' if ok_batch_norm else 'FAIL'} embed_batch norms: got {norms.round(4).tolist()}")

    # Cosine between first single and first batched should be ~1 (same image)
    cos = float(np.dot(v, vs[0]))
    ok_cos = abs(cos - 1.0) < 1e-4
    print(f"{'PASS' if ok_cos else 'FAIL'} same-image cosine: got {cos:.6f}")
