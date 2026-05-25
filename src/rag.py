"""
rag.py — Retrieve K visually-similar pool images for a query.

Loads the FAISS index built by scripts/06_build_rag_index.py and exposes a
single function `retrieve()` that returns (image_path, true_label, cosine)
triples sorted by relevance under a chosen policy.

Three policies (per RAG_DESIGN.md §4.3):
  - "topk_pure":           K nearest in embedding space, regardless of label.
  - "topk_class_balanced": prefer label diversity — fill first with one demo per
                           class in similarity order, then top up by similarity.
  - "topk_diverse":        greedy MMR — high cosine to query, low cosine to
                           already-picked. Avoids near-duplicates.

Self-tests at the bottom mirror parser.py's pattern.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from embedder import embed_image, EMBEDDING_DIM

_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "rag"
_index = None
_meta: pd.DataFrame | None = None


def _ensure_loaded():
    global _index, _meta
    if _index is not None:
        return
    import faiss
    idx_path = _INDEX_DIR / "index_biomedclip.faiss"
    meta_path = _INDEX_DIR / "index_meta.csv"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"RAG index missing under {_INDEX_DIR}. "
            "Run `python scripts/06_build_rag_index.py` first."
        )
    _index = faiss.read_index(str(idx_path))
    _meta = pd.read_csv(meta_path)
    if _index.ntotal != len(_meta):
        raise RuntimeError(
            f"Index/meta length mismatch: ntotal={_index.ntotal} vs meta={len(_meta)}"
        )


def _candidates(query_vec: np.ndarray, n: int):
    """Top-n FAISS candidates. Returns (sims, idxs) as 1-D arrays length n."""
    n = min(n, _index.ntotal)
    sims, idxs = _index.search(query_vec.reshape(1, -1).astype(np.float32), n)
    return sims[0], idxs[0]


def _result(idxs, sims):
    out = []
    for i, s in zip(idxs, sims):
        row = _meta.iloc[int(i)]
        out.append((row["image_path"], row["true_label"], float(s)))
    return out


def _is_excluded(i, exclude_id):
    return exclude_id is not None and _meta.iloc[int(i)]["image_id"] == exclude_id


def _topk_pure(query_vec, k, exclude_id):
    sims, idxs = _candidates(query_vec, k + (1 if exclude_id else 0))
    picks_i, picks_s = [], []
    for i, s in zip(idxs, sims):
        if _is_excluded(i, exclude_id):
            continue
        picks_i.append(i); picks_s.append(s)
        if len(picks_i) >= k:
            break
    return picks_i, picks_s


def _topk_class_balanced(query_vec, k, exclude_id):
    # Wide candidate pool, then pick K with class diversity preference.
    pool = min(max(k * 4, 16), _index.ntotal)
    sims, idxs = _candidates(query_vec, pool)
    picks_i, picks_s, seen = [], [], set()
    # Pass 1: one demo per class, in similarity order.
    for i, s in zip(idxs, sims):
        if _is_excluded(i, exclude_id):
            continue
        lab = _meta.iloc[int(i)]["true_label"]
        if lab in seen:
            continue
        seen.add(lab)
        picks_i.append(i); picks_s.append(s)
        if len(picks_i) >= k:
            return picks_i, picks_s
    # Pass 2: fill remaining slots by raw similarity (labels may repeat).
    chosen = set(int(j) for j in picks_i)
    for i, s in zip(idxs, sims):
        if int(i) in chosen or _is_excluded(i, exclude_id):
            continue
        picks_i.append(i); picks_s.append(s); chosen.add(int(i))
        if len(picks_i) >= k:
            break
    return picks_i, picks_s


def _topk_diverse(query_vec, k, exclude_id, lambda_=0.5):
    # MMR: each step picks argmax over candidates of
    #   lambda * sim(q, d) - (1 - lambda) * max sim(d, d') for d' already picked.
    # Vectors are L2-normalized, so dot product == cosine.
    pool = min(max(k * 4, 16), _index.ntotal)
    sims, idxs = _candidates(query_vec, pool)
    cand_vecs = np.stack([_index.reconstruct(int(i)) for i in idxs])  # (pool, dim)
    allowed = np.array(
        [not _is_excluded(i, exclude_id) for i in idxs], dtype=bool
    )

    picked_j: list[int] = []  # candidate-pool indices
    while len(picked_j) < k:
        best_j, best_score = -1, -np.inf
        for j in range(len(idxs)):
            if not allowed[j] or j in picked_j:
                continue
            if picked_j:
                div = max(float(cand_vecs[j] @ cand_vecs[p]) for p in picked_j)
            else:
                div = 0.0
            score = lambda_ * float(sims[j]) - (1.0 - lambda_) * div
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            break
        picked_j.append(best_j)

    return [idxs[j] for j in picked_j], [sims[j] for j in picked_j]


Policy = Literal["topk_pure", "topk_class_balanced", "topk_diverse"]
_POLICIES = {
    "topk_pure": _topk_pure,
    "topk_class_balanced": _topk_class_balanced,
    "topk_diverse": _topk_diverse,
}


def retrieve(query_image, k: int = 3,
             policy: Policy = "topk_class_balanced",
             exclude_id: str | None = None):
    """Return K retrieved demos as a list of (image_path, true_label, cosine) tuples.

    `query_image` may be a PIL Image, path-like, or string.
    `exclude_id`, if given, filters retrieved items whose image_id matches —
    use for leave-one-out queries when the query image is in the index.
    """
    _ensure_loaded()
    if policy not in _POLICIES:
        raise ValueError(f"Unknown policy: {policy}. Options: {list(_POLICIES)}")
    q = embed_image(query_image)
    idxs, sims = _POLICIES[policy](q, k, exclude_id)
    return _result(idxs, sims)


# Self-test
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(REPO_ROOT / "results" / "pilot_manifest.csv")
    row = manifest.iloc[0]
    q_id = row["image_id"]
    print(f"Query: {q_id}  ({row['true_label']})\n")

    def _show(label, hits):
        print(f"--- {label} ---")
        for path, lab, sim in hits:
            stem = Path(path).stem
            marker = "  <- self" if stem == q_id else ""
            print(f"  {stem}  {lab:8}  cos={sim:.4f}{marker}")

    # 1. Pure top-3 — query is in the index, top hit should be self at ~1.0.
    hits = retrieve(row["image_path"], k=3, policy="topk_pure")
    _show("topk_pure (no exclude)", hits)
    assert hits[0][2] > 0.99, f"PASS condition broken: top hit cos={hits[0][2]:.4f}"
    print("PASS  self-hit at top with cosine>0.99\n")

    # 2. Pure top-3 with exclude_id — self should be filtered, K=3 returned.
    hits = retrieve(row["image_path"], k=3, policy="topk_pure", exclude_id=q_id)
    _show("topk_pure (exclude self)", hits)
    assert all(Path(p).stem != q_id for p, _, _ in hits), "self leaked"
    assert len(hits) == 3, f"expected 3 hits, got {len(hits)}"
    print("PASS  self filtered, K=3 returned\n")

    # 3. Class-balanced should prefer label diversity.
    hits_b = retrieve(row["image_path"], k=3, policy="topk_class_balanced", exclude_id=q_id)
    _show("topk_class_balanced (exclude self)", hits_b)
    assert len(hits_b) == 3
    classes_b = {lab for _, lab, _ in hits_b}
    print(f"PASS  K=3, {len(classes_b)} unique classes\n")

    # 4. MMR / diverse.
    hits_d = retrieve(row["image_path"], k=3, policy="topk_diverse", exclude_id=q_id)
    _show("topk_diverse (exclude self)", hits_d)
    assert len(hits_d) == 3
    print("PASS  K=3 returned\n")

    # 5. Class-balanced is at least as diverse as pure on this query.
    hits_p = retrieve(row["image_path"], k=3, policy="topk_pure", exclude_id=q_id)
    div_p = len({lab for _, lab, _ in hits_p})
    div_b = len({lab for _, lab, _ in hits_b})
    print(f"diversity: pure={div_p} classes, class_balanced={div_b} classes")
    assert div_b >= div_p, "class_balanced should match or exceed pure on diversity"
    print("PASS  class_balanced >= pure on class diversity")
