# RAG for Medical VLM Classification — Implementation Design

This document describes how to add **retrieval-augmented generation (RAG)** to
the existing prompting pipeline for HAM10000 dermoscopy classification with
MedGemma 4B.

It assumes the current pipeline (see [CLAUDE.md](CLAUDE.md)) and
extends it without breaking the 7-class factorial of P1–P8.

---

## 1. Goal

Replace the fixed few-shot demo selection in P5–P8 with **dynamic, visually-similar
example retrieval** — and optionally inject **retrieved medical knowledge** (textbook
descriptions, dermoscopic feature definitions) into the prompt.

The hypothesis: at inference time, conditioning on the K nearest training images
(by visual embedding) plus their labels gives the VLM more relevant context than
3 fixed demos picked once. This is "retrieval-augmented in-context learning."

---

## 2. Why this is worth doing for this project

| Connection to project | How RAG helps |
|---|---|
| **Gap 2** (visual prompting — CLAUDE.md) | Retrieved images are visual context, not text |
| **Gap 4** (prompt vs fine-tuning) | RAG gives prompting access to training-set knowledge without fine-tuning |
| **P5–P8 few-shot conditions** | Already structured for demo injection; RAG just changes how demos are picked |
| **Binary-mel finding** (May 2026 pilot) | RAG should help most where the model already engages — i.e., binary, not 7-class |

**Honest caveat:** based on the recent pilot, the 7-class task is dominated by the
model's priors collapsing under task difficulty, not by lack of demonstrations.
RAG is more likely to move the needle on **binary** (or 3-class) framing first.
Build the infrastructure in a way that's task-framing-agnostic.

---

## 3. Architecture

```
                 ┌──────────────────┐
                 │  Query image     │
                 └─────────┬────────┘
                           │
                ┌──────────▼──────────┐
                │  Image embedder     │   (BiomedCLIP or OpenCLIP)
                │  → 512-d vector     │
                └──────────┬──────────┘
                           │
              ┌────────────▼──────────────┐
              │  Vector store (FAISS)     │   (built offline from train split)
              │  ANN search → top-K IDs   │
              └────────────┬──────────────┘
                           │
                ┌──────────▼─────────────┐
                │  Retrieval policy      │   class-balance / dedupe / filter
                └──────────┬─────────────┘
                           │
        ┌──────────────────▼─────────────────────┐
        │  Prompt builder                         │
        │  inject K (image, label) pairs as few-  │
        │  shot demos + optional retrieved text   │
        └──────────────────┬─────────────────────┘
                           │
                ┌──────────▼─────────────┐
                │  MedGemma generates    │
                └──────────┬─────────────┘
                           │
                ┌──────────▼─────────────┐
                │  Existing parser       │
                └────────────────────────┘
```

The retrieval module slots in **between** the manifest and the prompt builder —
the rest of the pipeline (model, parser, inspect, metrics) is untouched.

---

## 4. Components

### 4.1 Image embedder

**Recommended:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
- Trained on biomedical image-text pairs → far better dermatology embeddings than OpenCLIP
- 512-d output, fast on CPU/MPS for our scale
- Same model is on the project's roadmap for Gap 1 (mechanistic analysis), so this is shared infrastructure

**Fallback:** `openai/clip-vit-base-patch32` if BiomedCLIP install is painful.

```python
# embedder.py
from open_clip import create_model_from_pretrained, get_tokenizer
model, preprocess = create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
def embed_image(pil_img) -> np.ndarray:
    with torch.no_grad():
        x = preprocess(pil_img).unsqueeze(0)
        return model.encode_image(x).cpu().numpy().squeeze()
```

### 4.2 Vector store

**Recommended:** [FAISS](https://github.com/facebookresearch/faiss) (CPU). Reasons:
- Self-contained, no server, fits in repo
- 10K HAM10000 images at 512-d = ~20 MB index — trivial
- Top-K + filter (e.g., "only retrieve from class X") is built-in
- One pip install, no infra

**File layout:**
```
data/rag/
  index_biomedclip.faiss        # the index itself
  index_meta.parquet            # row-aligned: image_id, true_label, embedding_norm
  embedder_config.json          # what model produced this index — version pin
```

### 4.3 Retrieval policy

The naïve "top-K nearest" is a starting point but not the right answer for a
class-imbalanced dataset. Three policies to implement and ablate:

| Policy | Logic | When to prefer |
|---|---|---|
| `topk_pure` | K nearest in embedding space, regardless of label | Baseline — just visual similarity |
| `topk_class_balanced` | Round-robin K/C nearest from each class | Forces label diversity, prevents nv-only retrieval (HAM10000 is 67% nv) |
| `topk_diverse` | Greedy MMR-style: max similarity to query, min similarity to already-picked | Avoids 3 near-duplicate retrievals from same patient |

Default: `topk_class_balanced` with K=3 to match the existing P5–P8 structure.

### 4.4 Prompt builder integration

The current `_few_shot_preamble(demo_images, demo_labels)` already accepts a list
of demos. A new function `retrieve_demos(query_image, k=3, policy=...)` returns
exactly the same shape. No prompt-template changes needed.

**Optionally:** also inject retrieved **textual knowledge** for each candidate
class — this is the "T" in RAG. Source: a small `class_textbook.json` mapping
class codes to ~50-word ABCD/dermoscopic descriptions:

```json
{
  "mel": "Melanoma typically presents with asymmetry, irregular borders, ...",
  "nv":  "Benign melanocytic nevi show symmetry, regular pigment networks, ...",
  ...
}
```

This is text-only RAG and complements image RAG. Worth ablating separately.

---

## 5. Phased implementation plan

### Phase 0 — prerequisites (1 day)
- Decide train/test split. **Critical:** retrieval index must be built only from
  the *training* split, never from the eval set (otherwise it's leakage).
- Add `data/rag/` to `.gitignore` (artifacts are large)
- Pin dependencies: `faiss-cpu`, `open_clip_torch`, `torch` (already present on DCC)

### Phase 1 — index build (1 day)
File: `06_build_rag_index.py`
- Load training-split image_ids from `pilot_manifest_full.csv` (TODO: build this)
- Embed each image with BiomedCLIP
- Build FAISS `IndexFlatIP` (cosine similarity if embeddings are L2-normalized)
- Persist index + metadata
- Sanity check: query 5 random training images, eyeball nearest neighbors

### Phase 2 — retrieval module (1 day)
File: `rag.py`
- `embed_image(pil_img) → np.ndarray`
- `retrieve(query_image, k=3, policy="topk_class_balanced") → [(image_path, label), ...]`
- Self-tests at bottom (mirror `parser.py`'s pattern)

### Phase 3 — wire into prompts (½ day)
- New prompt functions `p5r_*` through `p8r_*` (R for "retrieval-augmented") in
  [prompts.py](prompts.py)
- Or add a `retrieve=True` flag to existing P5–P8 — both keep the factorial intact
- **Do not modify P1–P4** — they remain the no-retrieval baseline

### Phase 4 — runner + experiment (1 day)
- New runner `mac_run_rag.py` that mirrors [mac_run_pilot.py](mac_run_pilot.py)
  but takes a retrieval policy argument
- Output schema gets two new columns: `retrieved_ids` (semicolon-joined),
  `retrieved_labels` (semicolon-joined) — for post-hoc analysis of retrieval quality
- First run: 21-image mini pilot, P5R only, 3 retrieval policies × 1 prompt = 3 runs

### Phase 5 — full evaluation (2 days)
- Build the 350-image balanced eval set (as planned in `05_run_full.py`)
- Run all 8 conditions × {no-retrieval, retrieval} = 16 conditions
- Report: per-condition macro-F1, retrieval-quality metrics (precision-at-K,
  class-balance of retrieved demos), latency overhead

---

## 6. File structure (additions only)

```
06_build_rag_index.py    # offline: build the FAISS index from training split
rag.py                   # runtime: embed + retrieve
mac_run_rag.py           # runner: same as mac_run_pilot.py but with retrieval
prompts.py               # add p5r_*..p8r_* OR add retrieve= flag
data/
  rag/
    index_biomedclip.faiss
    index_meta.parquet
    embedder_config.json
  class_textbook.json    # optional: textual knowledge for text-RAG
pilot_manifest_train.csv # the retrieval pool — never overlaps with eval
```

---

## 7. Dependencies to add

| Package | Why | Concern |
|---|---|---|
| `faiss-cpu` | Vector index | Adds ~30 MB; pip install only — no infra |
| `open_clip_torch` | BiomedCLIP loader | Pulls torch + timm — heavy, but probably already on DCC |
| `torch` | Inference | Already in dependencies for DCC path |

**For Mac:** these all install cleanly via pip. CLIP encode on a 224×224 image
runs in ~30ms on M-series CPU; 10K images takes ~5 minutes to index.

**For DCC:** request a GPU partition for index build (10× faster); inference
retrieval is CPU-fine.

---

## 8. Evaluation methodology

For RAG to be a meaningful contribution to the project's thesis, the comparison
must isolate the retrieval effect from confounds:

| Comparison | Tells you |
|---|---|
| P5 (fixed demos) vs P5R (retrieved demos) | Does retrieval-as-demo-selection help? |
| P5R top-K-pure vs P5R class-balanced vs P5R MMR | Which retrieval policy matters? |
| P1 (no demos) vs P5R (retrieved demos) | What does few-shot-via-retrieval add over zero-shot? |
| P5R K=1 vs K=3 vs K=5 | What's the right K? |
| P5R-images-only vs P5R-images+text | Does text knowledge add over image demos? |

**Headline metric stays macro-F1**, but track two new ones for diagnosis:
- **Retrieval precision-at-K**: of the K retrieved, how many share the true label?
- **Retrieval class diversity**: Shannon entropy of retrieved-label distribution

If retrieval precision-at-K is low but final accuracy goes up anyway, the model
is using *visual similarity* of retrievals, not their labels — that's an
interesting finding on its own.

---

## 9. Risks and tradeoffs

| Risk | Mitigation |
|---|---|
| **Retrieval leakage** (index contains eval images) | Strict train/eval split discipline; assert in `rag.py` that no eval image_id appears in the index metadata |
| **Visual similarity ≠ class similarity** in dermoscopy (skin tone, lighting, framing dominate over diagnostic features) | Class-balanced retrieval policy; ablate with text-RAG which doesn't have this problem |
| **MedGemma 4B context window** — 3 demo images + query + reasoning may exceed it on CoT prompts | Test early; cap K=3 initially; consider downsampling demo images to 224×224 |
| **Retrieval latency adds to inference time** | Pre-compute query embedding once; FAISS top-K on 10K is microseconds. Real cost is the model processing more image tokens — measure |
| **The 7-class task may still collapse** even with RAG | Run binary RAG first to see if retrieval helps where the model already engages |
| **BiomedCLIP install pain on Apple Silicon** | Have OpenCLIP fallback ready |

---

## 10. What this connects to in the wider project

RAG bridges three of the five gaps the project is positioned against:

- **Gap 2 (visual prompting):** retrieved images ARE visual context — this is a
  text-prompt-augmentation that's strictly more visual than the baseline
- **Gap 4 (prompt vs fine-tuning):** RAG gives prompting access to training-set
  knowledge without weight updates — the cleanest "prompting can use training
  data without fine-tuning" experiment
- **Gap 1 (mechanistic):** the BiomedCLIP embedder used for retrieval is the
  same model used for the planned attention-map analysis — shared infrastructure

It does **not** address Gap 3 (soft prompts) or Gap 5 (domain-specific
frameworks).

---

## 11. Decision points before starting

Before implementing, confirm with the project lead:

1. **Train/eval split for the retrieval pool.** The current pipeline uses the
   same 21 images for everything. The full ~350-image eval set hasn't been built.
   Need to decide: hold out eval first, then build retrieval index from the rest.
2. **Whether to start RAG on the 7-class task or the binary task.** Binary is
   more likely to show signal given the recent pilot finding. But 7-class is
   the project's primary frame.
3. **MedGemma vs Qwen as RAG substrate.** Recent pilot showed Qwen 4-bit slightly
   outperforms MedGemma 4-bit on this task; RAG might amplify that.
4. **Whether to commit to BiomedCLIP** (medically grounded but adds
   infrastructure) or start with OpenCLIP (faster to ship, weaker embeddings).

---

## 12. Suggested first runnable artifact

The minimum-viable RAG demonstration is:
1. Build a FAISS index over the 21 pilot images (yes, leakage — this is just to
   prove the pipeline). 1 hour.
2. For each query image, retrieve K=3 nearest *excluding self*, inject as
   few-shot demos in P5R, run on the same 21 images.
3. Compare macro-F1 to the existing P1/P4 baseline.

This is throwaway code with deliberate leakage, but it answers "does the
plumbing work end-to-end?" before you invest in a clean train/eval split. Same
spirit as the original smoke test.
