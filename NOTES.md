# Experiment Notes — Prompting Strategies for Medical VLMs

A starting-stage exploration of how prompt design interacts with model choice on
dermoscopic skin-lesion classification. Local Mac runs only (Apple Silicon,
mlx-vlm), 4-bit quantized models. Numbers below are from a small demo pilot and
should be read as **behavioral observations, not measurements**.

---

## 1. What we've done

### Pipeline (built end-to-end and validated locally)

1. **Data acquisition.** [01_download_data.py](01_download_data.py) pulls
   HAM10000 from Kaggle (~3GB, 10,015 images), reads its metadata, and builds
   a balanced pilot manifest (3 images per class, fixed seed 42, 21 images
   total).
2. **Pilot manifest (mini).** A 7-image one-per-class subset
   ([pilot_manifest_mini.csv](pilot_manifest_mini.csv)) was carved off for the
   model-comparison demo so two models could be compared end-to-end in time.
3. **Smoke test.** [mac_smoke_test.py](mac_smoke_test.py) runs one image
   through the pipeline as a gate — confirms model loads, format compliance,
   parser hits.
4. **Original pilot.** [mac_run_pilot.py](mac_run_pilot.py) — 21 images × 2
   prompts (P1, P4) on MedGemma 4B. 0 parse failures, 40/42 strict matches.
   First behavioral signals emerged here.
5. **Comparison demo.** [compare_pilot.py](compare_pilot.py) parameterized
   over `(model_id, output_csv)`; runs all 4 zero-shot prompts on the
   7-image mini manifest. Run twice — once per model.
6. **Comparison inspector.** [compare_inspect.py](compare_inspect.py)
   ingests every `pilot_*.csv` with the canonical schema, parses raw outputs,
   and emits per-`(model, prompt)` accuracy / parse-fail / class-diversity /
   latency, plus role × CoT interaction tables and class-distribution lines.
7. **Bug fixes resolved en route:**
   - pandas 3.0 dropped grouping columns from `groupby().apply()` →
     [01_download_data.py:79-83](01_download_data.py#L79-L83) refactored to
     `groupby().sample()`.
   - `DataFrame.applymap` removed in pandas 3.0 →
     [compare_inspect.py:117](compare_inspect.py#L117) uses `.map`.
   - Qwen2.5-VL processor pulls in `Qwen2VLVideoProcessor` which requires
     `torch + torchvision` — installed CPU-only, ~600MB.

### What was scoped out (deliberately)

- DCC / Slurm path (per pivot earlier in the session).
- Few-shot prompts (P5–P8) — implemented in
  [prompts.py:111-168](prompts.py#L111-L168) but require fixed demo-image
  selection logic; not wired into the demo runner.
- Full-scale runs (50 images/class × 8 prompts).
- Proper metrics module (macro-F1, bootstrap CIs).
- Statistical significance — sample size is far too small.

---

## 2. Methods — the prompt design

### Three orthogonal axes (2³ = 8 conditions)

| Axis        | Levels                                                         |
|-------------|----------------------------------------------------------------|
| **Shots**   | 0 (zero-shot) · F (few-shot, k=3 demos)                        |
| **Role**    | N (no role) · R ("You are an experienced dermatologist…")      |
| **Reasoning** | D (direct answer) · C (chain-of-thought, ABCD rule)          |

Naming convention: `P<shots><role><reasoning>` — so `P0ND` = P1, `PFRC` = P8.

### The four conditions used in the demo (zero-shot 2×2)

| ID | Shots | Role | Reasoning | What it tests |
|----|-------|------|-----------|---------------|
| P1 | 0     | none | direct    | Baseline — model's prior, no scaffolding |
| P2 | 0     | none | CoT       | Effect of reasoning **without** persona priming |
| P3 | 0     | role | direct    | Effect of persona **without** reasoning |
| P4 | 0     | role | CoT       | Both interventions together |

This 2×2 lets us read **main effects** (does CoT help on its own? does role
help on its own?) and the **interaction** (does role+CoT exceed the sum of
its parts?).

### Prompt anchoring

- **Vocabulary discipline.** Every prompt uses the same 7-class vocabulary
  (`mel, nv, bcc, akiec, bkl, df, vasc`) defined once in
  [prompts.py:14-22](prompts.py#L14-L22).
- **Output format anchor.** Every prompt ends with the literal line
  `Final answer: <class_code>`. The parser
  ([parser.py](parser.py)) is hard-coded to look for this first. If the
  model deviates, the parser falls back to synonyms ("melanoma" → mel) and
  finally to whole-output token search. `PARSE_FAIL` is a real outcome —
  never silently dropped.
- **Determinism.** All generation is `temperature=0.0` (greedy). Same
  input → same output across runs.
- **Token budgets.** P1/P3 (direct): `max_tokens=60`. P2/P4 (CoT):
  `max_tokens=500`. CoT outputs would otherwise truncate before the
  final-answer line.

---

## 3. Models compared

| Model | Variant used | Size | Type | Specialization |
|-------|-------------|------|------|----------------|
| **MedGemma 4B** | `mlx-community/medgemma-4b-it-4bit` | ~3GB | Vision-language, instruction-tuned | Medical-specialized (Google) |
| **Qwen2.5-VL-3B** | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | ~2GB | Vision-language, instruction-tuned | General-purpose (Alibaba) |

Both run via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) on Apple Silicon
Metal at 4-bit quantization. Same prompts, same images, same parser,
same temperature.

**Why this pairing.** A medical-specialized model vs. a general-purpose VLM
of similar size lets the comparison ask: *do prompting effects we observe on
medical models transfer to general-purpose models, or are they
model-specific?* The 4-bit quantization is a known degradation but acceptable
for a behavioral pilot.

**What we did NOT use:** the full-precision bf16 versions on cluster GPUs
(would require DCC), LLaVA-Med, BiomedCLIP, or any of the larger 7B+ VLMs.

---

## 4. Dataset — HAM10000

- **Source:** Kaggle, `kmader/skin-cancer-mnist-ham10000` — the public
  redistribution of the ISIC HAM10000 dermoscopic image set.
- **Size:** 10,015 images, 7 classes, heavy imbalance.
- **Image type:** Dermoscopic (high-resolution close-ups of skin lesions
  taken with a dermatoscope, not regular photos).
- **Why this benchmark:** Most-benchmarked dermoscopic classification target
  with public fine-tuned baselines, 7 classes give prompting room to
  differentiate, manageable scale.

### Class vocabulary and frequency (full dataset)

| Code  | Full name                                       | Frequency |
|-------|--------------------------------------------------|-----------|
| nv    | Melanocytic nevus                               | ~67% (dominant) |
| mel   | Melanoma                                        | ~11%      |
| bkl   | Benign keratosis                                | ~11%      |
| bcc   | Basal cell carcinoma                            | ~5%       |
| akiec | Actinic keratosis / intraepithelial carcinoma   | ~3%       |
| vasc  | Vascular lesion                                 | ~1%       |
| df    | Dermatofibroma                                  | ~1%       |

The imbalance is severe — a model that always says `nv` gets ~67% raw
accuracy and tells us nothing. **Macro-F1 and class diversity are the right
lenses, not raw accuracy.**

### Sampling for the demo

- **Pilot manifest** (21 images, used in the original pilot): 3 per class,
  random seed 42 — covers the imbalance evenly.
- **Mini manifest** (7 images, used in the model comparison): 1 per class,
  taken as the first row per class from the pilot manifest. Small enough to
  run two models in available time, balanced enough to surface
  per-class-bias patterns.

---

## 5. Results

All numbers below are from **4 prompts × 21 images × 2 models = 168
inferences** (3 images per class × 7 classes, fixed seed 42). An earlier
n=7 demo run is preserved in [n7_demo/](n7_demo/) for comparison; some
findings reproduced, others changed at scale.

### 5.1 Headline table (n=21)

| Model | Prompt | Role | Reasoning | Acc | Parse-fails | # classes predicted | Avg latency (s) |
|-------|--------|------|-----------|-----|-------------|---------------------|-----------------|
| medgemma-4b   | P1 | no-role | direct | 14.3% | 0 | **1** | 6.75  |
| medgemma-4b   | P2 | no-role | CoT    | 19.0% | 0 | 4 | 10.11 |
| medgemma-4b   | P3 | role    | direct | 14.3% | 0 | 3 | 6.58  |
| medgemma-4b   | P4 | role    | CoT    | **23.8%** | 0 | 4 | 8.85 |
| qwen2.5-vl-3b | P1 | no-role | direct | 19.0% | 0 | 4 | 3.43  |
| qwen2.5-vl-3b | P2 | no-role | CoT    | 23.8% | 0 | 4 | 10.97 |
| qwen2.5-vl-3b | P3 | role    | direct | **28.6%** | 0 | 4 | 3.25 |
| qwen2.5-vl-3b | P4 | role    | CoT    | 19.0% | 0 | 3 | 7.65  |

### 5.2 Role × Reasoning interaction (accuracy, n=21)

**MedGemma 4B** — both interventions help, additive

|         | direct | CoT   |
|---------|--------|-------|
| no-role | 14.3%  | 19.0% |
| role    | 14.3%  | **23.8%** |

**Qwen2.5-VL-3B** — role helps, CoT *hurts* on top of role

|         | direct | CoT   |
|---------|--------|-------|
| no-role | 19.0%  | 23.8% |
| role    | **28.6%**  | 19.0% |

### 5.3 Class diversity (# unique classes predicted out of 7, n=21)

|              | P1 | P2 | P3 | P4 |
|--------------|----|----|----|----|
| medgemma-4b  | 1  | 4  | 3  | 4  |
| qwen2.5-vl-3b| 4  | 4  | 4  | 3  |

### 5.4 Predicted-class distributions (per cell, n=21)

| Cell                | Predicted distribution                                      |
|---------------------|-------------------------------------------------------------|
| medgemma / P1       | `nv:21` (full collapse to majority prior)                   |
| medgemma / P2       | `mel:17, nv:2, akiec:1, bcc:1`                              |
| medgemma / P3       | `mel:18, nv:2, bkl:1`                                       |
| medgemma / P4       | `mel:13, nv:6, bkl:1, vasc:1`                               |
| qwen2.5-vl / P1     | `mel:17, bcc:2, bkl:1, df:1`                                |
| qwen2.5-vl / P2     | `mel:14, vasc:3, bcc:2, nv:2`                               |
| qwen2.5-vl / P3     | `bcc:14, bkl:3, mel:3, nv:1`                                |
| qwen2.5-vl / P4     | `bcc:13, mel:7, vasc:1`                                     |

### 5.5 What changed n=7 → n=21 (validation of patterns)

| Pattern (from n=7)                        | At n=21                                | Verdict |
|-------------------------------------------|----------------------------------------|---------|
| MedGemma P1 collapses to `nv`             | `nv:21/21` — full collapse              | **Confirmed, stronger** |
| Qwen role causes `bcc`-bias               | P3: `bcc:14/21`, P4: `bcc:13/21`        | **Confirmed, reproducible** |
| MedGemma CoT unlocks class diversity      | Diversity 1→4, but mostly `mel:17`      | **Partial — diversity ≠ correctness** |
| Qwen P1 is "diverse"                      | Actually `mel:17/21` — n=7 was small-sample artifact | **Reversed** |
| Qwen CoT hurts class diversity            | Mixed; what's robust is **CoT hurts when role is on** (P4 19% vs P3 28.6%) | **Refined** |
| MedGemma P3 mel-bias                      | `mel:18/21` — strongest result          | **Confirmed, stronger** |
| Qwen has best cell at P3 (role-direct)    | 28.6% — best across all 8 cells          | **Confirmed** |

### 5.6 Behavioral findings (the science) — n=21

1. **Direct prompting reveals priors, not vision.** MedGemma P1 collapsed to
   `nv:21/21` — full majority-class collapse. Qwen P1 looks "diverse" by class
   count (4) but is actually `mel:17/21` — a different bias, not meaningful
   diversity. The 1-line outputs (e.g., `"Final answer: nv"`) suggest neither
   model spent any reasoning capacity on the image.

2. **CoT effect is model-dependent and asymmetric.**
   - MedGemma: CoT modestly improves accuracy (14.3 → 19.0% no-role; 14.3 →
     23.8% role). Class diversity expands from 1→4. Best cell is **P4 (role
     + CoT) at 23.8%**.
   - Qwen: CoT helps when no role is present (19.0 → 23.8%) but **hurts when
     role is present** (28.6 → 19.0%). Adding ABCD-rule scaffolding on top
     of the dermatologist persona pushes the model toward over-confident
     `bcc` predictions.
   - The interaction is **opposite-signed** between the two models. CoT is
     not a universal win; its sign depends on whether you've already primed
     the model with a role.

3. **Role effect creates model-specific class-bias.**
   - MedGemma + role → `mel`-bias: P3 `mel:18/21`, P4 `mel:13/21`.
   - Qwen + role → `bcc`-bias: P3 `bcc:14/21`, P4 `bcc:13/21`.
   - "You are an experienced dermatologist" carries class-priors of its own,
     and each model interprets it through a different lens. **The persona is
     not class-neutral.**

4. **The single best result across all 8 cells is Qwen P3 (role + direct,
   28.6%)** — not the most ornate prompt. Adding CoT on top of role made it
   worse. This argues against the "more scaffolding = better" intuition.

5. **Reasoning quality is independent of accuracy.** Qwen's P4 outputs walk
   through ABCD criteria (asymmetry, border, color, structures), explicitly
   compare top-2 candidates, and commit with stated rationale — coherent
   dermatologic reasoning prose. Yet they're often wrong (often `bcc`).
   MedGemma's reasoning had **factual errors about its own class
   abbreviations**: e.g., calling `nv` "non-vascular lesion" (it's
   *melanocytic nevus*) and `mel` "melanocytic nevus" (it's *melanoma*) in
   the same paragraph.

6. **Format compliance is excellent across both models at scale.** 0 PARSE_FAIL
   out of 168 outputs. The strict format anchor (`Final answer: <code>`) is
   robust. A handful of outputs use synonyms that the parser handles
   (e.g., `"Final answer: bcc (basal cell carcinoma)"` → matched).

7. **Latency:** Qwen-3B is ~2× faster than MedGemma-4B (direct: 3.3s vs
   6.7s; CoT: 9.3s vs 9.5s). MedGemma's CoT advantage shrinks (CoT is the
   dominant cost regardless of model).

### 5.7 Caveats and what the numbers do NOT say

- **N=21 images per cell** (3 per class × 7 classes). Better than n=7, but
  still small — confidence intervals on a single accuracy cell are ~±20
  percentage points. Class-bias *patterns* (e.g., Qwen `bcc:14/21`) are
  more robust than the cell accuracy numbers themselves.
- **Both models are 4-bit quantized.** Reasoning quality and class accuracy
  are degraded relative to full precision. The `bcc`-bias on Qwen P4 may not
  reproduce at bf16. Verifying this requires DCC.
- **One image per class.** Within-class variance is invisible. The "MedGemma
  P3 → mel-bias" finding rests on whether 5/7 of the *specific* images we
  picked happen to be ambiguous toward melanoma — different selection might
  shift the bias.
- **No ground-truth-style evaluation of reasoning.** When we say "Qwen's
  reasoning is more coherent," that's a qualitative read of a handful of
  raw outputs in [compare_inspection.csv](compare_inspection.csv), not a
  judged or scored measurement.
- **Single random seed.** Image selection used `random_state=42` once.
  Stability across seeds is unmeasured.

---

## 6. Future work

In rough priority order; mapped to the broader research thesis in
[CLAUDE.md](CLAUDE.md).

### Near-term (extensions of this codebase)

1. **Few-shot conditions (P5–P8).** Prompt scaffolding already exists in
   [prompts.py:111-168](prompts.py#L111-L168). Need to:
   - Pick 3 fixed demo images by hand (one easy mel, one easy nv, one
     tricky case) — held out from the test manifest.
   - Wire them into a few-shot variant of `compare_pilot.py`.
   - Re-run the 2×2×2 factorial → all 8 conditions per model.
2. **Scale up to a real test set.** Replace the 7-image mini manifest with a
   balanced ~350-image set (50/class). Same scripts, same parser. Write
   `05_run_full.py` paralleling [03_run_pilot.py](03_run_pilot.py)'s
   incremental-CSV pattern so partial runs survive interruptions.
3. **Proper metrics module.** Replace the back-of-envelope accuracy in
   [04_inspect.py](04_inspect.py) with macro-F1, balanced accuracy, per-class
   F1, parse-fail rate, confusion matrices, and bootstrap CIs.
4. **Add a third model** for cross-architecture validation. Candidates:
   LLaVA-Med 8B (medical-specialized, larger), Qwen2.5-VL-7B (general,
   larger), or a non-VLM baseline like a fine-tuned EfficientNet from the
   ISIC leaderboard for context.
5. **Move to full precision.** Run the same factorial on Duke DCC at bf16
   and compare against the 4-bit Mac numbers — quantifies how much of the
   `bcc`-bias / `nv`-collapse is quantization vs. real model behavior.
6. **Visualize.** Per-axis main-effect plots (CoT vs. direct, role vs.
   no-role), interaction plots, per-class F1 bars, gap-to-fine-tuned-baseline
   chart.

### Medium-term (other prompt-engineering levers)

7. **Soft prompts.** CoOp / BiomedCoOp continuous learnable prompts on
   BiomedCLIP. The Bhayana et al. anchor paper doesn't cover these; this is
   gap 3 in the broader thesis.
8. **Visual prompting.** Bounding-box markers, MedSAM contour overlays,
   color highlights — interventions that change the *image* the model sees
   rather than the text. Gap 2 in the thesis.
9. **Prompt vs. fine-tuning at varying data scales.** Compare the best
   prompted result against LoRA-fine-tuned MedGemma at 100, 1k, 10k training
   examples. Gap 4 in the thesis. This is the "value-add" framing.

### Longer-term (mechanistic / cross-domain)

10. **Mechanistic analysis.** Pull BiomedCLIP attention maps for failure
    cases — does the model attend to the lesion? Are CoT failures attention
    failures or generation failures? Gap 1 in the thesis.
11. **Domain transfer.** Repeat the prompting study on radiology (CXR) and
    pathology (histology) — does the model-prompt interaction we observed
    here generalize, or is it dermoscopy-specific? Gap 5.

---

## 7. Files of record

```
NOTES.md                  ← this file
CLAUDE.md                 ← project context (the durable doc)
README.md                 ← one-page overview

01_download_data.py       ← Kaggle download + balanced manifest
prompts.py                ← all 8 prompt conditions
parser.py                 ← free-text → class code, with synonyms
04_inspect.py             ← per-prompt summary for original pilot

mac_smoke_test.py         ← single-image gate
mac_run_pilot.py          ← original pilot (21 × P1,P4) on MedGemma

compare_pilot.py          ← parameterized runner: model_id → CSV
compare_inspect.py        ← multi-model inspector

pilot_manifest.csv        ← 21 rows (3 per class, seed 42)
pilot_manifest_mini.csv   ← 7 rows (1 per class)

pilot_results.csv         ← original pilot results (MedGemma P1+P4 × 21)
pilot_inspection.csv      ← parsed inspection of the original pilot
pilot_medgemma.csv        ← demo: MedGemma × P1-P4 × 7
pilot_qwen.csv            ← demo: Qwen2.5-VL × P1-P4 × 7
compare_inspection.csv    ← combined parsed table for the demo
```

---

## 8. One-paragraph elevator summary

> We built a clean prompt-comparison pipeline for medical VLMs and ran a
> two-model study on dermoscopic skin-lesion classification (HAM10000,
> 7 classes, 21 test images balanced across classes, 4 zero-shot prompt
> variants spanning role × CoT — 168 inferences total). The key behavioral
> finding: **prompts do not have universal effects across models.** On a
> medical-specialized model (MedGemma 4B), direct prompting collapses to the
> majority class (`nv` 21/21); CoT and role each modestly help, with role +
> CoT giving the model's best cell at 23.8%. On a general-purpose model of
> similar size (Qwen2.5-VL 3B), the **best cell is role + direct (28.6%)** —
> adding CoT on top of role *hurts* by pushing the model to over-commit to
> `bcc` (14/21). The persona "experienced dermatologist" carries
> class-priors that are model-specific: MedGemma drifts toward `mel`, Qwen
> drifts toward `bcc`. Both models maintain perfect output-format compliance
> (0 parse failures across 168 outputs), but reasoning content differs
> sharply: Qwen produces coherent ABCD walkthroughs while MedGemma sometimes
> confuses its own class abbreviations. The headline takeaway:
> **scaffolding is not monotonic — more prompt structure can hurt.** The
> pipeline is in place to scale this further (50 images/class × 8 conditions
> × multiple models) and compare against fine-tuned baselines on DCC.
