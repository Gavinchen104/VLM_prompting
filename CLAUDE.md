# CLAUDE.md — Project Context for Claude Code

This file orients Claude Code to the project. Read it fully before making changes.

---

## What this project is

A research study on **prompting strategies for medical Vision-Language Models (VLMs)**.
Starting-stage exploration: we run a small, controlled experiment first, then scale.

**Anchor paper:** Bhayana et al., arXiv:2511.11898 — "DSPy for Medical VQA." They use
automated prompt optimization (BootstrapFS, MIPROv2, SIMBA, GEPA) on medical VQA but
treat the model as a black box, only optimize text prompts, and don't compare against
fine-tuning.

**Our long-term thesis:** Address five gaps in that paper:
1. Black-box → mechanistic analysis (BiomedCLIP attention maps)
2. Text-only → visual prompting (markers, MedSAM contours, overlays)
3. No soft prompts → CoOp/BiomedCoOp continuous prompts
4. No fine-tuning comparison → prompt vs. LoRA at varying data scales
5. General-purpose → domain-specific frameworks (radiology vs. pathology)

**Current scope (this repo):** Just gap 4's setup — a clean prompt-comparison pipeline
on dermoscopic classification with HAM10000, MedGemma 4B as primary model.
Mechanistic / visual / soft-prompt experiments come later, in separate sub-projects.

---

## Current status

We have a working **pilot pipeline**: 21 images × 2 prompts (P1 zero-shot direct, P4
role + CoT). The pipeline runs locally on Apple Silicon via `mlx-vlm` (4-bit quantized
MedGemma) and on Duke DCC via `transformers` (bf16). Same prompts, same parser, same
inspection step on both.

**What's done:**
- Pilot manifest + downloader (Kaggle HAM10000)
- All 8 prompt conditions written (P1–P8); only P1 and P4 used in the pilot
- Output parser with synonym handling and PARSE_FAIL tracking
- Mac runner (mlx-vlm) and DCC runner (transformers + Slurm)
- Inspection script (parses raw outputs → labels → console summary + CSV for hand review)

**What's NOT done yet:**
- The pilot has not been *run* end-to-end yet (or has been run but results not
  reviewed — check `results/pilot_inspection.csv` to know which)
- Few-shot prompts (P5–P8) are written but not used yet — they need 3 demo images
  selected and wired in
- Full-experiment runner (`scripts/05_run_full.py`) for all 8 prompts × balanced
  350-image test set — to be written after the pilot is validated
- Metrics computation (macro-F1, balanced accuracy, per-class F1, confusion matrices) —
  basic version is in `scripts/04_inspect.py`; a proper metrics module comes after
  the full run
- Second model for cross-architecture validation (LLaVA-Med 8B or Qwen3-VL 8B)

---

## Two paths: Mac (local) vs DCC (cluster)

The same pilot can run on either, with different backends:

| Aspect              | Mac (Apple Silicon)               | Duke DCC                              |
|---------------------|-----------------------------------|---------------------------------------|
| Backend             | `mlx-vlm`                         | `transformers`                        |
| Model               | `mlx-community/medgemma-4b-it-4bit` (~3GB) | `google/medgemma-4b-it` (bf16, ~9GB) |
| Quality             | Slightly degraded (4-bit)         | Full precision                        |
| Speed               | ~10–30s/inference                 | ~1–3s/inference                       |
| Memory              | ~6GB peak                         | ~12GB peak                            |
| Smoke test script   | `scripts/mac/smoke_test.py`       | `scripts/dcc/02_smoke_test.py`        |
| Pilot runner        | `scripts/mac/run_pilot.py`        | `scripts/dcc/03_run_pilot.py` (via `scripts/dcc/run_pilot.sh`) |
| Setup guide         | `docs/setup_mac.md`               | `docs/setup_dcc.md`                   |

The **Mac path is for proving the pipeline works** (fast iteration, easy debugging).
The **DCC path is for real measurements** (full precision, scale to many images).
**Inspection (`scripts/04_inspect.py`) is identical for both.**

Default to the Mac path during development unless the user says otherwise.

---

## File map

```
README.md                          — one-page overview
CLAUDE.md                          — this file (Claude Code context)
LICENSE
.gitignore

docs/
  setup_mac.md                     — Mac setup: brew, venv, mlx-vlm, kaggle/HF auth
  setup_dcc.md                     — DCC setup: modules, conda, Slurm, scratch space

src/                               — importable modules; on sys.path via each
                                     script's bootstrap block
  prompts.py                       — All 8 prompt conditions (chat-template format)
  parser.py                        — Free-text → class code, with synonyms

scripts/
  01_download_data.py              — [shared] HAM10000 download + manifest builder
  04_inspect.py                    — [shared] Parse pilot_results.csv → summary
  mac/
    smoke_test.py                  — Mac: 1 image, 1 prompt — gate before pilot
    run_pilot.py                   — Mac: full pilot via mlx-vlm
    run_binary.py                  — Mac: mel vs not_mel side experiment
  dcc/
    02_smoke_test.py               — DCC: 1 image, 1 prompt
    03_run_pilot.py                — DCC: full pilot via transformers
    run_pilot.sh                   — Slurm batch script for 03_run_pilot.py
  compare/
    compare_pilot.py               — Run P1-P4 for a single model → results/pilot_<tag>.csv
    compare_inspect.py             — Aggregate all results/pilot_*.csv → comparison table

results/                           — all generated CSVs and logs (kept under git)
  pilot_manifest.csv               — 21 selected images + true labels + paths
  pilot_manifest_mini.csv          — smaller manifest for quick comparisons
  pilot_results.csv                — raw model outputs (one row per image × prompt)
  pilot_inspection.csv             — parsed labels + correctness + notes column
  pilot_<tag>.csv                  — per-model outputs from compare_pilot.py
  compare_inspection.csv           — combined comparison table
  smoke_test_output.txt
  *_run.log                        — captured run logs

data/                              — gitignored, large
  ham10000/                        — dataset (~3GB)
```

**Path conventions:** every script anchors to `REPO_ROOT = Path(__file__).resolve().parents[N]`
at the top and reads/writes under `RESULTS_DIR = REPO_ROOT / "results"` and
`DATA_DIR = REPO_ROOT / "data" / "ham10000"`. This lets scripts run from any CWD.
Imports of `prompts` / `parser` work because each script prepends `REPO_ROOT/src` to
`sys.path`.

---

## How the pipeline is wired

```
scripts/01_download_data.py
        ↓
   results/pilot_manifest.csv  (21 rows: image_id, true_label, image_path)
        ↓
   ┌──────────────────────────────┬─────────────────────────────┐
   ↓                                                            ↓
scripts/mac/smoke_test.py                       scripts/dcc/02_smoke_test.py     (one-image gate)
   ↓                                                            ↓
scripts/mac/run_pilot.py                        scripts/dcc/03_run_pilot.py      (21 × 2 inferences)
   ↓                                                            ↓
   └──────────────────────────────┬─────────────────────────────┘
                                  ↓
                  results/pilot_results.csv  (raw outputs, never re-parsed downstream)
                                  ↓
                          scripts/04_inspect.py
                                  ↓
                  results/pilot_inspection.csv  (parsed labels + notes)
```

Important: **parsing is decoupled from generation.** Raw outputs are persisted, so the
parser can be iterated on without re-running the model. Always preserve raw outputs.

---

## Class vocabulary (HAM10000, 7 classes)

Used everywhere — prompts, parser, evaluation. Don't change without sweeping.

| Code  | Full name                                            | Frequency |
|-------|------------------------------------------------------|-----------|
| nv    | Melanocytic nevus                                    | ~67% (dominant) |
| mel   | Melanoma                                             | ~11%      |
| bkl   | Benign keratosis                                     | ~11%      |
| bcc   | Basal cell carcinoma                                 | ~5%       |
| akiec | Actinic keratosis / intraepithelial carcinoma        | ~3%       |
| vasc  | Vascular lesion                                      | ~1%       |
| df    | Dermatofibroma                                       | ~1%       |

Heavy class imbalance is a real signal — **always report macro-F1 and balanced accuracy,
never raw accuracy alone.** A model that always predicts `nv` gets ~67% accuracy and
tells us nothing.

---

## Prompt design

**Three orthogonal axes, 2³ = 8 conditions:**

- **Shots:** 0 (zero-shot) vs F (few-shot, k=3 demos)
- **Role:** N (no role) vs R ("You are an experienced dermatologist…")
- **Reasoning:** D (direct answer) vs C (chain-of-thought, ABCD rule)

Naming: `P<shots><role><reasoning>`. So `P0ND` = zero-shot, no role, direct → P1.
And `PFRC` = few-shot, role, CoT → P8.

**Pilot uses only P1 and P4** (the most-different conditions). The full 8 conditions are
already implemented in `prompts.py` for the scaled-up run later.

**Few-shot demos must be picked deliberately and reused across all few-shot conditions.**
Don't sample new demos per query — that introduces variance that can't be attributed to
prompting effects. Suggested demos: one easy melanoma, one easy nevus, one tricky case
(e.g., a bcc). To be wired in when scaling beyond the pilot.

**All prompts end with `Final answer: <class_code>`.** This is the strict-format anchor
the parser depends on. Never relax this without updating the parser.

---

## Parser behavior

`parse(output: str) -> tuple[label, method]` where method is one of:
- `strict` — matched the literal `Final answer: <code>` line
- `synonym` — matched a known synonym (e.g., "melanoma" → mel)
- `fallback` — found a class code as a token anywhere in the output
- `fail` — no match → returns `PARSE_FAIL`

**`PARSE_FAIL` is a real outcome, never silently dropped, never counted as wrong.** The
parse-failure rate per condition is itself a measurement of prompt quality. If P1 has
>10% parse failures, the prompt isn't strict enough.

`parser.py` has self-tests at the bottom (run `python parser.py`). Add a test case
whenever you discover a new output pattern in the wild.

---

## Evaluation principles

- **Headline metric: macro-F1.** Per-class F1 averaged equally. Insulates against class
  imbalance.
- **Also report:** balanced accuracy, per-class F1, parse-failure rate, latency.
- **Confusion matrix** per condition is non-optional during inspection.
- **Determinism:** all generation uses `do_sample=False` / `temperature=0.0`. Same input
  → same output. If results aren't reproducible across runs, something's broken.
- **Always preserve raw outputs.** Re-parse, never re-generate.

---

## Known issues / gotchas

- **Mac, 16GB RAM:** Tight. Close Chrome/Slack/Docker before pilot runs. If it OOMs,
  reduce `max_tokens` for P4 from 500 → 350.
- **Mac, mlx-vlm:** First model load includes JIT compile — looks frozen for ~60s. Don't
  kill it.
- **Kaggle CLI:** Recent versions use `~/.kaggle/access_token` (single string), not the
  legacy `~/.kaggle/kaggle.json`. If `kaggle datasets list` complains about
  `kaggle.json`, run `pip install --upgrade kaggle`.
- **DCC compute nodes:** Often have no internet. Always download data and model weights
  on the login node before submitting batch jobs.
- **HF cache size:** ~9GB for MedGemma bf16, ~3GB for the MLX 4-bit version. Set
  `HF_HOME` to a directory with space — never let the default land in `~/.cache` if home
  has a quota.
- **MedGemma is gated on Hugging Face.** User must request access and accept terms before
  the model will download. Access usually approved instantly but can take a few hours.
- **Numeric file prefixes** (`01_…`, `02_…`) can't be imported normally; we use
  `importlib` for cross-script reuse on DCC. Keep the prefixes anyway — they communicate
  execution order.
- **Bootstrap blocks at the top of each script** set `REPO_ROOT`, add `src/` to
  `sys.path`, and define `RESULTS_DIR`. Keep them — they're what lets scripts run from
  any CWD. The `parents[N]` index is layout-sensitive: scripts/foo.py uses `parents[1]`,
  scripts/mac/foo.py uses `parents[2]`.

---

## Decision log

Decisions made so far that are easy to second-guess later. Don't overturn without reason:

1. **HAM10000 over CXR or pathology** — most-benchmarked classification target with
   public fine-tuned baselines, 7 classes give prompting room to differentiate, manageable
   scale. Radiology and pathology come in later sub-projects.
2. **MedGemma 4B as primary model** — small enough to iterate, medical-specific,
   instruction-tuned, available in MLX 4-bit for Mac development.
3. **Classification framing, not VQA or report generation** — VQA evaluation is messier
   (open vs closed-ended), report generation needs LLM-as-judge or RadGraph. Classification
   has a clean correctness signal.
4. **Compare against published fine-tuned baselines, don't fine-tune ourselves yet** —
   at starting stage, fine-tuning adds confounds without insight to the prompting question.
   ISIC leaderboard numbers are the reference.
5. **Pilot first, scale second** — 21 images × 2 prompts as a plumbing check before
   committing to a full run. The numbers from the pilot are not the experiment.
6. **macro-F1 as headline metric, never raw accuracy** — class imbalance demands it.
7. **Raw outputs persisted, parser decoupled from generation** — so we can iterate on the
   parser without re-burning compute.

---

## What you (Claude Code) should do by default

- **Default to the Mac path** (`scripts/mac/`) unless the user has clearly moved to DCC.
- **Always run `python src/parser.py` self-tests** before claiming the parser is healthy.
- **Never silently drop PARSE_FAIL rows.** They're data.
- **Preserve raw outputs.** If a script writes parsed-only results, fix the script.
- **Keep the 8-condition factorial intact.** Don't add a 9th condition without clearing
  it with the user — it inflates the run cost and complicates analysis.
- **Don't shortcut the smoke test.** Skipping it is the most common way the pilot blows up.
- **Don't introduce new dependencies casually.** Current minimal stack: `mlx-vlm`,
  `pillow`, `pandas`, `tqdm`, `scikit-learn`, `huggingface_hub`, `kaggle` (Mac). Add
  `transformers`, `accelerate`, `bitsandbytes` for DCC. Anything more, ask first.
- **Don't auto-commit or auto-push.** The user controls git.

---

## Likely next tasks

In rough order of likelihood, what the user will probably ask for next:

1. **Debug a smoke-test failure** — most common: HF auth, image format, max_tokens too
   small for CoT, model output not ending with the strict-format line.
2. **Iterate on prompts after pilot inspection** — tighten P1's format compliance, make
   P4's CoT actually mandatory, add output-format examples in the system prompt.
3. **Pick few-shot demos and wire them into P5–P8.** Need to load 3 specific image_ids
   from the manifest, build a `_few_shot_preamble`-compatible call, run the pilot for
   those conditions.
4. **Write `scripts/05_run_full.py`** — all 8 prompts × balanced ~350-image test set
   (50 per class). Should follow the same write-as-you-go CSV pattern as
   `scripts/dcc/03_run_pilot.py` so partial runs survive crashes.
5. **Build a real metrics module** (`src/metrics.py`) — macro-F1, balanced accuracy,
   per-class F1, confusion matrix, parse-failure rate, bootstrap CIs. The one in
   `scripts/04_inspect.py` is a quick-look, not a final-results version.
6. **Add a second model** — LLaVA-Med 8B or Qwen3-VL 8B. New runner, same prompts,
   same parser. Cross-model comparison is the contribution.
7. **Visualize results** — per-axis main effects, interaction plot (shots × CoT etc.),
   gap to fine-tuned reference. Save figures to `figures/`.
8. **Move from local to DCC** — once Mac results look healthy. The user is at Duke; cluster
   = `dcc-login.oit.duke.edu`.

---

## What you should NOT do without explicit user approval

- Run the full pilot without the smoke test passing first.
- Re-download HAM10000 if the metadata file already exists.
- Change the class vocabulary (`CLASSES`, `SYNONYMS`).
- Change the prompt naming convention (`P<shots><role><reasoning>`).
- Add new prompt conditions beyond the 8 already designed.
- Switch to a different primary model (currently MedGemma 4B).
- Pick few-shot demos randomly per query (must be fixed across all few-shot calls).
- Use raw accuracy as a headline metric.
- Silently drop PARSE_FAIL rows from analysis.
- Auto-commit or auto-push to git.
- Add a new dependency without asking.

---

## Useful one-liners

```bash
# Activate the environment
source .venv/bin/activate

# Self-test the parser
python src/parser.py

# Verify MLX sees the GPU
python -c "import mlx.core as mx; print(mx.default_device())"

# Re-build pilot manifest (no re-download)
rm -f results/pilot_manifest.csv && python scripts/01_download_data.py

# Quick stats on pilot results
python -c "import pandas as pd; df=pd.read_csv('results/pilot_results.csv'); print(df.groupby('prompt_id').size())"

# Open inspection CSV in Numbers (Mac)
open results/pilot_inspection.csv

# Watch a Slurm job (DCC)
watch -n 5 squeue -u $USER
```

---

## When the user says…

- *"run the smoke test"* → `python scripts/mac/smoke_test.py` (or `scripts/dcc/02_smoke_test.py` on DCC)
- *"run the pilot"* → smoke test first, then `python scripts/mac/run_pilot.py`
- *"inspect the results"* → `python scripts/04_inspect.py` then open `results/pilot_inspection.csv`
- *"the prompt isn't working"* → look at `results/pilot_inspection.csv`'s `raw_output`
  column, diagnose, then edit `src/prompts.py` (or `scripts/mac/run_pilot.py`'s `p4_inputs()`
  for the Mac pilot), rerun, re-inspect
- *"add few-shot"* → wire P5–P8 with 3 fixed demo images from the manifest
- *"scale up"* → write `scripts/05_run_full.py` with all 8 prompts × balanced 350-image set
- *"add another model"* → new runner file, same prompts, same parser, write outputs to
  a model-specific CSV (e.g., `results/pilot_llavamed.csv`)
- *"compute metrics"* → for now, `scripts/04_inspect.py`. For final analysis, write `src/metrics.py`.

---

## Style preferences

- Keep scripts standalone and runnable (one entrypoint each).
- Prefer `pandas.read_csv` + simple dict iteration over fancy abstractions.
- Always write results incrementally with `f.flush()` so partial runs are usable.
- Comment the *why*, not the *what*. The code shows what; the comments explain why.
- Print progress prominently (tqdm bars, step headers). The user runs these in
  Terminal and watches.
- Errors should be loud. Don't silently catch and continue unless the design demands
  it (e.g., per-image inference failures in batch runs — those go into the CSV as
  `<<EXCEPTION>>` rows).