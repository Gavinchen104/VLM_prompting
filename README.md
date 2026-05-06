# Medical VLM Prompting Pilot

A minimal-but-real first experiment for prompt comparison on HAM10000 with MedGemma 4B,
designed to run on Duke DCC.

## What this does

1. Downloads HAM10000 (Kaggle, ~3GB)
2. Builds a balanced 21-image pilot subset (3 per class)
3. Loads MedGemma 4B (1.5 if you have access, else v1)
4. Runs 2 prompts × 21 images = 42 inferences
   - **P1**: zero-shot, no role, direct answer
   - **P4**: zero-shot, dermatologist role, chain-of-thought
5. Saves raw outputs + a parsed inspection spreadsheet

The point is **not to get good numbers**. The point is to learn:
- Does the model obey our output format?
- Does CoT actually produce reasoning?
- Does our parser handle real outputs?
- What classes does the model over/under-predict?

## How to use

Read `00_setup.md` and follow it step by step. Key gates:

| Gate | What must work | If it fails |
|---|---|---|
| 1. Setup | `import torch; torch.cuda.is_available()` | Wrong CUDA install |
| 2. Auth | `huggingface-cli whoami` shows your name | Bad token |
| 3. Data | `pilot_manifest.csv` has 21 rows | Kaggle creds wrong |
| 4. Smoke | `02_smoke_test.py` prints model output | Access denied / OOM |
| 5. Pilot | `pilot_results.csv` has 42 rows | See slurm-*.err |
| 6. Inspect | `pilot_inspection.csv` opens and looks sensible | (this is the science) |

## Two paths: pick one

- **Local Mac (Apple Silicon)** — for proving the pipeline works. See `mac_quickstart.md`.
- **Duke DCC (cluster)** — for the actual measurement runs. See `00_setup.md`.

Recommended: do the Mac run *first* to validate plumbing, then DCC for real numbers.

## Files

```
README.md              ← You are here.
mac_quickstart.md      ← Mac-only setup (mlx-vlm, 4-bit MedGemma).
00_setup.md            ← DCC setup (Slurm, transformers, bf16).

01_download_data.py    ← [shared] Download HAM10000, build pilot subset.
prompts.py             ← [shared] All 8 prompt conditions (DCC version).
parser.py              ← [shared] Output → class-code parser, with synonyms.
04_inspect.py          ← [shared] Parse + summarize for hand review.

mac_smoke_test.py      ← Mac: one-image sanity check.
mac_run_pilot.py       ← Mac: full pilot (21 × 2 inferences).

02_smoke_test.py       ← DCC: one-image sanity check.
03_run_pilot.py        ← DCC: full pilot (run via sbatch).
run_pilot.sh           ← DCC: Slurm submission script for 03.
```

## What comes next (after the pilot)

Decide based on what you see in `pilot_inspection.csv`:

- **If parse failures > 10%** → tighten the prompt, re-run pilot
- **If P4 doesn't actually reason** → rewrite the CoT instruction
- **If model refuses or returns blanks** → the model may need a different message format
- **If everything looks healthy** → scale to all 8 prompts × ~50 images per class
  (an `05_run_full.py` to be written once we know the pilot is clean)

Then add a second model for cross-architecture validation (LLaVA-Med 8B or Qwen3-VL).
