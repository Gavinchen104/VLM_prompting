# Local Quickstart — Apple Silicon Mac (16GB)

This is the fast path for running the pilot **locally** on your Mac before
moving to DCC. We use Apple's MLX framework with a 4-bit-quantized MedGemma
(3GB instead of 9GB), which fits comfortably in 16GB unified memory.

The cluster code (`scripts/dcc/`) uses transformers and is for DCC. The Mac code
is in `scripts/mac/`. Same prompts, same parser, same pilot manifest — just a
different inference backend.

---

## 0. Prereqs

- macOS 14+ (Sonoma or later) on M1/M2/M3/M4
- Python 3.11+ installed (`python3 --version`)
- About 5GB free disk for the model + dataset
- A Hugging Face token (already done)
- A Kaggle account + token (https://kaggle.com/settings → Create New Token)

---

## 1. Set up a Python venv

In the project folder (where you `tar xzf`'d this):

```bash
cd ~/path/to/medvlm_pilot

# Create a venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install MLX-VLM + supporting libs
pip install --upgrade pip
pip install mlx-vlm pillow pandas tqdm scikit-learn
pip install huggingface_hub kaggle
```

Verify MLX sees your GPU (Metal):

```bash
python -c "import mlx.core as mx; print('Metal device:', mx.default_device()); print('mlx version:', mx.__version__ if hasattr(mx, '__version__') else 'ok')"
```

You should see something like `Metal device: Device(gpu, 0)`.

---

## 2. Configure HF + Kaggle credentials

```bash
# Hugging Face — paste your token when prompted
huggingface-cli login

# Kaggle — drop the kaggle.json you downloaded into ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test
kaggle datasets list -s ham10000 | head
```

---

## 3. Download HAM10000 + build pilot manifest

```bash
python scripts/01_download_data.py
```

This downloads ~3GB and writes `results/pilot_manifest.csv` with 21 rows. Same script as for DCC.

If you already have the dataset, this script is a no-op.

---

## 4. Smoke test (one image, one prompt)

```bash
python scripts/mac/smoke_test.py
```

What this does:
1. Downloads `mlx-community/medgemma-4b-it-4bit` (~3GB) on first run
2. Loads it via mlx-vlm
3. Runs the first image from your manifest with prompt P1
4. Saves output to `results/smoke_test_output.txt`

Expected timing on M1/M2 16GB: model load ~30s, generation ~10–20s for 100 tokens.

If this prints sensible output, **do not run the full pilot yet** — open
`results/smoke_test_output.txt` and read it. Make sure:
- The output isn't empty or an error
- The model said *something* about the image (color, shape, structure)
- The output ends with `Final answer: <code>` or close to it

---

## 5. Full pilot (21 images × 2 prompts)

```bash
python scripts/mac/run_pilot.py
```

Expected timing on M1/M2 16GB:
- Model already cached: ~5s startup
- Each inference: 10–30s (depends on prompt; CoT is longer)
- Total: 15–25 minutes for 42 inferences

Output: `results/pilot_results.csv` (same schema as DCC). The progress bar tells you
how it's going. **Don't quit when you see the model warning text** — first-run
warnings about chat templates are normal.

If your Mac gets warm or laggy: that's normal. MLX uses the GPU efficiently
but a 4B model is real work. Close other heavy apps (Chrome with 30 tabs,
Slack, etc.) to leave more memory for inference.

---

## 6. Inspect the results

```bash
python scripts/04_inspect.py
```

This is the **same script** as for DCC — it reads `results/pilot_results.csv` and
writes `results/pilot_inspection.csv` with parsed labels, plus a console summary.

Open `results/pilot_inspection.csv` in Numbers or Excel and read every row. The
inspection step is identical to the DCC plan.

---

## What's the same vs. different from DCC

|                  | Mac local           | DCC                          |
|------------------|---------------------|------------------------------|
| Model            | 4-bit MLX (~3GB)    | bf16 transformers (~9GB)     |
| Framework        | mlx-vlm             | transformers                 |
| Speed            | ~10-30s/inference   | ~1-3s/inference              |
| Memory           | ~6GB peak           | ~12GB peak                   |
| Quality          | Slightly degraded   | Full quality                 |
| Pilot manifest   | Same                | Same                         |
| Prompts          | Same                | Same                         |
| Parser           | Same                | Same                         |
| Inspection step  | Same                | Same                         |

The local pilot is for **proving the pipeline works**, not for measuring.
Once the pipeline is clean, the real numbers come from DCC at full precision.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: mlx` | `pip install mlx-vlm` (also installs `mlx`, `mlx-lm`) |
| Model download hangs | Check `~/.cache/huggingface/hub/` — it might already be there |
| `Metal device not found` | Check macOS version (need 14+) and `python -c "import platform; print(platform.machine())"` should print `arm64` |
| Inference seems stuck | First run is slow due to JIT compilation. Wait 60s before worrying. |
| Out of memory | Close Chrome/Slack/etc.; restart Python; reboot if needed |
| Output is garbage / empty | Check the chat template — `scripts/mac/smoke_test.py` will print it |
