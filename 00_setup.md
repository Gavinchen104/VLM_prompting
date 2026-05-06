# Medical VLM Prompting Pilot — Duke DCC Setup

Goal: get MedGemma 4B running on HAM10000 with 2 prompt variants on 21 images.
Once that works, we scale up.

---

## 0. Before logging into the cluster

**Already in flight (you started these):**
- ✅ Request access to `google/medgemma-4b-it` on Hugging Face
- ✅ Request access to `google/medgemma-1.5-4b-it` on Hugging Face (preferred, newer)
- ✅ Create a HF access token (Read scope)

**You also need:**
- ✅ A Kaggle account (free) — https://kaggle.com
- ✅ A Kaggle API token: account settings → "Create New Token" → downloads `kaggle.json`. Save it.

---

## 1. Log into DCC and pick a working directory

```bash
ssh <netid>@dcc-login.oit.duke.edu
```

Find a directory with space. Your home (`/hpc/home/<netid>`) has a small quota; do NOT cache models there.

```bash
# Check what you have access to
ls /hpc/group/   # group spaces
ls /work/        # if your group has a /work allocation
df -h /hpc/home/$USER
```

Pick a working directory. For this guide, I'll call it `$WORK`. Set it once:

```bash
# Replace with your actual path. Examples:
export WORK=/hpc/group/<yourlab>/$USER/medvlm
# or
export WORK=/work/$USER/medvlm

mkdir -p $WORK
cd $WORK
```

Add `export WORK=...` to your `~/.bashrc` so it persists.

---

## 2. Set up a conda environment (login node — has internet)

DCC has miniconda available as a module, or you can install your own. Easiest:

```bash
module load Miniconda3        # or: module load Anaconda3
# If neither is available, install miniconda yourself:
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $WORK/miniconda3
#   source $WORK/miniconda3/bin/activate

conda create -n medvlm python=3.11 -y
conda activate medvlm

# PyTorch with CUDA. Check `nvidia-smi` on a GPU node first to know your CUDA version.
# Most DCC GPUs are fine with cu121.
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install "transformers>=4.50.0" accelerate
pip install pillow pandas tqdm scikit-learn
pip install huggingface_hub kaggle
pip install "bitsandbytes>=0.43"   # only needed if we quantize later
```

Test that torch sees CUDA (we need to do this on a *GPU node*, not the login node):

```bash
# Grab a quick interactive GPU session to test
srun --partition=gpu-common --gres=gpu:1 --time=00:30:00 --mem=16G --pty bash
# Once you're on a GPU node:
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit   # back to login node
```

---

## 3. Configure HF and Kaggle credentials (login node)

**Hugging Face:**

```bash
# Use a directory in $WORK so the model cache doesn't fill your home quota
export HF_HOME=$WORK/hf_cache
mkdir -p $HF_HOME

# Add these to your ~/.bashrc:
echo 'export HF_HOME='$WORK'/hf_cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=$HF_HOME/transformers' >> ~/.bashrc

# Log in with your token (paste when prompted)
huggingface-cli login
```

**Kaggle:**

```bash
mkdir -p ~/.kaggle
# Upload your kaggle.json from your laptop to the cluster:
#   scp ~/Downloads/kaggle.json <netid>@dcc-login.oit.duke.edu:~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test
kaggle datasets list -s ham10000 | head
```

---

## 4. Download data and model — on the LOGIN NODE

This is critical: compute nodes on DCC may not have internet. Do all downloads here.

```bash
cd $WORK

# Copy the project files (assuming you scp'd this folder up)
ls   # you should see: 01_download_data.py, 02_smoke_test.py, ...

# 4a. Download HAM10000 (a few hundred MB)
python 01_download_data.py

# 4b. Pre-download the model weights (this caches into $HF_HOME)
# This is just a small Python snippet — it pulls ~9GB.
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/medgemma-4b-it')"
# If MedGemma 1.5 access came through, prefer it:
# python -c "from huggingface_hub import snapshot_download; snapshot_download('google/medgemma-1.5-4b-it')"
```

If the model download fails with a 401, your access request hasn't been approved yet — wait for the email.

---

## 5. Smoke test on a GPU node (interactive)

Before submitting a Slurm batch job, prove it all works in an interactive session:

```bash
srun --partition=gpu-common --gres=gpu:1 --time=01:00:00 --mem=24G --pty bash
cd $WORK
conda activate medvlm
python 02_smoke_test.py
```

Expected output:
- Model loads (takes ~30s)
- One HAM10000 image is processed
- You see model output text in the terminal
- A file `smoke_test_output.txt` is saved

If this works, **you're 80% of the way there.** Exit the interactive session.

---

## 6. Run the pilot via sbatch

```bash
cd $WORK
sbatch run_pilot.sh
squeue -u $USER       # see your job
# logs go to slurm-<jobid>.out and slurm-<jobid>.err
```

The job runs 21 images × 2 prompts = 42 inferences. Should take 5–15 minutes on any modern GPU.

When it finishes, you'll have `pilot_results.csv` with raw outputs.

---

## 7. Inspect

Pull the CSV back to your laptop or open it in Jupyter on DCC:

```bash
python 04_inspect.py
```

This prints a summary and writes `pilot_inspection.csv` with parsed labels and a `notes` column for you to fill in by hand.

**Look at every row.** Specifically check:
- Did the model use the `Final answer: <class>` format?
- Did P4 produce actual reasoning, or skip it?
- Are there parse failures? Why?
- Are the predicted classes in our vocabulary?

This is the most important step. The numbers don't matter yet — the *behavior* matters.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `OSError: You are trying to access a gated repo` | HF access not approved | Check email; visit model page |
| `CUDA out of memory` | GPU too small for bf16 | Use 4-bit quantization (see notes in 02_smoke_test.py) |
| `No module named 'transformers'` | Conda env not activated | `conda activate medvlm` |
| Model download hangs on compute node | No internet on that partition | Download from login node first |
| `Permission denied` on kaggle.json | Wrong file permissions | `chmod 600 ~/.kaggle/kaggle.json` |
| Disk quota exceeded | HF cache in home dir | Set `HF_HOME` to `$WORK/hf_cache` and re-download |
