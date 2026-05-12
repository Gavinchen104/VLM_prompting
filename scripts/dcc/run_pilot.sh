#!/bin/bash
#SBATCH --job-name=medvlm-pilot
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ---- Edit these if your environment differs ----
# WORK should be exported in your ~/.bashrc; if not, set it here.
: "${WORK:=$PWD}"

# Conda activation. Adjust path if you installed miniconda elsewhere.
# Try common DCC patterns:
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
    module load Miniconda3 2>/dev/null || module load Anaconda3 2>/dev/null || true
fi
conda activate medvlm

# HF cache must point to a big-enough directory
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

cd "$WORK"

echo "Host:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:      $(which python)"
echo "Workdir:     $(pwd)"
echo "HF_HOME:     $HF_HOME"
echo "Start:       $(date)"
echo

python scripts/dcc/03_run_pilot.py

echo
echo "End:         $(date)"
