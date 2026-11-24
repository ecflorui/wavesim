#!/bin/bash
# All CPU/GPU/hybrid measurements as one batch job (owner: Krish).
# Submit from the repository root on a Lonestar6 login node:
#   mkdir -p out/results && sbatch tools/results.slurm
# Add -A YOUR_ALLOCATION to sbatch if needed. Takes about 5 minutes.
# Results: docs/RESULTS.md (tables updated), out/results/, and the log
# out/results/job-<id>.out.
#SBATCH -J wavesim-results
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o out/results/job-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
make gpu
python3 tools/run_results.py
echo "Results finished. Commit docs/RESULTS.md to keep them."
