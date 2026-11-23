#!/bin/bash
# Submit from the repository root on LS6:
#   mkdir -p out/gpu-check
#   sbatch tools/check_gpu.slurm
# Add -A YOUR_ALLOCATION to sbatch if needed.
#SBATCH -J wavesim-gpu-check
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o out/gpu-check/job-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda
hostname
nvidia-smi
nvcc --version

mkdir -p out/gpu-check
nvcc -O2 -std=c++17 -arch=sm_80 -Iinclude \
    src/gpu_solver.cu tools/test_gpu.cpp -o out/gpu-check/test_gpu
out/gpu-check/test_gpu --full
out/gpu-check/test_gpu --bench
echo "GPU standalone checks and benchmark finished successfully."
