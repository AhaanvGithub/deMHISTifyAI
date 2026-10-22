#!/bin/bash
#SBATCH --job-name=embed-views
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Usage (from the gutfeeling repo root):
#   sbatch research/embed_views.sbatch <image_dir> <out_dir> <prompts.json> [extra args, e.g. --limit 50]
set -euo pipefail
mkdir -p logs
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
nvidia-smi --query-gpu=name,memory.total --format=csv

python research/embed_views.py --input-dir "$1" --out-dir "$2" --prompts "$3" "${@:4}"
