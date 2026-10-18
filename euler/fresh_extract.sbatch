#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=kpb_fresh_x
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/fresh_extract_%j.log
# FRESH flights (kpbench.fresh, euler/flights_fresh.txt): RaCo + SuperPoint candidates on one GPU, models loaded once.
# Folders first: python -m kpbench.fresh link --flights ... (login node). Labels after: euler/fresh_labels.sbatch (CPU).
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
nvidia-smi -L || true
python -m kpbench.extract --flights "$(paste -sd, ${FL:-euler/flights_fresh.txt})" --dets raco,sp
