#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --job-name=kpb_extract_gpu
#SBATCH --array=0-30
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/extract_%A_%a.log
# same as extract.sbatch but the detectors run on one GPU (about 1 min per flight instead of 20)
# usage: sbatch --array=2-30 euler/extract_gpu.sbatch
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
nvidia-smi -L || true
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" euler/flights.txt)
python -m kpbench.extract --flights $F --dets raco,sp
python -m kpbench.labels --flights $F
python -m kpbench.thumbs --flights $F
