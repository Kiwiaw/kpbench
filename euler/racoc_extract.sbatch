#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --job-name=kpb_racoc
#SBATCH --array=1,13
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/racoc_%A_%a.log
# candidates from RaCo with the retrained ranker head (det "racoc") for one flight; labels and thumbs already exist.
# usage: RACO_WEIGHTS=/cluster/scratch/kkaczmarzyk/md_cube/rank.pth sbatch euler/racoc_extract.sbatch
set -euo pipefail
source ~/kpbench/euler/env.sh
: "${RACO_WEIGHTS:=/cluster/scratch/kkaczmarzyk/md_cube/rank.pth}"
export RACO_WEIGHTS
test -s "$RACO_WEIGHTS"
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" euler/flights.txt)
python -m kpbench.extract --flights $F --dets racoc
