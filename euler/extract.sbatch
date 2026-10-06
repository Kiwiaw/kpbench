#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --job-name=kpb_extract
#SBATCH --array=0-30
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/extract_%A_%a.log
# detector candidates + labels + thumbnails for one flight (CPU; RaCo and SuperPoint run fine on 8 cores)
set -euo pipefail
source ~/kpbench/euler/env.sh
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" euler/flights.txt)
python -m kpbench.extract --flights $F --dets raco,sp
python -m kpbench.labels --flights $F
python -m kpbench.thumbs --flights $F
