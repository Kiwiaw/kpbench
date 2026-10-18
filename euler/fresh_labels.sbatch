#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=kpb_fresh_l
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/fresh_labels_%j.log
# FRESH flights: per-query labels (plane_frac from the SuperPoint cache, depth, coverage) on CPU, after fresh_extract.sbatch.
set -euo pipefail
source ~/kpbench/euler/env.sh
python -m kpbench.labels --flights "$(paste -sd, ${FL:-euler/flights_fresh.txt})"
