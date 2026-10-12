#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_sub1
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/sub1_%j.log
# SUB1 (docs/sub1_metrics.md): DEV rows with the candidate metric columns for sd, cube0.5 and grid, all three on one node
# (same LightGlue device), then the pre-registered selection -> results/sub1/. usage: sbatch euler/sub1.sbatch [SPLIT=held_out]
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
SPLIT=${SPLIT:-dev}
for p in sd cube0.5 grid; do
  python -m kpbench.evaluate --picker $p --split $SPLIT --metrics --save
done
python -m kpbench.sub1 --split $SPLIT
