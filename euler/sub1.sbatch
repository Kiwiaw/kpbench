#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_sub1
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/sub1_%j.log
# SUB1 (docs/sub1_metrics.md): DEV rows with the candidate metric columns for sd, cube0.5 and grid, all three on one node
# (same LightGlue device), then the pre-registered selection -> results/sub1/. usage: [SPLIT=held_out] [PICKERS='dpp0.5'] sbatch euler/sub1.sbatch
# PICKERS other than the default triple only writes the eval rows (no selection rerun, sd/cube0.5/grid rows untouched).
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
SPLIT=${SPLIT:-dev}
PICKERS=${PICKERS:-sd cube0.5 grid}
for p in $PICKERS; do
  python -m kpbench.evaluate --picker $p --split $SPLIT --metrics --save
done
[ "$PICKERS" != "sd cube0.5 grid" ] || python -m kpbench.sub1 --split $SPLIT
python -m kpbench.sub1_ferranti --split $SPLIT    # Ferranti baseline over every picker file present
case " $PICKERS " in *" dpp0.5 "*) python -m kpbench.sub1_prereg --split $SPLIT ;; esac    # prereg v2 rules 1-3 (needs sd cube0.5 dpp0.5)
