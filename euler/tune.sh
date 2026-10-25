#!/bin/bash
# Euler login/compute node (or laptop: forwards over ssh after the VPN check): Optuna study on DEV flat queries.
#   bash euler/tune.sh <study> [trials=30] [workers=6]        e.g. bash euler/tune.sh dpp_dev1 30
# Workers share results/tune/<study>.journal; a final job writes results/tune/<study>.json/.md (mirrored by the agent executor).
set -euo pipefail
STUDY=$1; TRIALS=${2:-30}; W=${3:-6}
case $STUDY in *_grid*) W=$TRIALS;; esac   # fixed grid: one arm per worker, no TPE arm (kpbench.tune GRIDS)
if [ -z "${SLURM_CLUSTER_NAME:-}" ] && ! hostname | grep -q '^eu-'; then
  ssh -o ConnectTimeout=5 -o BatchMode=yes euler true || { echo "VPN off: tune $STUDY not submitted"; exit 2; }
  exec ssh euler "cd ~/kpbench && bash euler/tune.sh $STUDY $TRIALS $W"
fi
source ~/kpbench/euler/env.sh >/dev/null 2>&1; cd ~/kpbench
python -m kpbench.tune --study $STUDY --seed-only
PER=$(( (TRIALS + W - 1) / W ))
J=$(sbatch --parsable --array=0-$((W-1)) --account=ls_polle --cpus-per-task=8 --mem-per-cpu=6G --time=12:00:00 --job-name=kpb_tune \
  --output=$HOME/kpbench/logs/tune_${STUDY}_%A_%a.log --wrap="bash -c 'source ~/kpbench/euler/env.sh && python -m kpbench.tune --study $STUDY --trials $PER'")
S=$(sbatch --parsable --dependency=afterany:$J --account=ls_polle --time=00:10:00 --mem-per-cpu=4G --job-name=kpb_tsum \
  --output=$HOME/kpbench/logs/tsum_%j.log --wrap="bash -c 'source ~/kpbench/euler/env.sh && python -m kpbench.tune --study $STUDY --summary'")
echo "tune $STUDY: array $J ($W workers x $PER trials), summary $S"
