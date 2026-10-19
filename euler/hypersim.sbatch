#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=kpb_hs
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/hs_%A_%a.log
# Hypersim from the shared CVG processed copy (tools/hypersim_processed.py), one scene per task, all pickers on one node.
# Protocol = laptop results_hypersim_w2: keyframe every 2nd frame, map = keyframes within +-2 frames. Separate cache and
# results dir (results_hypersim_eu), never mixed with TartanAir or the laptop GPU rows.
# usage: python tools/hypersim_processed.py --select 12 --skip ai_001_001,ai_001_002 > euler/hypersim_scenes.txt (those two have frame gaps here)
#        PICKERS="dpp0.5 sd" sbatch --array=0-11 euler/hypersim.sbatch
set -euo pipefail
source ~/kpbench/euler/env.sh
SC=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${SCENES:-euler/hypersim_scenes.txt})
export KPBENCH_INTR="554.256258 554.256258 320 240" KPBENCH_KF=2 KPBENCH_WIN=2
export KPBENCH_CACHE=$KPBENCH_DATA/kpbench/cache_hs KPBENCH_RESULTS=$HOME/kpbench/results_hypersim_eu
python tools/hypersim_processed.py --scene $SC --out $KPBENCH_DATA/hypersim
F=hypersim/$SC
python -m kpbench.extract --flights $F --dets raco,sp
python -m kpbench.labels --flights $F
for p in ${PICKERS:-dpp0.5 sd cube0.5}; do
  python -m kpbench.run --det raco --picker $p --K 100 --map same --matcher lg --flights $F --force --notes "hypersim processed KF2 WIN2"
done
