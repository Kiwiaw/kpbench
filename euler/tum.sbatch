#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_tum
#SBATCH --array=0-9
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/tum_%A_%a.log
# TUM RGB-D (one sequence per task, all pickers on one node): convert (tools/tum_convert.py, every 3rd frame = 10 Hz),
# extract, labels, then LightGlue K100 same-map runs. Protocol change vs TartanAir: keyframe every 5th kept frame (0.5 s),
# map = keyframes within +-10 kept frames (1 s). Separate cache + results dirs (results_tum), never mixed with TartanAir.
# usage: PICKERS="dpp0.5 sd cube0.5" sbatch euler/tum.sbatch
set -euo pipefail
source ~/kpbench/euler/env.sh
TUM=${TUM:-$KPBENCH_DATA/tum}
SEQ=$(ls -d $TUM/rgbd_dataset_freiburg* | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
export KPBENCH_INTR="560 560 320 240" KPBENCH_KF=5 KPBENCH_WIN=10
export KPBENCH_CACHE=$KPBENCH_DATA/kpbench/cache_tum KPBENCH_RESULTS=$HOME/kpbench/results_tum
python tools/tum_convert.py $SEQ --out $KPBENCH_DATA/kpbench_tum --stride 3
F=tum/$(python -c "import sys; sys.path.insert(0, 'tools'); from tum_convert import short_name; print(short_name('$SEQ')[0])")
python -m kpbench.extract --flights $F --dets raco,sp
python -m kpbench.labels --flights $F
for p in ${PICKERS:-dpp0.5 sd cube0.5}; do
  python -m kpbench.run --det raco --picker $p --K 100 --map same --matcher lg --flights $F --force --notes "tum stride3 KF5 WIN10"
done
