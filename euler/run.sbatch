#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_run
#SBATCH --array=0-30
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/run_%A_%a.log
# one flight, every configuration listed in RUNS (space separated "det:picker:K:map[:tag]")
# usage: [PICKS=1] [MATCHER=lg] RUNS="raco:sd:100:same raco:grid:100:same" sbatch euler/run.sbatch
#   MATCHER nn (default, mutual nearest neighbour) or lg (LightGlue, runs on the CPU here)
set -euo pipefail
source ~/kpbench/euler/env.sh
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" euler/flights.txt)
: "${RUNS:=raco:sd:100:same raco:cube0.5:100:same raco:grid:100:same sp:sd:100:same sp:cube0.5:100:same sp:grid:100:same}"
: "${MATCHER:=nn}"
for r in $RUNS; do
  IFS=: read det picker K map tag <<< "$r"
  python -m kpbench.run --det $det --picker $picker --K $K --map $map --matcher $MATCHER --flights $F ${PICKS:+--save-picks} ${tag:+--tag $tag}
done
