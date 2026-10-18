#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_gate
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/gate_%A_%a.log
# one flight of the gate: candidate and incumbent pickers back to back in this task (same node = same LightGlue device).
# Submitted by euler/gate.sh (array = the split's flights). Run ids carry tag g-<CAND> so parallel gates never share files.
# FL / TAG (set by gate.sh): flight list and run tag (split fresh: euler/flights_fresh.txt, g-<CAND>-fresh).
set -euo pipefail
source ~/kpbench/euler/env.sh
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${FL:-euler/flights.txt})
for p in $CAND $INC; do
  python -m kpbench.run --det ${DET:-raco} --picker $p --K ${K:-100} --map same --matcher lg --flights $F --force --tag ${TAG:-g-$CAND}
done
