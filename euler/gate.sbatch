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
# SOLVER / SOLVER_INC (set by gate.sh, default ap3p): PnP solver of the candidate / incumbent arm (ippe adds -ippe to that run id).
# SAVE_PICKS=1 (set by gate.sh from the queue item's "save_picks", T33): --save-picks on both runs (per-pick u,v + on-plane labels).
set -euo pipefail
source ~/kpbench/euler/env.sh
F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${FL:-euler/flights.txt})
PICKS=""; [ "${SAVE_PICKS:-0}" = 1 ] && PICKS=--save-picks
python -m kpbench.run --det ${DET:-raco} --picker $CAND --K ${K:-100} --map same --matcher lg --flights $F --force --tag ${TAG:-g-$CAND} --solver ${SOLVER:-ap3p} $PICKS
python -m kpbench.run --det ${DET:-raco} --picker $INC --K ${K:-100} --map same --matcher lg --flights $F --force --tag ${TAG:-g-$CAND} --solver ${SOLVER_INC:-ap3p} $PICKS
