#!/bin/bash
# laptop or Euler login node: run the keep/revert gate of picker CAND against incumbent INC on a frozen split.
#   bash euler/gate.sh <cand_picker> <inc_picker> [dev|held_out|fresh] [K] [ap3p|ippe] [ap3p|ippe]  e.g. bash euler/gate.sh dpp0.5 cube0.5 dev
# K = keypoint budget (default 100); K != 100 adds -K<K> to the run tag (T8 budget-conditioned gate, docs/t8_k400_gate.md).
# 5th arg SOLVER (R19) = PnP solver of BOTH arms (default ap3p); 6th arg SOLVER_INC overrides the incumbent's solver (default = SOLVER).
# ippe adds -ippe to that arm's run id, so cached AP3P rows are never reused. Same picker on both arms needs two different solvers
# (cand = inc compares the solvers on one picker: `gate.sh cube0.5 cube0.5 fresh 100 ippe ap3p`), otherwise the gate is refused.
# Submits one array task per flight of the split (both pickers per task), then a summary job that writes
# results/gates/<cand>__vs__<inc>__<split>.json and appends the verdict to results/gates/log.jsonl (on Euler).
# Held-out is refused by kpbench.gate unless the candidate passed DEV and has no earlier held-out verdict.
# fresh: flights from splits_fresh.json / euler/flights_fresh.txt (kpbench.fresh), run tag g-<CAND>-fresh.
# SBATCH_BEGIN=<time> in the environment books both jobs for later (sbatch reads it).
set -euo pipefail
CAND=$1; INC=$2; SPLIT=${3:-dev}; K=${4:-100}; SOLVER=${5:-ap3p}; SOLVER_INC=${6:-$SOLVER}
for s in $SOLVER $SOLVER_INC; do case $s in ap3p|ippe) ;; *) echo "solver $s not in ap3p|ippe"; exit 2 ;; esac; done
if [ "$CAND" = "$INC" ] && [ "$SOLVER" = "$SOLVER_INC" ]; then echo "refused: $CAND vs itself with solver $SOLVER on both arms = one run id"; exit 2; fi
if [ -z "${SLURM_CLUSTER_NAME:-}" ] && ! hostname | grep -q '^eu-'; then
  ssh -o ConnectTimeout=5 -o BatchMode=yes euler true || { echo "VPN off: gate $CAND vs $INC ($SPLIT) not submitted"; exit 2; }
  cd "$(dirname "$0")/.." && bash euler/push.sh >/dev/null
  exec ssh euler "cd ~/kpbench && bash euler/gate.sh $CAND $INC $SPLIT $K $SOLVER $SOLVER_INC"
fi
source ~/kpbench/euler/env.sh >/dev/null 2>&1; cd ~/kpbench
FL=euler/flights.txt; SPL=splits.json; TAG=g-$CAND
if [ "$SPLIT" = fresh ]; then FL=euler/flights_fresh.txt; SPL=splits_fresh.json; TAG=g-$CAND-fresh; fi
[ "$K" = 100 ] || TAG=$TAG-K$K
IDX=$(python - "$SPLIT" "$FL" "$SPL" <<'PY'
import json, sys
fl = [l.strip() for l in open(sys.argv[2]) if l.strip()]
print(','.join(str(fl.index(f)) for f in json.load(open(sys.argv[3]))[sys.argv[1]]))
PY
)
if [ "$SPLIT" = held_out ]; then
  python -c "import json,sys; from kpbench.gate import held_out_allowed; w=held_out_allowed('$CAND'); sys.exit(('refused: '+w) if w else 0)"
fi
J=$(sbatch --parsable --array=$IDX --export=ALL,CAND=$CAND,INC=$INC,FL=$FL,TAG=$TAG,K=$K,SOLVER=$SOLVER,SOLVER_INC=$SOLVER_INC euler/gate.sbatch)
R="raco_%s_K${K}_same_lg_$TAG"; RC=$R; RI=$R; [ "$SOLVER" = ap3p ] || RC=$R-$SOLVER; [ "$SOLVER_INC" = ap3p ] || RI=$R-$SOLVER_INC
S=$(sbatch --parsable --dependency=afterok:$J --account=ls_polle --time=00:20:00 --mem-per-cpu=4G --job-name=kpb_gsum \
  --output=$HOME/kpbench/logs/gsum_%j.log \
  --wrap="bash -c 'source ~/kpbench/euler/env.sh && python -m kpbench.gate --cand $(printf $RC $CAND) --inc $(printf $RI $INC) --split $SPLIT --log'")
echo "gate $CAND vs $INC on $SPLIT (K $K, solver $SOLVER / $SOLVER_INC): array $J ($IDX), summary $S"
