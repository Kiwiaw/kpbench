#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=kpb_sub1
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/sub1_%j.log
# SUB1 (docs/sub1_metrics.md): DEV rows with the candidate metric columns for sd, cube0.5 and grid, all three on one node
# (same LightGlue device), then the pre-registered selection -> results/sub1/. usage: [SPLIT=held_out|fresh] [PICKERS='dpp0.5'] [K=400] sbatch euler/sub1.sbatch
# PICKERS other than the default triple only writes the eval rows (no selection rerun, sd/cube0.5/grid rows untouched).
# K (default 100) = keypoint budget of the rows (Q210 v4 K400 sanity case: K=400 PICKERS='cube0.5 dpp0.5'); the selection scripts read K100 rows only.
# Every picker file is checked for NFLAT flat rows (DEV 936, FRESH-1 2075); any failed or short picker -> exit 1 after all pickers ran.
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
SPLIT=${SPLIT:-dev}
PICKERS=${PICKERS:-sd cube0.5 grid}
K=${K:-100}
NFLAT=${NFLAT:-$(case $SPLIT in dev) echo 936 ;; fresh) echo 2075 ;; *) echo 0 ;; esac)}   # expected flat rows per picker file (0 = no check)
BAD=""
echo "[sub1] $(date +%T) job ${SLURM_JOB_ID:-local} split $SPLIT K $K pickers: $PICKERS"
for p in $PICKERS; do    # one failing picker no longer kills the rest; every step is stamped in the log
  echo "[sub1] $(date +%T) start $p ($SPLIT, K $K)"
  python -m kpbench.evaluate --picker $p --split $SPLIT --K $K --metrics --save || { echo "[sub1] FAIL $p exit $?"; BAD="$BAD $p"; continue; }
  python - "$p" "$SPLIT" "$NFLAT" "$K" <<'PY' || BAD="$BAD $p"
import gzip, json, os, sys
from kpbench.evaluate import EVAL_DIR
p, split, nf, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(gzip.open(os.path.join(EVAL_DIR, split, 'raco_%s_K%d_lg_m.json.gz' % (p, K)), 'rt'))
n = sum(d['flat']); print('[sub1] rows %s: %d queries, %d flat (expected %s)' % (p, len(d['rows']), n, nf or 'any'))
sys.exit(1 if nf and n != nf else 0)
PY
done
if [ "$K" = 100 ]; then    # the selection scripts read the K100 row files; a K400 sanity run only writes rows
[ "$PICKERS" != "sd cube0.5 grid" ] || python -m kpbench.sub1 --split $SPLIT || BAD="$BAD sub1"
python -m kpbench.sub1_ferranti --split $SPLIT || BAD="$BAD sub1_ferranti"    # Ferranti baseline over every picker file present
case " $PICKERS " in *" dpp0.5 "*) python -m kpbench.sub1_prereg --split $SPLIT || BAD="$BAD sub1_prereg" ;; esac    # prereg v2 rules 1-3 (needs sd cube0.5 dpp0.5)
fi
[ -z "$BAD" ] || { echo "[sub1] FAILED:$BAD"; exit 1; }
echo "[sub1] $(date +%T) all OK"
