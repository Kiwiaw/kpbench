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
# Every picker file is checked for NFLAT flat rows (DEV 936); any failed or short picker -> exit 1 after all pickers ran.
set -euo pipefail
source ~/kpbench/euler/env.sh
export DEV=cuda
SPLIT=${SPLIT:-dev}
PICKERS=${PICKERS:-sd cube0.5 grid}
NFLAT=${NFLAT:-$([ "$SPLIT" = dev ] && echo 936 || echo 0)}   # expected flat rows per picker file (0 = no check)
BAD=""
for p in $PICKERS; do    # one failing picker no longer kills the rest; every step is stamped in the log
  echo "[sub1] $(date +%T) start $p ($SPLIT)"
  python -m kpbench.evaluate --picker $p --split $SPLIT --metrics --save || { echo "[sub1] FAIL $p exit $?"; BAD="$BAD $p"; continue; }
  python - "$p" "$SPLIT" "$NFLAT" <<'PY' || BAD="$BAD $p"
import gzip, json, os, sys
from kpbench.evaluate import EVAL_DIR
p, split, nf = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(gzip.open(os.path.join(EVAL_DIR, split, 'raco_%s_K100_lg_m.json.gz' % p), 'rt'))
n = sum(d['flat']); print('[sub1] rows %s: %d queries, %d flat (expected %s)' % (p, len(d['rows']), n, nf or 'any'))
sys.exit(1 if nf and n != nf else 0)
PY
done
[ "$PICKERS" != "sd cube0.5 grid" ] || python -m kpbench.sub1 --split $SPLIT || BAD="$BAD sub1"
python -m kpbench.sub1_ferranti --split $SPLIT || BAD="$BAD sub1_ferranti"    # Ferranti baseline over every picker file present
case " $PICKERS " in *" dpp0.5 "*) python -m kpbench.sub1_prereg --split $SPLIT || BAD="$BAD sub1_prereg" ;; esac    # prereg v2 rules 1-3 (needs sd cube0.5 dpp0.5)
[ -z "$BAD" ] || { echo "[sub1] FAILED:$BAD"; exit 1; }
echo "[sub1] $(date +%T) all OK"
