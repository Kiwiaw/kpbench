#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=kpb_sub1_score
#SBATCH --output=/cluster/home/kkaczmarzyk/kpbench/logs/sub1_score_%j.log
# SUB1 scorers only (CPU, 0 GPU-h): Ferranti leave-one-flight-out fit and the prereg v2 rules on row files that already exist
# in results/eval/<split>/ (written by euler/sub1.sbatch). usage: [SPLIT=fresh] [K=400] [METRICS='hull pcov'] [STRONG=cube0.5] [OTHER=none] sbatch euler/sub1_score.sbatch
# Default = the K400 K-dependence read-out (dispatch 8 Nov, T10): Q210 v4 sanity rows cube0.5 + dpp0.5 at K400 on FRESH-1,
# so the prereg scorer runs one pair only (dpp0.5 vs cube0.5, no sd at K400); outputs results/sub1/<split>_K<K>_ferranti[_<m>].md
# and <split>_K<K>_prereg.md. The K100 prereg verdict is never touched (different file names).
set -euo pipefail
source ~/kpbench/euler/env.sh
cd ~/kpbench
SPLIT=${SPLIT:-fresh}
K=${K:-400}
METRICS=${METRICS:-hull pcov}
STRONG=${STRONG:-cube0.5}
OTHER=${OTHER:-none}
echo "[sub1_score] $(date +%T) job ${SLURM_JOB_ID:-local} split $SPLIT K $K metrics: $METRICS strong $STRONG other $OTHER"
ls -la results/eval/$SPLIT/raco_*_K${K}_lg_m.json.gz
BAD=""
for m in $METRICS; do
  python -m kpbench.sub1_ferranti --split $SPLIT --K $K --metric $m || BAD="$BAD ferranti_$m"
done
python -m kpbench.sub1_prereg --split $SPLIT --K $K --uniform dpp0.5 --strong $STRONG --other $OTHER || BAD="$BAD prereg"
[ -z "$BAD" ] || { echo "[sub1_score] FAILED:$BAD"; exit 1; }
echo "[sub1_score] $(date +%T) all OK"
