#!/bin/bash
# Hypersim runs on the laptop GPU: extract, labels, then LightGlue K100 same-map runs (separate cache + results dirs).
# Protocol change vs TartanAir (frames ~1 m apart): keyframe every 2nd frame, query uses keyframes within +-3 frames (4 keyframes: q-3, q-1, q+1, q+3; WIN=2 -> 2 keyframes).
set -e
cd "$(dirname "$0")/.."
export KPBENCH_DATA="C:/Users/Kinga/Desktop/ETH_lab/local_data" KPBENCH_INTR="554.256258 554.256258 320 240"
export KPBENCH_KF=2 KPBENCH_WIN=${WIN:-3} KPBENCH_CACHE="$PWD/cache_hypersim" KPBENCH_RESULTS="$PWD/results_hypersim${RTAG:-_w3}" DEV=cuda
export RACO_DIR="C:/Users/Kinga/Desktop/ETH_lab/local_tools/RaCo"
FL=${FL:-hypersim/ai_001_001,hypersim/ai_001_002}   # ai_003_001 dropped: black renders
python -m kpbench.extract --flights $FL --dets raco,sp
python -m kpbench.labels --flights $FL
for p in ${PICKERS:-dpp0.5 sd cube0.5}; do
  python -m kpbench.run --det raco --picker $p --K 100 --map same --matcher lg --flights $FL --notes "hypersim KF2 WIN$KPBENCH_WIN"
  echo "RUN_DONE $p"
done
echo ALL_DONE
