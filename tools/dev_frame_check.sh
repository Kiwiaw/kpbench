#!/bin/bash
# Euler login node: which flights of a split lost frames? Exit 1 if any flight would be refused by kpbench.run.  usage: bash tools/dev_frame_check.sh [dev|held_out|fresh]
set -euo pipefail
source ~/kpbench/euler/env.sh >/dev/null 2>&1; cd ~/kpbench
python tools/dev_frame_check.py --split "${1:-dev}"
