#!/bin/bash
# laptop side: fetch the named runs from Euler and OVERWRITE the local copies (pull.sh keeps local files, which mixed
# laptop and Euler rows inside one run: audit 11 Oct, C4). Usage: bash euler/pull_runs.sh <run_id> [<run_id> ...]
set -euo pipefail
cd "$(dirname "$0")/.."
for rid in "$@"; do
  rm -rf "results/runs/$rid.tmp"; mkdir -p "results/runs/$rid.tmp"
  ssh euler "cd ~/kpbench/results/runs/$rid && tar czf - ." | tar xzf - -C "results/runs/$rid.tmp"
  rm -rf "results/runs/$rid"; mv "results/runs/$rid.tmp" "results/runs/$rid"; echo "pulled $rid"
done
[ -n "${NOEXPORT:-}" ] || python -m kpbench.export
