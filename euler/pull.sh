#!/bin/bash
# laptop side: fetch results (runs + labels) and thumbnails from Euler into this repo, then export the page.
# Existing local files are kept (tar --keep-old-files), so laptop runs are never overwritten by cluster runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results docs/data/thumbs
ssh euler 'cd ~/kpbench && tar czf - results' | tar xzf - --keep-old-files 2>/dev/null || true
ssh euler 'cd ~/kpbench/docs/data && tar czf - thumbs' | tar xzf - -C docs/data --keep-old-files 2>/dev/null || true
python -m kpbench.export
