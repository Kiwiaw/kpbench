#!/bin/bash
# laptop side: copy the code to Euler (~/kpbench). Results, caches and site data are never sent this way.
set -euo pipefail
cd "$(dirname "$0")/.."
ssh euler 'mkdir -p ~/kpbench/logs ~/kpbench/results /cluster/scratch/kkaczmarzyk/kpbench/cache'
tar czf - --exclude=results --exclude=cache --exclude=docs/data --exclude=.git --exclude=__pycache__ --exclude=logs . | ssh euler 'tar xzf - -C ~/kpbench'
echo "code synced to euler:~/kpbench"
