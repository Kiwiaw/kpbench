# Can euler/sub1.sbatch run on FRESH-1? (2026-10-31, laptop, nothing submitted)

Answer: mechanically yes, `SPLIT=fresh PICKERS="..." sbatch --time=06:00:00 euler/sub1.sbatch`. Dispatch kill not met: sub1_prereg and sub1_ferranti read `flat` from the eval rows, not from Q202 labels.

Choices made:
- `SPLIT=fresh`, not the `FRAMES=<json>` the dispatch named: FRESH_set.json holds flights and counts, no frame list, and FRESH-1 = all queries of its 14 flights. Same frames.
- Flights come from the committed results/gates/FRESH_set.json. NFLAT default for fresh = 2075, so a short picker file fails loudly.
- Log path: already fixed by `#SBATCH --output=~/kpbench/logs/sub1_%j.log`, no change. The Q206 log was never read: `sub1_18042016.log` still needs VPN.

Checked on the laptop: `query_sets('fresh')` = 14 flights; all 10 Q210 picker names parse; `bash -n` passes; on one synthetic scene `pcov`, `sel_logdet`, `sel_hull`, `hull` all come out non-None under both solvers.

NOT checked:
- MICRO dry-run: the laptop has no MICRO flight data, so no real rows were written and the scorers did not run on them.
- Fresh labels: `eval_flight` reads results/labels/<fresh flight>.json. The laptop has none (0 of 14); Euler should, since the FRESH gates used them. Check with VPN.
- factory-free edge: oldtown_easy/P004 has 0 flat queries; leave-one-flight-out in sub1_ferranti has not been run with such a flight.

Which of the 10 pickers lack FRESH rows: all 10 lack usable ones. Gate rows have only COLS; the SUB1 metric columns come only from `evaluate --metrics`. So gate rows cannot be reused, and the "6 of 10 have FRESH rows" in the dispatch does not shorten this run. K100 FRESH gate jsons exist for 3 of the 10 (dpp0.5, dppr0.486, dppr0.317); results/sub1/q210_fallback.md, which the dispatch cites, is not in the agent repo.

Cost estimate, unmeasured on FRESH: DEV 3 plain pickers took 32 min (job 16977857), about 11 min per picker on 8273 queries. FRESH has 13440 queries, about 17 min per plain picker. 5 dppr at up to 2.1x plus deco give about 3-4 h on one node, so use --time 06:00:00, not the 04:00:00 in the script. It also gives sd + cube0.5 K100 FRESH rows, which could replace the E1 gate only if run on one node (it does, same job).

Submit only after a Kinga or planner decision: this replaces Q210's DEV read-out only if the prereg (DEV, 22 Oct) is amended first.
