# Q210 v4 hand submit (engineer 2026-11-02 13:40, dispatch 13:15 SUB1 step 1)

- Old log read: ~/kpbench/logs/sub1_18042016.log (23 Oct) ends in `hospital/P044: cache has 577 frames, flight has 13` -> DEV raw frames were already short on 23 Oct.
- DEV inventory today (tools/dev_frame_check.sh dev, results/main/frame_inventory_2026-11-02.md): 3 of 12 DEV flights short (hospital/P044 384/577, hospital/P047 401/533, factory/P008 617/920), so the prereg v4 SPLIT rule selects FRESH (results/sub1/q210_fallback.md: FRESH unless the DEV check passes). FRESH: 14 of 14 flights complete.
- Submitted on Euler from ~/kpbench (code synced by euler/push.sh at 13:25, kpbench 34df3eef + today's uncommitted tools):

| job | command | time | state 13:40 |
|---|---|---|---|
| 19377170 | `SPLIT=fresh PICKERS='dpp0.5 cube0.5 grid sd dppr0.486a0.453b1.02t0.9 dppr0.609a0.731b2.91t0.9 dppr0.317a1.6b2.01t0.9 dppr0.287a1.47b2.08t0.9 dppr0.345a1.98b3t0.9 deco' sbatch --mem-per-cpu=10G --time=06:00:00 euler/sub1.sbatch` | 6 h, 1 GPU | RUNNING (eu-g3-062) |
| 19377172 | `SPLIT=fresh K=400 PICKERS='cube0.5 dpp0.5' sbatch --mem-per-cpu=10G --time=03:00:00 euler/sub1.sbatch` (v4 K400 sanity case, rows only) | 3 h, 1 GPU | RUNNING (eu-g3-066) |

- Logs: ~/kpbench/logs/sub1_19377170.log and sub1_19377172.log (sbatch --output). Rows land in ~/kpbench/results/eval/fresh/raco_<picker>_K100_lg_m.json.gz (K400: _K400_). sub1.sbatch checks 2075 flat rows per picker and runs sub1_ferranti + sub1_prereg (K100 job only).
- Read-out: pre-registered in results/sub1/q210_fallback.md (rules 1-3, leave-one-picker-out and leave-one-flight-out binding, K400 sanity case cube <= dpp and ordering not following rep5). Analyst step after COMPLETED; the kill 'no candidate passes' closes SUB1 as 'no second metric'.
- Caveat: FRESH-1 is spent for MAIN gates (8 gates), SUB1 rows on it are new and confirmatory only, no tuning.
