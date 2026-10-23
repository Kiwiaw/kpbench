# T8 step 3: budget-conditioned gate (spec, 2026-10-23; dispatch 2026-10-23T10:50 MAIN step 1)

## 1. The K = 100 gate is closed
- 10 % bar at K 100 on 936 DEV flat: 14.42 % x 0.9 = 12.98 %, i.e. a drop of 1.44 pt.
- Per-query oracle floor (best of dpp0.5, dppr0.317 per query, 936 DEV flat): 12.29 % [4.27, 18.97]. A perfect K 100
  picker of this pair has 0.69 pt headroom below the bar's 1.44 pt (agent results/PLAN_2026-10-23.md, T11).
- Best open candidates: dppr0.317a1.6b2.01t0.9 -0.75 [-1.86, +0.31] 4/1 REVERT, dppr0.388a1.91b2.85t0.9 -0.11 [-1.38, +1.07]
  3/3 REVERT (both recomputed from rows, audit 2026-10-23 C25). dppr0.317 would need ~2800 flat frames (3 x 936) and its
  point estimate still misses the bar.
- Default (board decision for Kinga, by 2026-10-24): freeze dpp0.5 as the K 100 incumbent.

## 2. New T8 line: gate at the budget where the effect is
Per-K flat fail on 936 DEV flat (agent results/NIGHT_2026-10-21_r7.md; seed-23 recompute NIGHT_2026-10-22_r3):

| K | dpp0.5 | cube0.5 | dpp - cube [flight-block 95 %] | flights cube better / worse |
|---|---|---|---|---|
| 50 | 18.48 | 19.87 | -1.39 [-4.06, +1.30] | 3 / 4 |
| 200 | 14.64 | 14.10 | +0.53 [-2.25, +2.75] | 4 / 2 |
| 400 | 16.88 | 13.03 | +3.85 [+0.85, +7.40] | 5 / 0 |

All frames at K 400: cube0.5 better by 0.37 pt (8273 queries).

### Pre-registered DEV gate (before the gate json exists)
- Rows: Euler results/runs/raco_cube0.5_K400_same_lg vs raco_dpp0.5_K400_same_lg (Q191 K sweep, 12 DEV flights each,
  both pickers of a flight in one array task). No new compute.
- Command (Euler login node): `python -m kpbench.gate --cand raco_cube0.5_K400_same_lg --inc raco_dpp0.5_K400_same_lg --split dev`
  (no --log until Kinga accepts the K-conditioned gate; gate.py checks K, device and labels, and now writes K into the json).
- Pass = the unchanged rule in kpbench/stats.py: route A (one-sided 97.5 % upper bound < 0) or route B (flight sign test),
  AND relative drop >= 10 % (bar at K 400 = 16.88 x 0.9 = 15.19 %, i.e. >= 1.69 pt), AND all-frames upper bound <= +0.2 pt.
- Kill: upper bound >= 0 and sign test fails, or drop < 1.69 pt, or all-frames > +0.2 -> the K 400 reversal stays a SUB4
  observation, no incumbent at any K.
- Expected from the table: -3.85 pt, 22.8 % relative, 5/0 flights (sign p 0.031 one-sided) -> likely KEEP shape; it rests on
  one picker seed (Q209 3-seed check cancelled 23 Oct, revive if the gate passes).

### New tooling (this step)
- euler/gate.sh takes an optional 4th argument K (default 100); K != 100 runs tag g-<cand>-K<K>, so fresh gate reruns at
  K 400 never share files with K 100 gates. euler/gate.sbatch already read K.
- kpbench.gate writes "K" into each gate json.

### Open for Kinga (not decided here)
- Held-out discipline is per picker name (kpbench.gate.held_out_allowed). cube0.5 already has held-out history at K 100, so a
  K 400 held-out exam would be refused. Options: (a) key the one-shot rule by (picker, K); (b) use FRESH-1 (2075 flat) as the
  K 400 confirmation; (c) no K 400 confirmation, DEV-only result in the paper. Nothing in the rule changes until she picks.
- Whether the MAIN target becomes "best picker per budget" (a K-indexed leaderboard) rather than one K 100 incumbent.
