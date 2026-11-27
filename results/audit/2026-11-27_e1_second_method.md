# E1 + FRESH-1 K100 close-out line: second method (SUB2, 2026-11-27) [P]

Script: tools/audit_e1_second.py (own reader, own fail rule, query-id join; boot 100000 seed 1; delta-method CI; full sign-flip enumeration). First method: tools/e1_equal_compute.py, tools/dpp_vs_cube_k100.py (boot 10000 seed 0). Tolerance 0.05 pt.

| contrast | flights | flat | fails a / b | diff pt | boot CI (2nd) | delta CI | flights neg/pos/zero | sign p | 1st method | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| E1: cube0.5 K400 - dpp0.5 K100 | 13 | 2075 | 57 / 62 | -0.24 | [-1.58, +1.47] | [-1.85, +1.37] | 7 / 1 / 5 | 0.8828 | -0.24 [-1.59, +1.45], p 0.8828 | PASS |
| dpp0.5 K100 - cube0.5 K100 (close-out) | 13 | 2075 | 62 / 95 | -1.59 | [-3.45, -0.26] | [-3.30, +0.12] | 7 / 2 / 4 | 0.0508 | -1.59 [-3.46, -0.25], p 0.0508 | PASS |
| cube0.5 K400 - dpp0.5 K400 | 13 | 2075 | 57 / 64 | -0.34 | [-1.05, +0.31] | [-1.07, +0.39] | 5 / 2 / 6 | 0.5781 | -0.34 [-1.03, +0.32], p 0.5781 | PASS |

Verdict: **PASS** (diff and both bootstrap bounds within 0.05 pt, same signs, sign p equal to 3 decimals).
Delta CI is a second interval shape (normal approximation on 13 flights), shown for context, not part of the tolerance test.

## Reading (engineer 2026-11-27)
- Pre-check: working-tree results/leaderboard.json equals HEAD except the 'generated' stamp (parsed JSON compare) -> 4-row worktree audit of 26 Nov still holds.
- E1 loses the '1 method' tag: -0.24 [-1.58, +1.47] vs [-1.59, +1.45]; tie.
- K100 close-out line reproduces (-1.59 [-3.45, -0.26], p 0.0508). Catch: the delta-method interval [-3.30, +0.12] includes 0. So "the CI excludes 0" holds for the percentile bootstrap only; with 13 flights (one, hospital_easy/P006, carries 55 % of the gap) the interval shape decides it. Print it as "not confirmed", as the flight sign test already says.
