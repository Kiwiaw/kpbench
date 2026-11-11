# K dependence on FRESH-1: counts (engineer part, 11 Nov 2026; the analyst adds the reading)

All cells: 14 FRESH-1 flights, 13440 queries, 2075 flat (plane_frac >= 0.85), LightGlue, fail = no pose or > 25 cm or > 2 deg. Recount: tools/fresh_counts.py -> results/main/fresh1_counts_2026-11-11.md (9 runs, all reproduce).

| picker | K100 flat fails | K100 % | K400 flat fails | K400 % |
|---|---|---|---|---|
| sd (top score) | 148 | 7.13 | 148 | 7.13 |
| cube0.5 | 95 | 4.58 | 57 | 2.75 |
| dpp0.5 | 62 | 2.99 | 64 | 3.08 |

- sd K400 rows mirrored 11 Nov from Euler jobs 20220547/48 (run raco_sd_K400_same_lg_g-cube0.5-fresh-K400): 13440 rows, 2075 flat rows, 148 flat fails. Kill "rows != 13440" not met.
- n_sel line: sd K400 has n_sel = 400 on 13440 / 13440 rows (sd K100: 100 on 13440 / 13440; cube0.5 K400: 400 on 13440 / 13440). The sd K400 arm used the K400 budget. Kill "K100 budget on sd K400" not met.
- The two sd 148 are different frames: 88 fail at both budgets, 60 only at K100, 60 only at K400 (Jaccard 0.4231); per flight the counts move both ways, e.g. oldtown_easy/P002 47 -> 66, hospital_easy/P006 22 -> 39, factory_hard/P011 12 -> 1 (results/audit/r29_148_verdict_2026-11-11.md). Pre-registered rule 3 therefore fired; the freeze on the sd cells is SUB2's call.
- K400 gate cube0.5 vs sd: -4.39 [-7.01, -2.01] pt, flights 6/0, recomputed from rows on the Euler login node today (results/audit/euler_login_2026-11-08.md, addendum).
