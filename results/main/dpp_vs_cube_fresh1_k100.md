# dpp0.5 vs cube0.5, FRESH-1 K100 flat, flight-level replication test (MAIN, dispatch 2026-11-15 16:05)

PRE-REGISTRATION (written and committed before the test is computed): one test, alpha 0.05, two-sided; statistic = flight-cluster bootstrap 95 % CI of the pooled flat fail diff (dpp0.5 - cube0.5, pt) + exact two-sided sign-flip p of the per-flight count diffs (zeros dropped) + flights better/tie/worse. Counts already seen (62 vs 95 of 2075, not blind); this test is kept OUTSIDE the 31-flight Holm family (C41). Rows: raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh, raco_cube0.5_K100_same_lg_g-cube0.5-fresh. Kill: CI covers 0 or p >= 0.05 -> "DPP beats cubes" stays "lower, not flight-significant". Tag [P] until C43 PASS (C43 PASS 2026-11-15, results/audit/c43_k_interaction.md, so tag [T] if the recount = 62/95).

## Result (tools/dpp_vs_cube_k100.py, boot 10000, seed 0) [T: recount 62/95 = C43 cells]
- 13 flights with flat frames (0 of the 14 FRESH-1 flights are among the 31 published), 2075 flat.
- dpp0.5 62 vs cube0.5 95: -33 frames = -1.59 pt, flight-cluster CI [-3.46, -0.25], dpp0.5 better/tie/worse 7/4/2, exact two-sided sign-flip p 0.0508.
- Pre-registered kill MET (p 0.0508 >= 0.05; the CI alone excludes 0): "DPP beats cubes" stays "fewer fails, not flight-significant"; method claim rests on vs grid (Holm 0.018) and vs sd (0.015). Not re-tested on FRESH-1.
- Concentration: hospital_easy/P006 alone carries -18 of -33 (55 %).

| flight | n flat | dpp0.5 | cube0.5 | d |
|---|---|---|---|---|
| oldtown_easy/P002 | 301 | 19 | 21 | -2 |
| oldtown_easy/P005 | 60 | 0 | 0 | +0 |
| oldtown_easy/P001 | 338 | 0 | 0 | +0 |
| oldtown_easy/P000 | 121 | 0 | 0 | +0 |
| oldtown_easy/P007 | 80 | 0 | 0 | +0 |
| hospital_easy/P025 | 221 | 2 | 1 | +1 |
| hospital_easy/P004 | 124 | 6 | 13 | -7 |
| hospital_easy/P006 | 237 | 14 | 32 | -18 |
| hospital_easy/P019 | 111 | 0 | 2 | -2 |
| factory_hard/P011 | 99 | 2 | 1 | +1 |
| factory_hard/P009 | 177 | 11 | 12 | -1 |
| factory_hard/P010 | 20 | 2 | 3 | -1 |
| factory_easy/P005 | 186 | 6 | 10 | -4 |
