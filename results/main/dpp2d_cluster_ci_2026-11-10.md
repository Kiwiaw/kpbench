# Contribution 2: pixel dpp2d120a1 vs 3D dpp0.5 (K100) on FRESH-1: flight-cluster CI next to the frame CI (2026-11-10 13:27)

Rows: local mirror of raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh and raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh (14 FRESH-1 flights, 13440 queries, 2075 flat). Cluster bootstrap: 2000 resamples of the 13 flights with flat frames, seed 22 (tools/class_bootstrap.cluster_ci); the default 1000 / seed 0 is given too.

| pair | flat fails cand / inc | frame 20-frame block CI (gate json) | flights better / worse / tied | cluster CI 2000 x seed 22 | cluster CI 1000 x seed 0 | sign p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| dpp2d120a1 vs dpp0.5, K100 | 119 / 62 | +2.75 [+1.08, +4.90] | 3 / 5 / 5 | +2.75 [+0.05, +5.58] | +2.75 [+0.19, +5.82] | 0.855 | 0.945 |
| E1 cube0.5 vs sd, K100 (reference) | 95 / 148 | -2.55 [-4.80, -0.34] | 5 / 3 / 5 | -2.55 [-5.43, +0.82] | -2.55 [-5.36, +0.82] | 0.363 | 0.230 |

Cluster CI half-width for the pair dpp2d120a1 vs dpp0.5, K100: 2.76 pt (kill line 2 pt: MET -> not decidable at flight level on FRESH-1).

## Per-flight flat fails, dpp2d120a1 vs dpp0.5, K100

| flight | n flat | dpp2d120a1 fails | dpp0.5 fails | diff | dpp2d120a1 % | dpp0.5 % |
|---|---|---|---|---|---|---|
| oldtown_easy/P002 | 301 | 34 | 19 | +15 | 11.30 | 6.31 |
| oldtown_easy/P005 | 60 | 0 | 0 | +0 | 0.00 | 0.00 |
| oldtown_easy/P001 | 338 | 12 | 0 | +12 | 3.55 | 0.00 |
| oldtown_easy/P000 | 121 | 0 | 0 | +0 | 0.00 | 0.00 |
| oldtown_easy/P007 | 80 | 0 | 0 | +0 | 0.00 | 0.00 |
| hospital_easy/P025 | 221 | 1 | 2 | -1 | 0.45 | 0.90 |
| hospital_easy/P004 | 124 | 4 | 6 | -2 | 3.23 | 4.84 |
| hospital_easy/P006 | 237 | 46 | 14 | +32 | 19.41 | 5.91 |
| hospital_easy/P019 | 111 | 0 | 0 | +0 | 0.00 | 0.00 |
| factory_hard/P011 | 99 | 2 | 2 | +0 | 2.02 | 2.02 |
| factory_hard/P009 | 177 | 13 | 11 | +2 | 7.34 | 6.21 |
| factory_hard/P010 | 20 | 3 | 2 | +1 | 15.00 | 10.00 |
| factory_easy/P005 | 186 | 4 | 6 | -2 | 2.15 | 3.23 |
| **total** | **2075** | **119** | **62** | **+57** | | |

Flights with 0 flat frames are not rows (oldtown_easy P004). Descriptive: FRESH-1 has been read 10x, CIs are nominal and unadjusted; K400 cells stay [T] until the 148 = 148 audit (results/audit/r29_148_prep_2026-11-09.md).

Reading (engineer 10 Nov, dispatch 12:30 MAIN, pre-registered kill 'cluster CI covers 0'): the pixel-cap arm fails MORE (119 vs 62 of 2075), so the sign here is 3D better; the frame CI +2.75 [+1.08, +4.90] is quoted from the agent note results/main/dpp2d_fresh1.md (the FRESH-1 gate json is not mirrored locally). The flight-cluster CI +2.75 [+0.05, +5.58] (2000 x seed 22) and [+0.19, +5.82] (1000 x seed 0) both exclude 0 by a hair; the sign test is 5 flights for 3D, 3 against, 5 tied (two-sided p 0.73) and the one-sided Wilcoxon for 3D-lower on the 8 untied flights is p 0.137: the gain rests on hospital_easy/P006 (+32) and oldtown_easy/P002 (+15). Kill not met, but only just: the wording stays 'per-frame significant, flight-level borderline (cluster CI excludes 0 at the 2.5 % tail, sign and Wilcoxon do not)'; no verdict changes, no submit. Three FRESH-1 pairs now carry frame + cluster CIs (K400 cube0.5 vs dpp0.5, E1, dpp2d).
