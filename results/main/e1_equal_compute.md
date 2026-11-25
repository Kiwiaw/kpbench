# E1 equal-compute check, cube0.5 vs dpp0.5, FRESH-1 flat (2026-11-25 13:15) [P]

Rows: raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400, raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400, raco_cube0.5_K100_same_lg_g-cube0.5-fresh, raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh. 13 flights with flat frames, 2075 flat. Script: tools/e1_equal_compute.py (boot 10000, seed 0). Sign: first minus second, negative = first fails less.

Flat fails of 2075: cube0.5 K100 95, K400 57; dpp0.5 K100 62, K400 64.

| pair | frames | points | flight-cluster 95 % CI | flights neg / pos / zero | exact sign-flip p2 |
|---|---|---|---|---|---|
| cube0.5 K400 - dpp0.5 K100 | -5 | -0.24 | [-1.59, +1.45] | 7 / 1 / 5 | 0.8828 |
| cube0.5 K100 - dpp0.5 K100 | +33 | +1.59 | [+0.28, +3.45] | 2 / 7 / 4 | 0.05078 |
| cube0.5 K400 - dpp0.5 K400 | -7 | -0.34 | [-1.03, +0.32] | 5 / 2 / 6 | 0.5781 |

| flight | n flat | cube K100 / K400 | dpp K100 / K400 |
|---|---|---|---|
| oldtown_easy/P002 | 301 | 21 / 16 | 19 / 23 |
| oldtown_easy/P005 | 60 | 0 / 0 | 0 / 0 |
| oldtown_easy/P001 | 338 | 0 / 0 | 0 / 0 |
| oldtown_easy/P000 | 121 | 0 / 0 | 0 / 0 |
| oldtown_easy/P007 | 80 | 0 / 0 | 0 / 0 |
| hospital_easy/P025 | 221 | 1 / 1 | 2 / 2 |
| hospital_easy/P004 | 124 | 13 / 5 | 6 / 4 |
| hospital_easy/P006 | 237 | 32 / 28 | 14 / 25 |
| hospital_easy/P019 | 111 | 2 / 0 | 0 / 0 |
| factory_hard/P011 | 99 | 1 / 0 | 2 / 1 |
| factory_hard/P009 | 177 | 12 / 6 | 11 / 7 |
| factory_hard/P010 | 20 | 3 / 0 | 2 / 0 |
| factory_easy/P005 | 186 | 10 / 1 | 6 / 2 |

## Reading (engineer 2026-11-25, dispatch 13:15 MAIN, T20 step E1)
- Selection cost (results/main/t55_lazy_time.md, same laptop run, ratios only): cube0.5 time is flat in K (K100 and K400 within 2 % of each other), dpplazy0.5 at K100 is 2.2x cube0.5 at K100. So cube0.5 at K400 costs LESS selection time than dpp0.5 at K100; "affordable K" for cubes is any K <= 400, not ~K200-250.
- Equal selection time: cube0.5 K400 fails 57 of 2075 flat, dpp0.5 K100 62: -5 frames = -0.24 pt [-1.59, +1.45], 7 / 1 flights, p 0.88. A tie, point estimate favouring cubes.
- Dispatch kill "cube0.5 at the affordable K fails <= 62 of 2075 flat" = MET (57 <= 62). Headline becomes "equal-K only": dpp0.5 beats cube0.5 at the same K100 (+1.59 [+0.28, +3.45] for cubes, p 0.0508), not at equal selection time.
- Caveat: matcher (LightGlue) and pose time grow with K and were not timed; at equal TOTAL compute K400 cubes cost more than K100 DPP. The fair framing is "at a fixed keypoint budget", which is the benchmark's own protocol.
- Check: counts 95 / 57 / 62 / 64 reproduce results/main/k_interaction_fresh1.md and the K table (2026-11-21). FRESH-1 is post hoc (selection seen), tag [P].
