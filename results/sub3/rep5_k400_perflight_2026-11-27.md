# rep5 at K400: dpp0.5 - cube0.5 per flight, FRESH-1 flat (SUB3, 2026-11-27) [P]

Rows: raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400 vs raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400 (mirrored, committed). Frames: flat with rep5 in both, n 2063 over 13 flights. Script: tools/rep5_k400_perflight.py (boot 10000, seed 0). rep5 = share of picks re-detected within 5 px in the next frames (bench column).

Pooled: dpp0.5 0.3518 vs cube0.5 0.2455, diff +0.1063.
- paired block bootstrap (20-frame blocks): [+0.0946, +0.1181]
- flight-cluster bootstrap: [+0.0851, +0.1244]
- flights dpp0.5 higher / lower: 13 / 0 of 13 (one-sided sign p 0.0001)

Dispatch kill (interval includes 0 or < 10 of 13 flights positive): NOT MET -> K400 rep5 is a second dpp0.5 claim [P]

| flight | n flat | dpp0.5 rep5 | cube0.5 rep5 | diff |
|---|---|---|---|---|
| factory_easy_P005 | 186 | 0.2306 | 0.1782 | +0.0523 |
| factory_hard_P009 | 177 | 0.2411 | 0.1890 | +0.0521 |
| factory_hard_P010 | 20 | 0.1681 | 0.1317 | +0.0364 |
| factory_hard_P011 | 99 | 0.3066 | 0.2227 | +0.0838 |
| hospital_easy_P004 | 124 | 0.2835 | 0.2051 | +0.0784 |
| hospital_easy_P006 | 233 | 0.3830 | 0.2357 | +0.1473 |
| hospital_easy_P019 | 111 | 0.4012 | 0.2663 | +0.1349 |
| hospital_easy_P025 | 217 | 0.3921 | 0.2612 | +0.1308 |
| oldtown_easy_P000 | 121 | 0.5022 | 0.3760 | +0.1263 |
| oldtown_easy_P001 | 338 | 0.3986 | 0.2736 | +0.1250 |
| oldtown_easy_P002 | 297 | 0.3145 | 0.2194 | +0.0951 |
| oldtown_easy_P005 | 60 | 0.4360 | 0.3478 | +0.0882 |
| oldtown_easy_P007 | 80 | 0.4677 | 0.3069 | +0.1608 |

## Reading (engineer 2026-11-27)
- Equal budget checked: n_sel = 400 on every query in both runs, so the gap is not fewer picks.
- dpp0.5 keeps more repeatable picks at K400 in every flight (+0.036 to +0.161), while flat fail there is a tie (+0.34 [-0.40, +1.00]). So at K400 the claim is "same failure, higher repeatability", tagged [P] (FRESH-1 selection already seen), not a failure claim.
- Feeds SUB4: second budget for the repeatability line (1 of 4 -> 2 of 4 budgets).
