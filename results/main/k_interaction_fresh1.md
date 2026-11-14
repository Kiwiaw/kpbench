# Picker x K interaction, cube0.5 vs dpp0.5, FRESH-1 flat (2026-11-14 14:17)

Rows: raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400, raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400, raco_cube0.5_K100_same_lg_g-cube0.5-fresh, raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh. 13 flights with flat frames, 2075 flat. Script: tools/k_interaction.py (boot 10000, seed 0).

Flat fails of 2075: K400 cube0.5 57 / dpp0.5 64; K100 cube0.5 95 / dpp0.5 62.

| quantity | value |
|---|---|
| cube0.5 - dpp0.5 at K400 | -7 frames = -0.34 pt |
| cube0.5 - dpp0.5 at K100 | +33 frames = +1.59 pt |
| interaction (K400 - K100) | -40 frames = -1.93 pt |
| flight-cluster 95 % CI | [-3.35, -0.66] pt |
| flights neg / pos / zero | 7 / 0 / 6 |
| exact sign-flip p2 (counts) | 0.01562 |

| flight | n flat | K400 c / i | K100 c / i | d |
|---|---|---|---|---|
| oldtown_easy/P002 | 301 | 16 / 23 | 21 / 19 | -9 |
| oldtown_easy/P005 | 60 | 0 / 0 | 0 / 0 | +0 |
| oldtown_easy/P001 | 338 | 0 / 0 | 0 / 0 | +0 |
| oldtown_easy/P000 | 121 | 0 / 0 | 0 / 0 | +0 |
| oldtown_easy/P007 | 80 | 0 / 0 | 0 / 0 | +0 |
| hospital_easy/P025 | 221 | 1 / 2 | 1 / 2 | +0 |
| hospital_easy/P004 | 124 | 5 / 4 | 13 / 6 | -6 |
| hospital_easy/P006 | 237 | 28 / 25 | 32 / 14 | -15 |
| hospital_easy/P019 | 111 | 0 / 0 | 2 / 0 | -2 |
| factory_hard/P011 | 99 | 0 / 1 | 1 / 2 | +0 |
| factory_hard/P009 | 177 | 6 / 7 | 12 / 11 | -2 |
| factory_hard/P010 | 20 | 0 / 0 | 3 / 2 | -1 |
| factory_easy/P005 | 186 | 1 / 2 | 10 / 6 | -5 |
