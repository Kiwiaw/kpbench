# C43 K interaction, second method (2026-11-15 16:04): PASS

Script tools/c43_k_check.py (raw gz rows + label json, join by query id, convolution sign-flip, multinomial-weight bootstrap 20000, seed 7; CI tolerance 0.15 pt, counts exact). First method: tools/k_interaction.py -> results/main/k_interaction_fresh1.md.

| quantity | expected (14 Nov) | recomputed | match |
|---|---|---|---|
| n | 2075 | 2075 | yes |
| c100 | 95 | 95 | yes |
| i100 | 62 | 62 | yes |
| c400 | 57 | 57 | yes |
| i400 | 64 | 64 | yes |
| inter | -40 | -40 | yes |
| lo | -3.35 | -3.36 | yes |
| hi | -0.66 | -0.66 | yes |
| neg | 7 | 7 | yes |
| pos | 0 | 0 | yes |
| p | 0.0156 | 0.0156 | yes |

Interaction -40 of 2075 = -1.93 pt; cube0.5 - dpp0.5 at K100 +33, at K400 -7.

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
