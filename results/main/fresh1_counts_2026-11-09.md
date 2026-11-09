# FRESH-1 flat-fail recount from mirrored rows (tools/fresh_counts.py, laptop)

Labels: 14 flights, 13440 queries, 2075 flat (plane_frac >= 0.85). Fail = no pose or > 25 cm or > 2 deg, counted by hand.

| run | rows | flat rows | flat fails | flat fail % | r28 | match |
|---|---|---|---|---|---|---|
| K400 cube0.5 | 13440 | 2075 | 57 | 2.75 | 57 | yes |
| K400 dpp0.5 | 13440 | 2075 | 64 | 3.08 | 64 | yes |
| K100 dpp0.5 | 13440 | 2075 | 62 | 2.99 | 62 | yes |
| K100 dppr0.317 | 13440 | 2075 | 67 | 3.23 | 67 | yes |
| E1 cube0.5 | 13440 | 2075 | 95 | 4.58 | 95 | yes |
| E1 sd | 13440 | 2075 | 148 | 7.13 | 148 | yes |
| dpp2d gate inc dpp0.5 | 13440 | 2075 | 62 | 2.99 | 62 | yes |
| dpp2d120a1 (5.73 %) | 13440 | 2075 | 119 | 5.73 | - | yes |

**all counts reproduce r28** (labels 13440/2075 vs 13440/2075)

## Engineer 2026-11-09 13:00 (dispatch 13:12 MAIN step, laptop only)
| run | rows | flat rows | flat fails | note |
|---|---|---|---|---|
| sd K400 (Qg-cube0.5-vs-sd-K400-fresh, jobs 20220547/48, 15/15 COMPLETED 8 Nov) | - | - | 148 (gate json only) | NOT MIRRORED: ssh euler timeout 13:00 9 Nov (ETH VPN down); rows exist only on Euler; the 148 = 148 audit laptop half is results/audit/r29_148_prep_2026-11-09.md |

The 8 local cells above reproduce r28 again today (57/64/62/67/95/148/62/119 of 2075 flat, 13440 rows each, 18 run dirs). squeue not readable (no ssh); last known state 10:42 empty.
