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
| K400 sd (gate 8 Nov, mirrored 11 Nov) | 13440 | 2075 | 148 | 7.13 | 148 | yes |

**all counts reproduce r28** (labels 13440/2075 vs 13440/2075)
