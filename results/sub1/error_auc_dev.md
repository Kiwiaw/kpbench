# Error magnitude on dev flat queries, n = 936 (leaderboard runs, not gate runs)

| picker | fail % | median cm | median deg | AUC 5 cm | AUC 10 cm | AUC 25 cm | fails no pose | fails bad pose |
|---|---|---|---|---|---|---|---|---|
| dpp0.5 | 14.42 (135) | 0.59 | 0.060 | 0.714 | 0.779 | 0.825 | 20 | 115 |
| cube0.5 | 15.60 (146) | 0.70 | 0.074 | 0.688 | 0.759 | 0.811 | 48 | 98 |
| grid | 19.12 (179) | 0.58 | 0.065 | 0.695 | 0.749 | 0.787 | 46 | 133 |
| sd | 22.22 (208) | 0.78 | 0.079 | 0.635 | 0.702 | 0.750 | 43 | 165 |

Per-flight median position error (cm); pair = left better / left worse / tied, of flights with flat queries:

| pair | median better/worse/tied | flights |
|---|---|---|
| dpp0.5 vs cube0.5 | 10/2/0 | 12 |
| cube0.5 vs grid | 2/10/0 | 12 |
| grid vs sd | 10/1/1 | 12 |
| cube0.5 vs sd | 7/5/0 | 12 |
| dpp0.5 vs sd | 9/3/0 | 12 |

Pre-registered (dispatch 2026-10-30T13:40), checked in this order:
- median order dpp0.5 < cube0.5 < grid < sd: NO (0.59 / 0.70 / 0.58 / 0.78 cm)
- each neighbouring pair per-flight >= 10 better (of 12 flights): NO
- median separates cube0.5 from grid (>= 10 flights): NO (2)
- gain sits in bad pose, not no pose (sd - dpp0.5): yes (bad pose 50, no pose 23 frames)
- verdict: KILL: uniformity does not lower error magnitude by this rule; SUB1 keeps only Q210

Notes:
- `python tools/error_magnitude.py`. dppr0.317 is absent: its rows are on Euler. Rows carry err_cm, rot_deg; None = no pose, no schema change needed.
- Post-hoc, not part of the rule: AUC 25 cm keeps the fail order (0.825 / 0.811 / 0.787 / 0.750) but it mostly restates the fail count at the same threshold; AUC 5 cm already flips cube0.5 and grid (0.688 vs 0.695).
- grid has the best median (0.58 cm) and a worse fail % than cube0.5: on flat frames spread moves the tail, not the typical frame.
