# J7 part 1: per-flight forest, dpp0.5 vs cube0.5 K100 flat fails + register of tests (SUB2, engineer 2026-11-17, dispatch 2026-11-17T20:10)

Script: `python tools/j7_forest.py --set board|fresh1` (raw row files + label json, fail = err None or > 25 cm or > 2 deg, flat = plane_frac >= 0.85; per-flight CI = paired query bootstrap, 5000, seed 0). Plots: results/main/j7_forest_board_k100.png, results/main/j7_forest_fresh1_k100.png.

## Verdict
- Kill check (sums must equal published counts): **PASS both sets**; kpbench.audit C40 (independent code) gives the same board counts and p2 0.03857 (reports/audit_2026-11-17_C1-C2-C6-C7-C8-C9-C10-C11-C15-C40-C41.md, 0 FAIL, C41 WARN = Holm wording, C3/C4/C5/C12/C13 skipped: no laptop flight data). 31 leaderboard flights (27 with flat frames): 192 vs 227 of 1813 = -1.93 pt, two-sided sign-flip 0.0386 (= ledger row 3 0.039, Holm 0.077), 10/2 flights. FRESH-1 (13 flights): 62 vs 95 of 2075 = -1.59 pt, p2 0.0508 (= dpp_vs_cube_fresh1_k100.md).
- Leave-one-flight-out: board p2 ranges 0.003 (drop oldtown/P007, +6 against DPP) to 0.077 (drop hospital/P043); FRESH-1 0.031 to 0.102, pooled diff -0.82 pt without hospital_easy/P006 (carries -18 of -33). The vs-cubes gap is real in direction on both sets but rests on 2-3 flights each: "fewer fails, not flight-significant" wording stays.

## Tables
```
sums: flat 1813, dpp0.5 192, cube0.5 227, -1.93 pt, p2 0.0386 -> MATCH
| flight | n flat | dpp0.5 | cube0.5 | diff pt | 95 % CI (queries) | pooled without it |
|---|---|---|---|---|---|---|
| factory/P000 | 16 | 3 | 5 | -12.50 | [-31.25, +0.00] | -1.84 (p2 0.051) |
| factory/P001 | 102 | 5 | 7 | -1.96 | [-4.90, +0.00] | -1.93 (p2 0.051) |
| factory/P002 | 1 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.93 (p2 0.039) |
| factory/P003 | 41 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.98 (p2 0.039) |
| factory/P004 | 17 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.95 (p2 0.039) |
| factory/P005 | 2 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.93 (p2 0.039) |
| factory/P006 | 49 | 7 | 8 | -2.04 | [-6.12, +0.00] | -1.93 (p2 0.046) |
| factory/P007 | 46 | 3 | 8 | -10.87 | [-21.74, -2.17] | -1.70 (p2 0.069) |
| factory/P008 | 30 | 1 | 1 | +0.00 | [-10.00, +10.00] | -1.96 (p2 0.039) |
| hospital/P037 | 11 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.94 (p2 0.039) |
| hospital/P040 | 8 | 5 | 7 | -25.00 | [-62.50, +25.00] | -1.83 (p2 0.051) |
| hospital/P041 | 153 | 40 | 46 | -3.92 | [-9.15, +0.65] | -1.75 (p2 0.074) |
| hospital/P042 | 13 | 10 | 10 | +0.00 | [+0.00, +0.00] | -1.94 (p2 0.039) |
| hospital/P043 | 204 | 22 | 32 | -4.90 | [-8.33, -1.47] | -1.55 (p2 0.077) |
| hospital/P044 | 4 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.93 (p2 0.039) |
| hospital/P046 | 102 | 5 | 12 | -6.86 | [-12.75, -1.96] | -1.64 (p2 0.076) |
| hospital/P047 | 13 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.94 (p2 0.039) |
| hospital/P048 | 54 | 1 | 3 | -3.70 | [-9.26, +0.00] | -1.88 (p2 0.051) |
| oldtown/P000 | 156 | 0 | 0 | +0.00 | [+0.00, +0.00] | -2.11 (p2 0.039) |
| oldtown/P001 | 240 | 49 | 48 | +0.42 | [-2.50, +3.75] | -2.29 (p2 0.031) |
| oldtown/P002 | 86 | 0 | 5 | -5.81 | [-11.63, -1.16] | -1.74 (p2 0.069) |
| oldtown/P003 | 152 | 12 | 12 | +0.00 | [-2.63, +2.63] | -2.11 (p2 0.039) |
| oldtown/P004 | 32 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.97 (p2 0.039) |
| oldtown/P005 | 33 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.97 (p2 0.039) |
| oldtown/P006 | 75 | 0 | 0 | +0.00 | [+0.00, +0.00] | -2.01 (p2 0.039) |
| oldtown/P007 | 104 | 29 | 23 | +5.77 | [+0.00, +11.54] | -2.40 (p2 0.003) |
| oldtown/P008 | 69 | 0 | 0 | +0.00 | [+0.00, +0.00] | -2.01 (p2 0.039) |
wrote results\main\j7_forest_board_k100.png

sums: flat 2075, dpp0.5 62, cube0.5 95, -1.59 pt, p2 0.0508 -> MATCH
| flight | n flat | dpp0.5 | cube0.5 | diff pt | 95 % CI (queries) | pooled without it |
|---|---|---|---|---|---|---|
| oldtown_easy/P002 | 301 | 19 | 21 | -0.66 | [-3.32, +1.99] | -1.75 (p2 0.094) |
| oldtown_easy/P005 | 60 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.64 (p2 0.051) |
| oldtown_easy/P001 | 338 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.90 (p2 0.051) |
| oldtown_easy/P000 | 121 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.69 (p2 0.051) |
| oldtown_easy/P007 | 80 | 0 | 0 | +0.00 | [+0.00, +0.00] | -1.65 (p2 0.051) |
| hospital_easy/P025 | 221 | 2 | 1 | +0.45 | [-0.90, +2.26] | -1.83 (p2 0.031) |
| hospital_easy/P004 | 124 | 6 | 13 | -5.65 | [-10.48, -0.81] | -1.33 (p2 0.102) |
| hospital_easy/P006 | 237 | 14 | 32 | -7.59 | [-10.97, -4.22] | -0.82 (p2 0.102) |
| hospital_easy/P019 | 111 | 0 | 2 | -1.80 | [-4.50, +0.00] | -1.58 (p2 0.094) |
| factory_hard/P011 | 99 | 2 | 1 | +1.01 | [+0.00, +3.03] | -1.72 (p2 0.031) |
| factory_hard/P009 | 177 | 11 | 12 | -0.56 | [-2.82, +1.69] | -1.69 (p2 0.070) |
| factory_hard/P010 | 20 | 2 | 3 | -5.00 | [-15.00, +0.00] | -1.56 (p2 0.070) |
| factory_easy/P005 | 186 | 6 | 10 | -2.15 | [-4.84, +0.00] | -1.54 (p2 0.102) |
wrote results\main\j7_forest_fresh1_k100.png
```

## Register of flight-level tests read so far (from results/sub4/ledger.md, gate files, this file)
| # | comparison | frames | flight p (two-sided) | family | source |
|---|---|---|---|---|---|
| 1 | dpp0.5 vs grid flat | 31 flights, 1813 | 0.0061 | Holm 4 -> 0.018 | results/main/j2b_dpp_vs_grid.md |
| 2 | dpp0.5 vs cube0.5 flat | 31, 1813 | 0.0386 | Holm 4 -> 0.077 | j2b; this file |
| 3 | dpp0.5 vs sd flat | 31, 1813 | 0.0038 | Holm 4 -> 0.015 | j2b |
| 4 | dpp0.5 vs cube0.5 very flat | 31, 1058 | 0.104 | Holm 4 -> 0.104 | j2b, C41 |
| 5 | cube0.5 vs strongest all frames | 31 | 0.035 | none (page headline) | audit C9 C11 |
| 6 | cube0.5 vs grid flat | 31, 1813 | 0.457 | none | results/main/j2_cube_vs_grid.md |
| 7 | dpp2d120a1 vs dpp0.5 flat | FRESH-1, 2075 | 0.195 | none | results/audit/r30_dpp2d_fresh1.md |
| 8 | picker x K interaction (K100 - K400) | FRESH-1, 2075 | 0.0156 | none | results/main/k_interaction_fresh1.md, C43 |
| 9 | dpp0.5 vs cube0.5 K100 flat | FRESH-1, 2075 | 0.0508 | none (prereg 15 Nov) | results/main/dpp_vs_cube_fresh1_k100.md |
| 10 | held-out swap dpp0.5 vs cube0.5 | HELD_OUT, 877 | gate route (6/0) | gate rule | gate file, C15 |
- 10 flight-level tests on 4 frame sets; FRESH-1 has been read by >= 10 gate/analysis passes (SUB4 ed. 21). Gate-file DEV/FRESH verdicts (stats.py rule) are not in this list: they are rule-decided, not claims.
- Rule from here: FRESH-2 has exactly one confirmatory endpoint (results/main/fresh2_k100_endpoint_prereg.md); every other FRESH-2 read-out is labelled secondary.
