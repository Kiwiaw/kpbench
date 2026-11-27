# Scout exp (b): flat fail vs selection time, FRESH-1 (MAIN, engineer 2026-11-27, dispatch 13:20) [P]

Gist: plot every committed (arm, K) point as x = selection time relative to cube0.5 at K100, y = flat fail %. At every time budget where both pickers have a point, cube0.5 is cheaper and fails no more than dpp0.5. So DPP is an upper reference at equal K only.

No rows read, no new contrast: every number is copied from the file in the last column. Post hoc [P] (FRESH-1 selection already seen). Timing = laptop, ratios only (results/main/t55_lazy_time.md). dpp0.5 time uses dpplazy0.5, which gives the same picks (0 of 3980 frames differ, results/main/t55_lazy_identity.md); the benchmark code path is 10.3x slower still at K100.

Toy for x: cube0.5 K100 takes 1.83 ms per frame, dpplazy0.5 K100 3.96 ms -> 3.96 / 1.83 = 2.2x.

## Points (FRESH-1, 13 flights, 2075 flat frames)

| arm | K | x = time / cube0.5 K100 | flat fails | y = flat fail % | source (y; x always t55_lazy_time.md) |
|---|---|---|---|---|---|
| cube0.5 | 50 | 0.99x | - | blank (no FRESH-1 run) | - |
| cube0.5 | 100 | 1.00x | 95 | 4.58 | results/main/e1_equal_compute.md |
| cube0.5 | 200 | 1.01x | - | blank | - |
| cube0.5 | 400 | 1.02x | 57 | 2.75 | results/main/e1_equal_compute.md |
| dpp0.5 | 50 | 0.71x | - | blank | - |
| dpp0.5 | 100 | 2.16x | 62 | 2.99 | results/main/e1_equal_compute.md |
| dpp0.5 | 200 | 7.27x | - | blank (run exists, no cube0.5 K200 pair) | results/main/k_table_2026-11-21.md, Missing |
| dpp0.5 | 400 | 25.5x | 64 | 3.08 | results/main/e1_equal_compute.md |
| sd | 100 | blank (not timed) | 148 | 7.13 | results/gates/raco_cube0.5_K100_..._vs_raco_sd_K100_..._fresh.json (flat_inc) |
| sd | 400 | blank (not timed) | 148 | 7.13 | results/gates/raco_cube0.5_K400_..._vs_raco_sd_K400_..._fresh.json (flat_inc) |
| grid | any | blank | - | blank (no FRESH-1 run) | - |

## Matched-time reading
- Budget about 2.2x (dpp0.5 K100): cube0.5 can run any K <= 400 within it. Best such point cube0.5 K400, 57 vs 62 fails: -0.24 pt [-1.58, +1.47] cube0.5 minus dpp0.5 (E1, now by 2 methods, results/audit/2026-11-27_e1_second_method.md). Tie.
- Budget about 25x (dpp0.5 K400): cube0.5 K400 costs 1.02x and fails 57 vs 64: dpp0.5 - cube0.5 = +0.34 [-0.40, +1.00]. Tie, point estimate favours cubes.
- Budget 1x: only cube0.5 has points.
- Dispatch kill ("dpp0.5 not below cube0.5 or grid at any matched time ratio") = **MET**. DPP is printed as an upper reference at equal K only; no further equal-time work.

## Catches
- Selection time only. LightGlue matching and PnP grow with K and were not timed, so at equal TOTAL compute cube0.5 K400 costs more than dpp0.5 K100.
- 2 of 4 arms have no x (sd, grid not timed) and grid has no FRESH-1 run, so the "curve" is 4 timed points, not a curve.
- Looked odd, checked: sd flat fails are exactly 148 of 2075 at both K100 and K400. Row check (engineer, same day): the two runs are different (n_sel 100 vs 400, 0 of 13440 rows equal in err..n_inl) and share only 88 of the 148 failed frames, so the equal count is a coincidence, not a copied run.
