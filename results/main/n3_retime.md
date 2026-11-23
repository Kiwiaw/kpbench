# N3 re-time: dpp0.5 selection with identical picks, 6 laptop flights (MAIN, engineer 2026-11-23, dispatch 2026-11-23T13:15) [P]

Question (objection N3): is the 34.5 ms per frame of dpp0.5 at K100 (results/main/select_time_2026-11-22.md) a cost of the method or of our code?
Answer: of the code. The same greedy, returning the same picks on 3980 of 3980 query frames, takes 2.8 ms at K100 (was 30.7 ms in the same run, 11x faster); cube0.5 takes 1.1 ms. The gap to cubes shrinks from 27x to 2.5x at K100, but still grows with K (7.5x at K200, 30x at K400).

## What was changed (tools/n3_retime.py; kpbench/pickers.py untouched, so no benchmark number moves)

- repo: `kpbench.pickers.dpp_greedy`, already the fast greedy MAP of Chen, Zhang, Zhou (NeurIPS 2018), one Cholesky row per pick, but it re-stacks all rows at every pick.
- pre: same arithmetic, rows in one preallocated array.
- lazy: pre + lazy candidate prefix (lazy-greedy idea of Minoux 1978 / Hemmi et al. NeurIPS 2022, here with a closed-form bound). The gain of candidate i can never exceed q_i^2, and q_i = exp(-rank_i / 100) falls with rank; while the best gain inside the first m candidates is >= q_m^2 no later candidate can win, so only the first m are updated; m doubles when the bound is crossed. Toy: best gain in the prefix 0.02, q_256^2 = exp(-5.12) = 0.006 -> candidates 256..1999 are skipped this pick.

## Check 1: identical picks (`python tools/n3_retime.py check`)

Ordered pick list of pre and lazy vs the repo picker, every query frame of the 6 flights cached on the laptop, K = 50, 100, 200, 400 (K100 and K400 against a direct repo call, K50 / K200 against the first K picks of the K400 call: the greedy is nested).

| flight | queries | frames with a differing pick list (pre or lazy vs repo, K 50-400) | candidates min / median / max | median final prefix m at K 50 / 100 / 200 / 400 |
|---|---|---|---|---|
| hospital/P041 | 530 | 0 | 2000 / 2000 / 2000 | 128 / 512 / 1024 / 1983 |
| oldtown/P000 | 711 | 0 | 2000 / 2000 / 2000 | 128 / 256 / 512 / 512 |
| oldtown/P001 | 794 | 0 | 2000 / 2000 / 2000 | 128 / 256 / 512 / 1024 |
| oldtown/P002 | 835 | 0 | 2000 / 2000 / 2000 | 128 / 256 / 512 / 1024 |
| oldtown/P006 | 453 | 0 | 2000 / 2000 / 2000 | 128 / 256 / 512 / 1024 |
| oldtown/P007 | 657 | 0 | 2000 / 2000 / 2000 | 128 / 256 / 256 / 512 |
| total | 3980 | 0 | | |

Kill "any pick mismatch" NOT met: 0 of 3980 frames differ. The prefix column shows why lazy is fast: at K100 the greedy only ever looks at the strongest 256-512 of 2000 candidates.

## Check 2: time (`python tools/n3_retime.py time --every 20`)

Picker call only (no detector, matcher, pose), pickers and K interleaved per query, all four in pure numpy / Python on one CPU. 201 queries (every 20th query of 6 flights), fastest of 3 repeats, median ms per frame (90th percentile in brackets). Machine: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 20 logical cores, numpy 2.4.2, Python 3.14.2.

| K | dpp0.5 repo | dpp0.5 pre | dpp0.5 lazy | cube0.5 | repo / lazy | lazy / cube0.5 | repo / cube0.5 |
|---|---|---|---|---|---|---|---|
| 50 | 6.03 (9.22) | 4.00 (7.11) | 0.95 (1.92) | 1.06 (1.83) | 6.3x | 0.9x | 5.7x |
| 100 | 30.67 (45.55) | 10.12 (22.97) | 2.80 (5.68) | 1.13 (1.88) | 11.0x | 2.5x | 27.2x |
| 200 | 142.68 (216.30) | 29.44 (71.89) | 11.21 (21.72) | 1.50 (1.97) | 12.7x | 7.5x | 95.4x |
| 400 | 493.66 (794.33) | 62.39 (120.62) | 32.32 (65.77) | 1.07 (1.89) | 15.3x | 30.1x | 460.1x |

Kill "K100 still >= 34.5 ms" NOT met: 2.80 ms.

## Reading

- For the paper: at K100 dpp0.5 selection costs about 3 ms per frame with the standard fast greedy plus a lazy prefix, 2.5x the cube cap (1.1 ms); "19x" was our unoptimised loop. Both are small against detection and matching.
- The growth with K stays: lazy is 0.95 / 2.8 / 11.2 / 32.3 ms at K50 / 100 / 200 / 400 while cubes stay near 1 ms. At 10 frames/s (100 ms per frame) dpp0.5 now fits at every tested K (was: K100 only).
- The lazy bound needs the rank-decaying quality q; it carries over to dppr only if its quality term is bounded the same way (not tested).
- Limits: one laptop CPU, 201 timed frames, every frame has 2000 candidates, 6 flights from 2 environments (5 oldtown, 1 hospital), numpy code for both pickers; repo time here 30.7 ms vs 34.5 ms yesterday (other flights, same machine). No pose, no fail number. Tag [P]. The benchmark picker is unchanged; swapping lazy into kpbench/pickers.py would need the SUB2 identity check on all 31 flights first.
