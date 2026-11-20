# Power of the FRESH-2 single endpoint, dpp0.5 vs cube0.5 K100 flat (MAIN, engineer 2026-11-19, dispatch 2026-11-19T13:15)

Script: `python tools/power_fresh2.py --reps 10000 --seed 0` (and `--seed 1 --targets 2774,4849`). Status: [T] since 2026-11-20 (second method tools/power_fresh2_check.py, 24/24 cells within 1.0 pt; results/audit/2026-11-18.md addendum).

## Answer
- Kill rule "power < 50 % at 4849 flat" is **not met**: 93-99 % if the true gap equals the gap seen so far, 75-80 % if it is only half of it.
- Caveat on the 4849: the registered endpoint (fresh2_k100_endpoint_prereg.md) uses FRESH-2 frames only, about 2774 flat; 4849 = FRESH-1 2075 + FRESH-2 2774 pooled. At 2774 flat the power is 69-89 % (gap as seen) and 49-60 % (half gap).
- Flat frames needed for 80 %: about 3000-3500 (gap as seen, two-stage), about 5000-5500 (half gap). The same-env FRESH-2 pool alone (2774) is short of both.
- Reading: FRESH-2 is worth running if the gap seen so far is real; if the true gap is half, the registered FRESH-2-only test is a coin flip (49-60 %).

## What was simulated
- Test = the registered one: per flight d_f = dpp0.5 - cube0.5 flat fails, exact one-sided sign-flip on d_f (zeros dropped), reject at p < 0.05. Toy: flights d = (-5, -3, +1) -> 8 sign patterns, 2 have sum <= -7 -> p 0.25, no reject.
- A future set = whole flights drawn with replacement from flights already run until the flat count reaches the target (unit = flight). Sources: FRESH-1 (13 flights with flat frames, 62 vs 95 of 2075), board (27 flights, 192 vs 227 of 1813), pooled (40).
- "flights" = drawn flights keep their observed d_f. "two-stage" = the queries inside each drawn flight are redrawn too (a new flight like it, not the same frames). "gap x 0.5" = every flight's true gap halved first (the observed gap is a noisy estimate, selected after two near-0.05 reads). "gap x 0.0" = no true gap: must reject <= 5 % (simulator check: 1.9-4.8 %, OK).
- Inputs come from the row files (tools/j7_lofo_check.flight_counts) and were re-verified today (results/audit/2026-11-18.md addendum: sums and LOFO PASS).
- The registered one-sided test applied to the existing sets (descriptive, already-read data): FRESH-1 p 0.0254, board p 0.0193, pooled p 0.0016.

## Table (power in %, 10000 sets per cell, seed 0, Monte Carlo SE <= 0.5 pt)
| source | resampling | 2075 flat | 2774 flat | 4150 flat | 4849 flat | 6000 flat | 8000 flat | 12000 flat | flights at 2774 / 4849 |
|---|---|---|---|---|---|---|---|---|---|
| board | flights | 78.4 | 88.1 | 96.9 | 98.5 | 99.5 | 100.0 | 100.0 | 42 / 73 |
| board | two-stage | 67.2 | 78.1 | 90.4 | 94.0 | 97.1 | 99.4 | 100.0 | 42 / 73 |
| board | two-stage, gap x 0.5 | 49.9 | 58.7 | 73.4 | 78.8 | 86.4 | 93.5 | 98.7 | 42 / 73 |
| board | two-stage, gap x 0.0 | 3.5 | 3.6 | 4.2 | 3.9 | 4.1 | 4.3 | 4.1 | 42 / 73 |
| fresh1 | flights | 75.0 | 89.0 | 98.4 | 99.5 | 99.9 | 100.0 | 100.0 | 18 / 31 |
| fresh1 | two-stage | 52.8 | 69.4 | 88.2 | 93.1 | 97.2 | 99.5 | 100.0 | 18 / 31 |
| fresh1 | two-stage, gap x 0.5 | 35.3 | 49.0 | 68.7 | 75.7 | 83.8 | 93.0 | 98.7 | 18 / 31 |
| fresh1 | two-stage, gap x 0.0 | 1.9 | 2.7 | 3.3 | 3.7 | 3.8 | 4.6 | 4.2 | 18 / 31 |
| pooled | flights | 74.8 | 85.4 | 96.3 | 97.8 | 99.5 | 99.9 | 100.0 | 29 / 51 |
| pooled | two-stage | 62.1 | 74.4 | 89.5 | 93.4 | 97.0 | 99.3 | 100.0 | 29 / 51 |
| pooled | two-stage, gap x 0.5 | 45.8 | 55.4 | 71.2 | 77.5 | 85.2 | 92.9 | 98.4 | 29 / 51 |
| pooled | two-stage, gap x 0.0 | 2.4 | 3.2 | 3.8 | 3.9 | 4.0 | 4.0 | 4.5 | 29 / 51 |

Seed 1 at 2774 / 4849 flat: board 88.0 / 98.5, 78.6 / 94.4, half gap 59.9 / 80.1; fresh1 88.9 / 99.3, 69.7 / 93.7, half gap 49.0 / 75.3; pooled 86.1 / 98.1, 73.9 / 93.2, half gap 55.5 / 78.0. Seeds differ by <= 1.3 pt; no cell crosses 50 % between seeds except fresh1 half gap at 2774 (49.0 both seeds, on the line).

## Limits
- Plug-in power: it assumes future flights look like the 13 / 27 already run. FRESH-1 has only 9 flights with a nonzero d_f and one flight (hospital_easy/P006) carries -18 of -33; the fresh1 rows are the least stable.
- Flight sizes are copied from the source (FRESH-1 mean 160 flat per flight, board 67), so "flights at 2774" differs by source (18 vs 42); the real FRESH-2 list is not frozen yet.
- No number here is a result on FRESH-2; nothing was run on Euler (FREEZE).
