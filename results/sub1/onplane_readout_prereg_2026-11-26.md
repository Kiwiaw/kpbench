# T33 read-out rule: on-plane share and depth spread per arm, FRESH-1 flat (SUB1, written 2026-11-27 before any picks exist)

Gist: when the one post-freeze picks dump exists, this is the only read of it for SUB1. Written now, with no rows, so the read cannot be tuned to the data (SUB1 died 0/12 partly on signs that flipped with the budget: pcov +0.0247 at K100 -> -0.0002 at K400).

Status: no data. FRESH-1 rows have has_picks False; the dump waits for one save_picks (+ inlier index, T59) item after the FREEZE. File name keeps the 26 Nov date the dispatch gave it.

## Quantities (per flat query, from the onplane entry, kpbench.pickarms.FIELDS)
- off = off-plane share of the picks that have depth. Toy: 100 picks with depth, 20 off the dominant plane -> off 0.20.
- dspan = depth spread of the picks, (q90 - q10) / q50 of their depths. Toy: depths 10 / 15 / 30 m at q10 / q50 / q90 -> (30 - 10) / 15 = 1.33.
- Per arm and flight: mean over that flight's flat queries. Flat = plane_frac >= 0.85 (bench label, unchanged).

## Arms, set, unit
- Arms: dpp0.5 (reference), cube0.5, dpp2d120a1. K100, lg matcher, ap3p, same map: the T59 item spec.
- Set: FRESH-1, the 13 flights with flat frames (2075 flat). Unit of independence = flight (13), not frame.
- Contrasts (2 only): cube0.5 - dpp0.5 and dpp2d120a1 - dpp0.5, for off and for dspan = 4 reads.

## Statistics
- Paired block bootstrap of the per-query difference (kpbench.stats.paired_diff, 20-frame blocks, 10000 draws, seed 0), 95 % percentile CI.
- Flight sign count: flights with a positive / negative per-flight mean difference, exact two-sided sign-flip p over non-zero flights.
- A contrast is "a difference" only when the block CI excludes 0 AND the sign-flip p < 0.05. With 4 reads, Holm over the 4 p values.
- Missing data: a query with no depth for any pick has off None and is dropped from that contrast; the count dropped is printed. More than 5 % dropped in any arm -> the read is NOT EVALUABLE (no partial result).

## What it can say (fixed now)
- Mechanism line for SUB3 / SUB4 only, never a benchmark metric (SUB1 stays KILLED as a metric lane).
- Expected sign if "uniform pickers keep more off-plane structure": cube0.5 and dpp2d120a1 off differ from dpp0.5 in the same direction as their flat fail (dpp2d120a1 fails +2.75 pt [+1.08, +4.90] more than dpp0.5 on FRESH-1, gate 2026-11-04). Any other pattern is reported as is, with no second definition of off or dspan.

## Not results
- The hospital/P041 smoke (1 flight, 530 queries, placeholder poses) is a pipeline check only and is never reported as a number.
- No re-binning, no other K, no other arms after the read.

## Kill
- Owner answer B / B-lite (default 2026-12-03 12:00) with no rerun allowed -> T33 is closed as "ready, never run" and SUB1 is fully closed.
