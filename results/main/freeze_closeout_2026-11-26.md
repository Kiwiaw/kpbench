# MAIN freeze close-out, 26 Nov 2026

Gist: dpp0.5 beats cube0.5 on near-planar ("flat") frames only at the same budget K <= 100. At K200/K400, and at equal selection time, it is a tie or worse. FRESH-2 (~2774 flat frames, 12 GPU-h) stays unspent. This note is what option B of board/decision.md closes on.

Claim: **equal K, K <= 100 only.** dpp0.5 stays incumbent, unconfirmed against cube0.5 (p 0.0508).

## Numbers (pt = flat fail % difference; every number copied from the file in the last column)

| contrast | set | K | flat frames | fails | diff pt [95 %] | p | source |
|---|---|---|---|---|---|---|---|
| dpp0.5 - cube0.5 | leaderboard, 31 flights | 100 | 1813 | 192 / 227 | -1.93 | 0.0386 (two-sided; Holm 4 -> 0.077) | results/main/j7_forest_register.md l.11, l.66 |
| dpp0.5 - cube0.5 | FRESH-1, 13 flights | 100 | 2075 | 62 / 95 | -1.59 [-3.46, -0.25] | 0.0508 (sign-flip, 7/4/2 flights) | results/main/dpp_vs_cube_fresh1_k100.md l.7 |
| dpp0.5 - cube0.5 | DEV seed 23 | 200 | 936 | - | +0.53 [-2.25, +2.75] | - | results/main/k_table_2026-11-21.md l.9 |
| dpp0.5 - cube0.5 | DEV seed 23 | 400 | 936 | - | +3.85 [+0.85, +7.40] | - (dpp0.5 worse) | results/main/k_table_2026-11-21.md l.10 |
| dpp0.5 - cube0.5 | FRESH-1 | 400 | 2075 | 64 / 57 | +0.34 [-0.40, +1.00] | - | results/main/k_table_2026-11-21.md l.13 |
| E1: cube0.5 K400 - dpp0.5 K100 (equal selection time) | FRESH-1 | 400 vs 100 | 2075 | 57 / 62 | -0.24 [-1.59, +1.45] | 0.8828 (7/1/5 flights) | results/main/e1_equal_compute.md l.9, l.31 |

Sign: negative = first picker fails less. In the E1 row the first picker is cube0.5, so -0.24 favours cubes (a tie).

## Reading
- K100: two sets agree in sign (-1.93 and -1.59 pt); the out-of-sample one misses alpha 0.05 by 0.0008 (p 0.0508).
- Budget: the gain flips sign between K100 and K200 on DEV, and between K100 and K400 on FRESH-1; DEV K400 is significantly worse for dpp0.5.
- Compute: cube0.5 at K400 costs about the same selection time as dpp0.5 at K100 and fails 57 vs 62 of 2075. E1 kill met -> no "better at equal compute" claim.
- Tags: FRESH-1 rows are post hoc (selection seen) [P]; leaderboard K100 counts audited by two methods (J2b, C40).

## What stays open
- FRESH-2 K100 confirm (registered J7 endpoint, power 69-89 %), only on the owner's (A) in board/decision.md by 2026-12-03 12:00.
- E1 interval has one method so far (SUB2 analyst carry-over).
- Numbers here use committed files only; the uncommitted working-tree edits were audited the same day (results/audit/2026-11-26_worktree.md, PASS 4/4).
