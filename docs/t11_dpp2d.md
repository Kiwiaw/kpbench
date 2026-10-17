# T11 step 1: K knob, dpp2d, oracle floor (17 Oct 2026)

Reviewer objections (a) "the gain is K-specific" and (b) "the gain comes from depth at query time" need DEV-only runs.

## K knob
Already there: `kpbench.run --K`, `kpbench.evaluate --K`, `euler/run.sbatch` RUNS entries `det:picker:K:map[:tag]`
(run ids `..._K<K>_...`). No code change. tune.py keeps K = 100 (the leaderboard protocol).

## dpp2d<sigma>[a<alpha>]
Same greedy DPP as dpp<sigma>, kernel on pixel coordinates (sigma in px) instead of camera-frame metres.
Depth still decides which candidates are eligible (z < 1000, as for every picker except score), not how far apart they are.
Matched sigma: median depth of dpp0.5 picks on MICRO = 2.15 m, so 0.5 m ~ 0.5 * 320 / 2.15 = 74.5 px -> **dpp2d74**.

## Cache-only MICRO check (laptop has the candidate caches, not the flight images/poses, so no fail %)
80 random MICRO flat queries (of 393), RaCo candidates. cov = share of 16 x 12 cells (40 px) hit by the K picks;
jac = |picks ∩ dpp0.5 picks| / |picks ∪ dpp0.5 picks|. All pickers pass kpbench.evaluate.checked_pick (K unique valid indices).

| K | sd cov | cube0.5 cov | dpp0.5 cov | dpp2d74 cov | dpp2d74 jac | dpp2d40 cov | sd jac |
|---|---|---|---|---|---|---|---|
| 50 | 0.161 | 0.191 | 0.199 | 0.232 | 0.511 | 0.225 | 0.418 |
| 100 | 0.258 | 0.305 | 0.328 | 0.391 | 0.531 | 0.373 | 0.403 |
| 200 | 0.389 | 0.446 | 0.472 | 0.568 | 0.560 | 0.545 | 0.504 |
| 400 | 0.544 | 0.599 | 0.610 | 0.641 | 0.748 | 0.705 | 0.715 |

Reading: dpp2d74 spreads more in the image than dpp0.5 (0.391 vs 0.328 cells at K 100) and shares only ~half its picks,
so (b) is a real contrast. At K 400 sd already shares 0.715 of dpp0.5's picks: few candidates with depth are left to choose
from, so a shrinking gain at K 400 is expected from the pool size alone; report it with that caveat.

## DEV runs (held for hand submit, VPN down 17 Oct; agent queue/_held Q191, Q192)
- Q191: `MATCHER=lg RUNS="raco:sd:50:same raco:cube0.5:50:same raco:dpp0.5:50:same raco:sd:200:same raco:cube0.5:200:same raco:dpp0.5:200:same raco:sd:400:same raco:cube0.5:400:same raco:dpp0.5:400:same" sbatch --array=0,1,2,6,7,13,14,16,19,22,29,30 euler/run.sbatch` (the 12 DEV rows of euler/flights.txt; no held-out flight).
- Q192: `MATCHER=lg FORCE=1 RUNS="raco:dpp0.5:100:same:t11 raco:dpp2d74:100:same:t11" sbatch --array=<same> euler/run.sbatch` (tag t11 = fresh same-node pair; leaderboard runs untouched).
- Kill (from the ticket): (a) dpp0.5 not best flat fail at >= 2 of 3 K; (b) dpp2d74 within 0.64 pt of dpp0.5 on DEV flat -> depth is not the mechanism.

## Left (T11 step 2)
Oracle floor gt_match_pnp: replace LightGlue matches by ground-truth correspondences (project picks into the keyframes with
true pose + depth, 3 px), then the same PnP. Needs a bench.py match mode and flight data (Euler), not a picker.
