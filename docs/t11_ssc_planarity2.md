# MAIN 18 Oct 2026: ssc picker (spread-class baseline) and a second planarity definition

Dispatch 2026-10-18 13:15 (MAIN engineer step; scout edition 5: SSC/ANMS and 2D DPP are the spread-class baselines).

## ssc[<o>][p<M>]
Suppression via square covering (Bailo et al. 2018): among the strongest M (default 2000) candidates with depth, binary
search on the square side w so that greedy covering in score order keeps ~o*K points; the K strongest of those, fill by rank.
Tune family `ssc` (o 1-2, M 200-2000 log; seeds ssc1p2000, ssc1p400, ssc1.5p2000).

Cache-only MICRO check (all 393 MICRO flat queries, RaCo candidates, K 100; cov = share of 16 x 12 cells of 40 px hit,
jac = overlap with dpp0.5 picks; picker seconds for all 393 frames). No fail % on the laptop (no flight poses).

| picker | cov | jac vs dpp0.5 | s |
|---|---|---|---|
| sd | 0.256 | 0.403 | 0.0 |
| dpp0.5 | 0.324 | 1.000 | 9.6 |
| dpp2d74 | 0.393 | 0.509 | 8.9 |
| ssc1 | 0.513 | 0.244 | 15.4 |
| ssc1p400 | 0.416 | 0.346 | 2.8 |
| ssc1.5 | 0.474 | 0.312 | 15.4 |
| ssc2 | 0.434 | 0.342 | 15.3 |

Reading: ssc is the most image-spread arm (0.513 vs dpp0.5 0.324) and shares only a quarter of dpp0.5's picks, so it is a
real contrast; the dispatch kill (ssc cov <= 0.328) is not met. Fail % and rep5 need the Euler smoke (agent Qt-ssc_smoke1).

## plane_resid (kpbench.planarity2)
RMS orthogonal distance of the strongest 300 sp candidates with depth to their total-least-squares plane / median depth.
flat2 = plane_resid <= 0.14074, fixed once so that the flat2 count equals the plane_frac >= 0.85 count over the 6 flights
whose sp caches are on the laptop (MICRO + contaminated DEV: oldtown P000 P001 P002 P006 P007, hospital P041).

| queries | flat (plane_frac) | flat2 | both | agree | Jaccard | Spearman(plane_frac, resid) |
|---|---|---|---|---|---|---|
| 3980 | 814 | 814 | 617 | 0.901 | 0.610 | -0.554 |

Reading: the two definitions share 617 of 814 flat queries (76 %); a claim on the flat bin should be reported under both.
Next: run `python -m kpbench.planarity2` on Euler (all 31 flights' sp caches) and add a flat2 column to the gate summary.
Labels in results/labels are not touched; gates keep plane_frac.
