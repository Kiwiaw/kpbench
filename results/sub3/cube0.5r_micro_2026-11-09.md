# cube0.5r vs cube0.5 on MICRO (K 100, 393 frames, cache only, 2026-11-09 13:24)

Within-cell order control (R27, STDLoc objection): same cubes in the same order, random member per cube instead of the strongest.

- occupied cells identical on 393 / 393 frames (kill "cells differ": not met)
- pick Jaccard cube0.5r vs cube0.5: mean 0.218, median 0.156, min 0.031, frames with Jaccard 1.0: 0 / 393 (kill "all 1.0": not met)
- on-plane share of the picks (bench.onplane, frames with a plane fit 393): cube0.5 0.803, cube0.5r 0.800, paired diff -0.0032
- cells occupied per frame: mean 72.9 of K 100 (cells with > 1 candidate are where the order matters)
- picker time: cube0.5 0.74 s, cube0.5r 1.14 s over 393 frames
- regression: cube0.5 picks identical to the previous pickers.py on 393 / 393 frames

## First 5 frames

| flight | frame | flat | cells | same cells | picks Jaccard | on-plane cube0.5 | on-plane cube0.5r |
|---|---|---|---|---|---|---|---|
| oldtown/P001 | 21 | True | 100 | True | 0.117 | 1.0 | 1.0 |
| oldtown/P001 | 22 | True | 100 | True | 0.093 | 1.0 | 1.0 |
| oldtown/P001 | 23 | True | 100 | True | 0.087 | 1.0 | 1.0 |
| oldtown/P001 | 24 | True | 100 | True | 0.081 | 1.0 | 1.0 |
| oldtown/P001 | 25 | True | 100 | True | 0.105 | 0.98 | 0.98 |

Flat frames (plane_frac >= 0.85): 393 of 393. Per-pick on-plane labels come from the frame's dominant plane (RANSAC, 30 cm), not from poses.

## Laptop limits (engineer 2026-11-09)
- `python -m kpbench.evaluate --picker cube0.5r` (MICRO, LightGlue) cannot run here: the laptop holds the candidate cache only, no flight images (`flight oldtown/P001 not found under local_data`), so cube0.5r has no fail % until a gate runs on Euler (held item queue/_held/Qg-cube0.5r-vs-cube0.5-fresh2.json in the agent repo, save_picks on). Stored MICRO cube0.5 (results/eval/micro, lg): fail 22.65 %, rep5 0.305 on 393 frames.
- Picker cost: the shuffle adds ~1 ms per frame (0.74 -> 1.14 s over 393 frames).

## O4 depth placement (engineer 2026-11-10, tools/depth_tercile_micro.py)

| picker | near-tercile share | diff vs sd | frames above / below | pick-depth std (m) | diff vs sd | frames above / below | on-plane share | diff vs sd |
|---|---|---|---|---|---|---|---|---|
| sd | 0.479 | +0.0000 | 0 / 0 | 2.407 | +0.0000 | 0 / 0 | 0.903 | +0.0000 |
| cube0.5 | 0.170 | -0.3083 | 65 / 320 | 2.820 | +0.4130 | 313 / 80 | 0.803 | -0.0995 |
| cube0.5r | 0.162 | -0.3168 | 65 / 317 | 2.814 | +0.4073 | 304 / 89 | 0.800 | -0.1027 |
| dpp0.5 | 0.231 | -0.2479 | 60 / 329 | 2.790 | +0.3826 | 325 / 68 | 0.840 | -0.0633 |
| grid | 0.433 | -0.0453 | 113 / 271 | 2.495 | +0.0883 | 290 / 103 | 0.915 | +0.0122 |
| dpp2d120a1 | 0.435 | -0.0432 | 136 / 253 | 2.453 | +0.0460 | 298 / 95 | 0.898 | -0.0051 |

Frames with a defined value: near share and depth std need >= 3 candidates with depth and >= 1 pick with depth; on-plane needs a plane fit (bench.onplane).

## Pre-registered reading

- sd nearest-tercile share 0.479 vs cube0.5 0.170: gap +0.308 (rule >= 0.05: met); frames with cube0.5 below sd 320 / 393 (rule >= 300 / 393: met)
- verdict: score ranking concentrates picks near-field on MICRO, voxel caps spread depth (descriptive, [T], 393 frames, no fail outcomes)

## O3 gated-cube flag check (engineer 2026-11-10, tools/plane_flag_micro.py)

Flag = RANSAC dominant-plane inlier share over the strongest 300 candidates with depth >= 0.85; label = depth-map plane_frac >= 0.85 (results/labels). Flights: hospital_P041, oldtown_P000, oldtown_P001, oldtown_P002, oldtown_P006, oldtown_P007.

| set | n | label flat | flag flat | both (TP) | label only (FN) | flag only (FP) | neither (TN) | agreement | flat recall | flat precision |
|---|---|---|---|---|---|---|---|---|---|---|
| all labelled queries | 3980 | 814 | 616 | 550 | 264 | 66 | 3100 | 0.917 | 0.676 | 0.893 |
| MICRO 393 (label-flat by construction) | 393 | 393 | 318 | 318 | 75 | 0 | 0 | 0.809 | 0.809 | 1.000 |

Agreement on all labelled queries by flag threshold (label fixed at 0.85): 0.60 -> 0.864, 0.70 -> 0.924, 0.75 -> 0.939, 0.80 -> 0.936, 0.85 -> 0.917, 0.90 -> 0.890, 0.95 -> 0.871
Spearman rank correlation of plane_frac_hat with the label plane_frac: 0.846; median |hat - label| 0.083.

## Pre-registered reading

- flag agreement on all labelled queries 0.917 (condition >= 0.9: met); on MICRO the flag recalls 0.809 of the 393 label-flat frames
- the detection-based flag is usable for a gated cube (cube0.5 when flagged, dpp0.5 otherwise); still a spec: the oracle composite guards (SUB2 r24_strata_fresh1.md) must clear first, then a gate on a gate-capable set
- caveat (engineer): agreement 0.917 is carried by the 3166 non-flat queries; the flag recalls only 0.676 of the 814 label-flat queries (precision 0.893), so a gated cube would run dpp0.5 on about a third of the truly flat frames; threshold 0.75 gives the best agreement 0.939 (descriptive, not pre-registered).
