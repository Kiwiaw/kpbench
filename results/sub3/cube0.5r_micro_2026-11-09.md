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
