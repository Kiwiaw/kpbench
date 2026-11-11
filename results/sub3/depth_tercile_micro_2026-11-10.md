# Depth placement of the picks on MICRO (K 100, 393 frames, cache only, 2026-11-10 13:33)

O4 / R20 (dispatch 2026-11-10 12:30 SUB1): nearest depth tercile = the closest third of the frame's candidates with depth; share = fraction of a picker's picks (with depth) in it. Paired per-frame differences vs sd; sign counts = frames where the picker's value is above / below sd (ties excluded).

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

## J4 depth-quota control + cube0.5r vs cube0.5 (engineer 2026-11-11, tools/depth_quota_micro.py, K 100, 393 MICRO frames, cache only)

dq3 = equal quota per depth bin (3 equal-count bins of the frame's candidates with depth), strongest score first inside a bin, no voxels. Paired per-frame differences vs cube0.5; a / b = frames above / below cube0.5 (ties dropped), p = two-sided sign test. dq3 returned K distinct picks on 393 / 393 frames.

| picker | near-tercile share | diff vs cube0.5 (a / b, p) | pick-depth std (m) | diff vs cube0.5 (a / b, p) | on-plane share | diff vs cube0.5 (a / b, p) | 0.5 m cubes occupied | diff vs cube0.5 (a / b, p) | pick Jaccard vs cube0.5 |
|---|---|---|---|---|---|---|---|---|---|
| sd | 0.479 | +0.3083 (320 / 65, p 1.4e-41) | 2.407 | -0.4130 (80 / 313, p 1.2e-33) | 0.903 | +0.0995 (229 / 58, p 3.4e-25) | 35.3 | -37.6590 (0 / 393, p 9.9e-119) | 0.302 |
| cube0.5 | 0.170 | - | 2.820 | - | 0.803 | - | 72.9 | - | 1.000 |
| cube0.5r | 0.162 | -0.0085 (96 / 196, p 4.9e-09) | 2.814 | -0.0057 (129 / 264, p 8.7e-12) | 0.800 | -0.0032 (64 / 125, p 1.1e-05) | 72.9 | +0.0000 (0 / 0, p 1) | 0.218 |
| dq3 | 0.340 | +0.1696 (360 / 29, p 8.6e-74) | 2.505 | -0.3152 (123 / 270, p 9.2e-14) | 0.916 | +0.1123 (266 / 24, p 8.4e-53) | 37.1 | -35.8422 (0 / 393, p 9.9e-119) | 0.291 |

- cube0.5r vs cube0.5 (SUB1 row): near-tercile share 0.162 vs 0.170, diff -0.0085, frames 96 above / 196 below, sign p 4.9e-09; vs sd -0.3168. Verdict: within 0.05 of cube0.5 -> the depth spread comes from the cells, not from the score order inside a cell; the random-within-cell arm is predicted to tie.
- dq3 vs cube0.5 (SUB3 control): near-tercile share 0.340 vs 0.170, diff +0.1696, frames 360 above / 29 below, sign p 8.6e-74. Kill (within 0.05 AND no paired sign difference): NOT met.
- Caveat: the dq3 share is 0.333 by construction (K / 3 per bin), so the share difference says only that cubes spend LESS than a third of the budget near, not more evenly in depth; the columns that are not fixed by construction are depth std, on-plane share, cubes occupied and the pick overlap. No fail outcomes on the laptop: descriptive, [T].
