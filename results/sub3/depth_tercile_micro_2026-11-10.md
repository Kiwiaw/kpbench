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
