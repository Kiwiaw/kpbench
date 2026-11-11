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
