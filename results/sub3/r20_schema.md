# R20 schema: on-plane share of the picks on MICRO (2026-11-02, 393 frames, 393 flat, K 100)

tools/r20_onplane_micro.py on the cached candidates (no poses, no fail %). on-plane share = picks with depth within 30 cm of the frame's dominant plane (RANSAC, strongest 300 with depth); the same number is now stored per query by kpbench.run / kpbench.evaluate (bench.onplane). Plane share of the fit set: mean 0.924 all, 0.924 flat.

| picker | on-plane share all [95 %] | flat [95 %] | flat diff vs sd [95 %] | frames with lower share than sd (flat) |
|---|---|---|---|---|
| sd | 0.903 [0.888, 0.916] | 0.903 [0.888, 0.916] | +0.000 [+0.000, +0.000] | 0 of 393 |
| grid | 0.915 [0.905, 0.925] | 0.915 [0.905, 0.925] | +0.012 [+0.006, +0.020] | 144 of 393 |
| cube0.5 | 0.803 [0.785, 0.821] | 0.803 [0.785, 0.821] | -0.100 [-0.113, -0.087] | 229 of 393 |
| dpp0.5 | 0.840 [0.822, 0.856] | 0.840 [0.822, 0.856] | -0.063 [-0.073, -0.054] | 233 of 393 |

Read: a spread picker with a lower on-plane share than sd on flat frames keeps more off-plane points; whether that lowers the fail rate is the R20 question for the DEV/FRESH rows (needs a rerun with the new field, no rows carry it yet).
