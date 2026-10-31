# R19 planar-aware solver option (2026-10-31, laptop)

Option exists: yes, added today. Before, bench.pnp had AP3P-RANSAC + LM only.

- Flag: `python -m kpbench.run --solver ippe`, `python -m kpbench.evaluate --solver ippe`, `bash euler/gate.sh <cand> <inc> <split> <K> ippe`. Default ap3p: that code path is unchanged.
- Method: AP3P-RANSAC as always, plus one planar hypothesis (SVD plane of the map points, homography RANSAC 3 px, IPPE, two poses). The pose with more inliers on the true 3D points wins, ties go to AP3P, then the same LM. Uses matches only.
- Plain IPPE alone was not used: it needs coplanar input, and real flat frames still hold off-plane points.
- Run id gets `-ippe`; meta and gate json carry `solver` and `solver_inc`. cand = inc compares the two solvers on one picker (candidate ippe, incumbent ap3p).

## Smoke

MICRO smoke NOT run: the laptop has no MICRO flight data (`flight oldtown/P001 not found`). No flight, frame or fail count exists for this option yet.

Synthetic scenes only, 300 repeats per row, 1 px noise (results/main/r19_solver_smoke.md, tools/solver_smoke.py). Fail % = no pose, > 25 cm or > 2 deg:

| scene | matches | outliers | ap3p fail % | ippe fail % | identical result |
|---|---|---|---|---|---|
| planar, tilt 30 | 12 | 30 % | 23.3 | 22.3 | 93 % |
| planar, tilt 30 | 25 | 30 % | 12.0 | 11.0 | 86 % |
| planar, tilt 30 | 50 | 50 % | 4.7 | 4.0 | 87 % |
| planar, tilt 60 | 25 | 30 % | 56.3 | 56.3 | 99 % |
| near-planar, 3 cm | 25 | 30 % | 0.7 | 0.3 | 92 % |
| near-planar, 3 cm | 50 | 50 % | 0.0 | 0.0 | 88 % |
| 3D, 1 m | 25 | 30 % | 0.0 | 0.0 | 100 % |
| 3D, 1 m | 50 | 50 % | 0.0 | 0.0 | 100 % |

Read: on exact-plane scenes the gain is at most 1.0 pt in this toy, and 3D scenes are untouched. That is an upper-side hint only, since synthetic errors are not TartanAir errors. The tilt-60 row fails equally for both, so it says nothing about the solver.

Dispatch kills: "> 1 h rewrite" not met. "MICRO counts identical" not testable on the laptop.

## Next, needs VPN

1. `python -m kpbench.evaluate --picker cube0.5 --split micro` with and without `--solver ippe` on one GPU node (about 2 x 1 min per picker; sd the same). Flight runtime was not measured.
2. Only if MICRO counts differ: R19 gate on FRESH-1, `bash euler/gate.sh cube0.5 cube0.5 fresh 100 ippe` (4 GPU-h in the dispatch plan).

Restore check: tools/dev_frame_check.sh [dev|held_out|fresh]. Self-test flags 384 vs 577 and 401 vs 533; missing images only warn, because cached runs never read them. Laptop run: 12/12 DEV flights MISSING (expected, no flight data), exit 1. Not yet run on Euler.
