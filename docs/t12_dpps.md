# T12: depth-sliced DPP `dpps<n>[q|l|e]` (pre-registration, 21 Oct 2026, written before any result)

Idea: slice the camera frustum by depth and give each depth slice a fair share of the K picks, so the picks do not pile up
on the nearest surface.

## Definition (fixed, no tuning)
- Identical to `dpp0.5` (sigma 0.5 m, alpha 1): same candidates (all with depth, z < DEPTH_MAX, score order), same Gaussian
  kernel on camera-frame metres, same quality q_i = exp(-rank_i / 100), same greedy log-det MAP (`dpp_greedy`),
  plus a per-slice cap in the greedy loop.
- Slice edges from the depths of the strongest 400 candidates with depth (all of them if fewer):
  - `q` equal-count: interior edges = depth quantiles k/n, k = 1..n-1;
  - `l` log-spaced: n slices of equal log width between the 5th and 95th depth percentile, outer slices open-ended
    (interior edges = geomspace(p5, p95, n+1)[1:-1]);
  - `e` equal-width: same as `l` with linear spacing (linspace(p5, p95, n+1)[1:-1]).
  Every candidate with depth gets slice id = number of interior edges <= its depth.
- Cap per slice = ceil(K / n). Phase 1: each greedy step takes the argmax gain among the candidates (gain > 1e-12) whose
  slice is not full. Once no non-full slice has a candidate left, phase 2 continues uncapped with the same objective
  (left-over picks redistributed). Candidates without depth ignored as in dpp; fill by rank if < K picks.
- Unit check (tests/test_core.py): `dpps1` reproduces `dpp0.5` picks exactly.

## Variants
Primary `dpps4q`; also `dpps4l` (literal frustum slicing) and `dpps8q`. The gate counts only `dpps4q`.

## Steps and rules
- Step 0 (laptop, cached frames, ~5 min): Jaccard of `dpps4q` vs `dpp0.5` picks on 80 random MICRO flat queries
  (oldtown P001 + hospital P041, seed 0), as in docs/t11_dpp2d.md. Mean Jaccard > 0.9 -> the cap never binds:
  STOP, report "mechanism absent".
- Step 1 (Euler, one node, ~1 h): `kpbench.evaluate --split dev_flat` (936 DEV flat queries, LightGlue, RaCo, K 100;
  fail = no pose, > 25 cm or > 2 deg) for `dpps4q`, `dpps4l`, `dpps8q` and `dpp0.5` rerun in the same job (device rule).
  KILL if the paired difference `dpps4q - dpp0.5` on flat fail > -0.5 pt.
- Step 2 (only if step 1 passes): DEV 12 flights full, one node per flight array (`euler/gate.sh dpps4q dpp0.5 dev`).
  CONTINUE if flat diff <= -1.44 pt with 95 % paired block-bootstrap CI upper < 0, flights better >= worse, all-frames
  diff <= +0.1 pt, rep5 drop <= 0.03. Near miss (-1.0 to -1.44, or CI spans 0) -> note "FRESH-1 retest per 19 Oct rule"
  (not run now). Else KILL. HELD_OUT only after CONTINUE (not run now).

## Known risk
An earlier log-depth-band round-robin picker (`dband`) lost +1.3 pt; 87 % of DEV failures are near frames whose candidates
span < 2 m, so slicing may change little.

## Results
(appended below after each step)

### Step 0 (laptop, 21 Oct): mechanism present -> go to step 1
80 MICRO flat queries (seed 0), RaCo K 100, Jaccard vs dpp0.5 picks: dpps4q mean 0.634 (median 0.646, min 0.325,
0/80 identical); dpps4l 0.686; dpps8q 0.623. Median p5-p95 depth span of the top-400 candidates 1.44 m.
Script: scratchpad t12_step0.py (not in repo).
