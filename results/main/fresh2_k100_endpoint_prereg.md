# Pre-registration: FRESH-2 single confirmatory endpoint, dpp0.5 vs cube0.5 K100 flat (written 2026-11-17, engineer, J7 part 1b; before any FRESH-2 row exists)

- Why: vs cubes missed 0.05 twice (31 flights p2 0.0386 -> Holm 0.077; FRESH-1 p2 0.0508), FRESH-1 read >= 10x; one fixed endpoint ends the forking paths (results/main/j7_forest_register.md).
- Frames: FRESH-2 only = same-env trajectories never read by any gate (pool census results/sub4/fresh2_pool_census.md in the agent repo: ~2774 flat est.); not FRESH-1, not HELD_OUT, not DEV. Flight list frozen in a splits_fresh2.json before any row is run.
- Arms: raco_dpp0.5_K100_same_lg vs raco_cube0.5_K100_same_lg on FRESH-2, matcher lg, solver ap3p, same commit for both arms.
- Unit = flight. Statistic: per flight d_f = dpp0.5 flat fails - cube0.5 flat fails (flat = plane_frac >= 0.85; fail = err None or > 25 cm or > 2 deg).
- Test: exact ONE-sided sign-flip on d_f (zeros dropped), H1 = dpp0.5 fails less (direction fixed by the two earlier reads); p = share of the 2^m sign patterns with sum <= observed sum. alpha 0.05. One look, no interim read, no added flights after the first read.
- Reported with it (descriptive, not tested): pooled diff in pt, flight-cluster bootstrap 95 % CI (10000, seed 0), flights better/tie/worse.
- Decision: p < 0.05 -> "dpp0.5 fails less than cube0.5 at K100" may be stated as flight-significant (FRESH-2). p >= 0.05 -> the cubes claim reduces to the K interaction (C43: -40 of 2075, p 0.0156) and "fewer fails, not flight-significant" stays final; no further test of this pair.
- Secondary (labelled secondary, no claim): K400 route B (results/main/k400_routeB_prereg.md), vs grid, vs sd, rep5, very-flat bin, SUB3 non-inferiority (agent repo results/sub3/noninf_margin_prereg.md is its own pre-registered test).
- Freeze holds (decision.md default since 13 Nov 12:00) and FRESH-2 needs Kinga's go; this file only fixes the test.
