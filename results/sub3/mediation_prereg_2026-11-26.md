# Scout exp (c) pre-registration: does flat failure track inlier spread at fixed inlier count? (26 Nov 2026)

Written before any value is read. Status at the end of this file: **NOT EVALUABLE** on the rows that exist.

Gist: uniform pickers might help on flat frames simply because they keep more PnP inliers, or because the inliers they keep are spread wider. This test holds the inlier count fixed and asks whether spread still predicts failure.

## Test (fixed now, not changed after reading)
- Frames: the registered flat label (bench flat flag), K100, lg matcher, ap3p solver.
- Fail: rotation error > 2 deg or translation error > 25 cm (the registered rule); a query with no pose counts as a fail.
- Arms: dpp0.5, cube0.5, grid, sd (leaderboard pickers). One observation = one (frame, arm).
- Flights: the 31 leaderboard flights. These rows were already seen, so the result is tagged post hoc [P]. FRESH-2 is not touched.
- Inlier count bins (n_inl): [6, 15), [15, 30), [30, 60), [60, inf). Queries with fewer than 6 inliers have no pose: they are dropped from this test (spread undefined) and their count is reported.
- Spread s: area of the convex hull of the PnP inlier pixels divided by the image area (0..1). Toy example: 40 inliers filling the left half of a 640x480 image -> s about 0.5.
- Inside each bin: split observations at the bin median of s into low and high spread.
- Statistic: d = weighted mean over bins of [fail % (low s) - fail % (high s)], weight = observations in the bin. d > 0 means wider spread fails less.
- Interval: flight-cluster bootstrap (resample flights), 10000 draws, seed 0, 95 % percentile CI.
- Unit of independence: flight (31), not frame or pair.
- Pass: CI lower bound > 0 AND d > 0 in at least 3 of 4 bins.
- Kill: CI includes 0, or d <= 0 -> "spread does not explain flat failure once inlier count is fixed"; ed. 35 keeps spread as a hypothesis only.
- One read, no second spread definition, no re-binning after the read.

## Schema check (field names only, no values read; kpbench HEAD ecef74d1)
- Rows (bench.COLS): flight, q, err_cm, rot_deg, n_sel, n_match, n_inl, rep1, rep5, rep10. Inlier COUNT yes, inlier COORDINATES no.
- --save-picks dump (bench.run_flight): per-pick u, v, on-plane label, depth. These are picks, not inliers: which picks survived matching and RANSAC is not stored.
- So s cannot be computed from any existing rows or from the T33 dump as built.

## Verdict
**(c) NOT EVALUABLE.** Needs one new optional dump field (inlier indices per query, from pose(..., full=True)) and a rerun with it; both are post-freeze work. Pick spread from the T33 dump is a different quantity and is NOT a substitute for this test.
