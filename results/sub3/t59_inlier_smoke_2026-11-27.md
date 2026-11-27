# T59: optional inlier-index field, smoke (SUB3, engineer 2026-11-27, dispatch 13:20) - plumbing only, no value

- Code: kpbench.run --save-inliers (needs --save-picks) -> bench.run_flight(save_inliers=True) adds a 6th field to each query's onplane entry: the PnP inliers as indices into the stored picks (None = no pose); meta has_inliers. Off by default.
- tools/inlier_smoke.py, hospital/P041, dpp0.5 K100 lg, placeholder poses: rows byte-identical with no flag / --save-picks / --save-picks --save-inliers; onplane fields 1-5 and picks identical; 530 of 530 queries with a valid 6th field (sorted unique indices < n_sel, length n_inl; None on the 6 no-pose queries) -> PASS.
- tools/picks_smoke.py --arms dpp0.5,cube0.5,dpp2d120a1 on this branch: 530/530 per arm, pickarms 153/153 flat, dumped pixels = direct picker call 530/530 -> PASS.
- tests/test_pickstats.py: new test_worker_inlier_pick_indices (synthetic 60 points, 10 outliers: 50 inliers mapped to pick indices) ; 9 passed with test_pickstats_file and test_pickarms.
- Exp (c) spread = convex hull of picks[idx] / image area, per the prereg (results/sub3/mediation_prereg_2026-11-26.md). Not computed here.
