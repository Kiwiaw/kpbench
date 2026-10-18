# SUB4 objection (f): abstention and within-query test (kpbench.sub4_abstain)

DEV flat queries, stored K = 100 LightGlue rows of dpp0.5 (936 queries, 135 fails). rep5 uses true pose + depth (measurement, upper reference); n_inl / n_match are test-time.

## (i) Abstention: flat fail % of the kept queries after rejecting the lowest x %

| score | AUROC | reject 0 % | reject 5 % | reject 10 % | reject 20 % | reject 30 % | rel. drop at 10 % |
|---|---|---|---|---|---|---|---|
| rep5 | 0.864 | 14.42 | 12.49 | 10.93 | 7.34 | 3.66 | 24.2 % |
| n_inl | 0.888 | 14.42 | 10.80 | 8.79 | 6.14 | 4.12 | 39.1 % |
| n_match | 0.836 | 14.42 | 11.02 | 9.14 | 7.34 | 5.95 | 36.6 % |

## (ii) Within query: picker-dependent flat frames (exactly one picker fails)

| pair | frames | score | posed side higher | ties | sign-test p |
|---|---|---|---|---|---|
| dpp0.5 vs cube0.5 | 55 | rep5 | 28 (0.55) | 4 | 0.58 |
| dpp0.5 vs cube0.5 | 55 | n_inl | 37 (0.69) | 1 | 0.0091 |
| dpp0.5 vs cube0.5 | 55 | n_match | 32 (0.58) | 0 | 0.28 |
| dpp0.5 vs sd | 95 | rep5 | 63 (0.70) | 5 | 0.00019 |
| dpp0.5 vs sd | 95 | n_inl | 59 (0.63) | 2 | 0.012 |
| dpp0.5 vs sd | 95 | n_match | 49 (0.52) | 1 | 0.76 |

Check: 936 queries, 135 fails, rep5 AUROC 0.864 = results/sub3/rep5_auroc.md (DEV flat dpp0.5).

Verdict (18 Oct, engineer, dispatch 13:15 SUB4 step): rep5 is actionable at query level: rejecting the lowest-rep5 10 % cuts
flat fail 14.42 -> 10.93 % (-24 % rel; kill was < 10 % rel, not met). But the test-time inlier count does better (-39 % rel,
AUROC 0.888), so as an abstention signal rep5 is beaten by a Ferranti-family baseline; its value is diagnostic (it needs pose).
Within query, rep5 separates dpp0.5 from sd on the frames where exactly one fails (posed side higher 0.70, p 0.0002) but not
dpp0.5 from cube0.5 (0.55, p 0.58, 55 frames): between two spread pickers, the frame-level failure is not a repeatability story.
Caveats: rows from the 936-frame laptop labels (PROVISIONAL-936 for any Euler comparison); reproj_med needs metric rows.
