# SUB1 pre-registration: a continuous metric where uniform picking lowers error (v1, 12 Oct 2026)

Written and committed before any of these metrics was computed on any split. Pickers: sd (strongest-K), cube0.5, grid; K = 100,
RaCo, LightGlue, same map, kpbench.bench protocol. Fail = no pose, > 25 cm or > 2 deg (unchanged).

## 1. Candidate metrics (per query; + = higher is better, - = lower is better)
Pose = LM-refined estimate; inliers = AP3P-RANSAC inlier set. Value for a query with no pose in brackets.

| # | name | metric | dir | no pose |
|---|------|--------|-----|---------|
| 1 | crlb | log10 of the camera-centre std in cm at 1 px iid noise: J = 2n x 6 reprojection Jacobian over the inliers at the estimated pose, Sigma = (J^T J)^-1, propagated to the centre c = -R^T t, metric = log10 sqrt(trace Sigma_c) | - | cap = DEV pooled 99th percentile |
| 2 | n_inl | inlier count | + | 0 |
| 3 | inl_ratio | n_inl / n_match | + | 0 |
| 4 | hull | convex-hull area of the inlier (u, v) / (640 x 480) | + | 0 |
| 5 | reproj_med | median inlier reprojection error at the refined pose, px | - | 3 (the RANSAC threshold) |
| 6 | log_t | log10(max(position error cm, 0.1)) | - | 2 (= 100 cm) |
| 7 | log_r | log10(max(rotation error deg, 0.01)) | - | log10(20) |
| 8 | thr_score | mean over tau in {1, 2, 5, 10, 25} cm of 1[t_err <= tau and r_err <= tau / 12.5 deg] | + | 0 |

## 2. Selection rule (DEV, 12 flights, all frames; flat bin reported descriptively only)
- Orient every metric so positive = uniform better. d_q = m(cube0.5) - m(sd), paired per query.
- z = mean(d) / SE, SE from a block bootstrap (20 consecutive frames per block within a flight, 2000 resamples).
- Predictive: AUROC of the oriented metric against the fail flag on the pooled sd + cube0.5 DEV rows. Eligible if AUROC >= 0.75.
  Metrics 6-8 contain the fail definition: they are eligible by construction but must also reach z >= 2 on the queries where both
  pickers succeed, otherwise they are fail % relabelled and dropped.
- Pick the eligible metric with the largest z; it must reach z >= 2.5 (about Bonferroni over 8 at one-sided 0.05). Exactly one
  primary metric goes to HELD_OUT.
- Grid sanity: z(grid - sd) on the chosen metric must be > 0. If it is < 0, cube0.5 is still tested on HELD_OUT but the claim is
  downgraded from "uniformity" to "cube0.5 specifically".

## 3. Confirmation (HELD_OUT, 19 flights, all frames, run once)
One test: paired mean d_q (cube0.5 - sd) on the chosen metric, H1 mean d > 0, one-sided block-bootstrap p, alpha = 0.05.
- Confirmed: p < 0.05 and mean d >= 0.5 x the DEV mean d.
- Not confirmed: anything else. No second metric, no subset re-test; flat-bin d is reported as a number only.

## 4. Kill and boring explanations
- Kill: no DEV metric reaches z >= 2.5, or HELD_OUT z < 1 or the sign reverses -> drop the continuous-metric line; fail % stays
  the only claim.
- Boring A, it is just inlier count: bin queries by n_inl(sd) in bins of 5, recompute z within bins (stratified paired
  difference). If z falls below 1, the metric adds nothing over n_inl; report n_inl itself.
- Boring B, depth spread, not image uniformity: regress d_q on the difference in inlier depth IQR; if the residual mean d loses
  more than 50 % of its size, it is a depth effect. Grid (2D only) must also reproduce >= 50 % of cube0.5's effect on the same
  metric, otherwise the win is 3D spread, not uniformity.

## Implementation
kpbench/metrics.py computes columns 1-8 (raw, uncapped; the no-pose values above are applied at analysis time) behind
`--metrics` in kpbench.evaluate; existing columns are unchanged. Results go to results/sub1/.
