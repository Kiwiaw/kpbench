# SUB1 metrics on dev (8273 queries, 936 flat; rule: docs/sub1_metrics.md)

fail %: sd 3.58, dpp0.5 2.18; crlb no-pose cap 0.893

| metric | mean d (dpp0.5-sd, oriented) | z | AUROC | z both ok | eligible | z grid-sd | grid share | mean d flat | z after n_inl | z after depth IQR | share kept after depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crlb | 0.0018 | 0.43 | 0.608 | None | False | None | None | 0.0958 | 1.04 | -0.71 | -1.66 |
| n_inl | -0.4516 | -1.9 | 0.915 | None | True | None | None | 0.5823 | None | -1.47 | 0.77 |
| inl_ratio | -0.0218 | -10.9 | 0.916 | None | True | None | None | -0.0098 | -13.51 | -10.64 | 0.98 |
| hull | 0.0372 | 16.4 | 0.877 | None | True | None | None | 0.0746 | 16.12 | 15.59 | 0.95 |
| reproj_med | 0.0478 | 2.31 | 0.667 | None | False | None | None | 0.4096 | 2.42 | 2.67 | 1.15 |
| log_t | 0.0125 | 1.51 | 0.999 | -2.73 | False | None | None | 0.1942 | 1.97 | -0.01 | -0.01 |
| log_r | 0.0451 | 6.35 | 0.852 | 4.56 | True | None | None | 0.2186 | 6.79 | 5.42 | 0.84 |
| thr_score | 0.012 | 3.55 | 1.0 | 1.13 | False | None | None | 0.0814 | 4.09 | 2.01 | 0.55 |

**chosen: hull**

## Hull given inlier count, dpp0.5 (engineer 2026-10-19, dispatch 13:10 SUB1)

Rows: Euler results/eval/dev, sd / cube0.5 / dpp0.5 written 18 Oct (post label sync, 936 flat). Same pre-registered rule as dev.md, d = dpp0.5 - sd:
hull d +0.0372, z 16.4, AUROC 0.877 (cube0.5 reference: +0.0178, z 8.6, AUROC 0.91); hull is again the chosen metric.

Logistic fail ~ log(1+n_inl) + hull (standardised, 300 block-bootstrap resamples over 20-frame blocks), coefficient of hull:

| picker | queries | fails | count coef [95 %] | hull coef [95 %] |
|---|---|---|---|---|
| dpp0.5 | 8273 | 180 | -1.33 [-1.61, -1.03] | -0.18 [-1.00, +0.32] |
| cube0.5 | 8273 | 200 | -1.54 [-1.96, -1.10] | -0.61 [-1.71, -0.05] |
| sd | 8273 | 296 | -0.66 [-0.88, -0.50] | -1.26 [-1.62, -1.01] |

Ferranti leave-one-flight-out (dev_ferranti.md): dpp0.5 count -> count+hull dAUROC -0.035, dPR-AUC -0.0003; pooled over 10 pickers -0.018 / +0.022.

Reading: hull separates pickers (z 16.4) and predicts failure on its own (0.877), but for the incumbent it adds nothing once the inlier count is known (CI covers 0, LOFO delta <= 0). It carries information only for sd, the picker with the least spread. Dispatch 13:10 kill met (hull CI covers 0 given inlier count): hull is an inlier-count proxy for dpp0.5; held-out hull run Q188 not submitted.
