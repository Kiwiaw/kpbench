# SUB1 metrics on dev (8273 queries, 937 flat; rule: docs/sub1_metrics.md)

fail %: sd 3.58, cube0.5 2.42, grid 2.85; crlb no-pose cap 0.925

| metric | mean d (cube-sd, oriented) | z | AUROC | z both ok | eligible | z grid-sd | grid share | mean d flat | z after n_inl | z after depth IQR | share kept after depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crlb | -0.0312 | -7.27 | 0.671 | None | False | 16.13 | -1.93 | -0.0192 | -1.76 | -8.29 | 1.14 |
| n_inl | -4.0856 | -14.04 | 0.922 | None | True | -29.58 | 1.59 | -6.7492 | None | -13.5 | 0.95 |
| inl_ratio | -0.0397 | -15.96 | 0.926 | None | True | -22.68 | 1.07 | -0.0693 | -4.53 | -15.54 | 0.97 |
| hull | 0.0178 | 8.6 | 0.91 | None | True | 33.5 | 5.24 | 0.0256 | 10.83 | 7.91 | 0.92 |
| reproj_med | 0.0252 | 1.27 | 0.728 | None | False | 1.11 | 0.89 | 0.2885 | 2.43 | 1.57 | 1.24 |
| log_t | -0.0238 | -2.79 | 0.999 | -8.3 | False | 12.29 | -3.52 | 0.1399 | -0.1 | -4.23 | 1.47 |
| log_r | -0.0004 | -0.07 | 0.883 | -2.41 | False | 9.12 | -132.41 | 0.0752 | 4.3 | -1.21 | 17.69 |
| thr_score | -0.0039 | -1.17 | 1.0 | -4.21 | False | 10.62 | -8.44 | 0.0544 | 1.81 | -2.84 | 2.34 |

**chosen: hull**
