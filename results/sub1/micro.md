# SUB1 metrics on micro (393 queries, 393 flat; rule: docs/sub1_metrics.md)

fail %: sd 34.10, cube0.5 22.65, grid 29.26; crlb no-pose cap 1.206

| metric | mean d (cube-sd, oriented) | z | AUROC | z both ok | eligible | z grid-sd | grid share | mean d flat | z after n_inl | z after depth IQR | share kept after depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crlb | -0.0524 | -1.26 | 0.69 | None | False | 4.41 | -2.36 | -0.0524 | 0.59 | -1.09 | 0.83 |
| n_inl | -5.4885 | -3.02 | 0.793 | None | True | -4.11 | 0.74 | -5.4885 | None | -3.0 | 0.99 |
| inl_ratio | -0.06 | -3.1 | 0.799 | None | True | -3.41 | 0.56 | -0.06 | -0.07 | -3.15 | 1.01 |
| hull | 0.0265 | 1.54 | 0.808 | None | True | 5.61 | 3.54 | 0.0265 | 2.16 | 1.77 | 1.11 |
| reproj_med | 1.1234 | 2.56 | 0.765 | None | True | -0.76 | -0.38 | 1.1234 | 2.38 | 2.11 | 0.78 |
| log_t | 0.2265 | 3.43 | 0.999 | -1.2 | False | 3.69 | 0.79 | 0.2265 | 4.06 | 3.14 | 0.85 |
| log_r | 0.1476 | 2.27 | 0.881 | -1.24 | False | 2.6 | 0.83 | 0.1476 | 2.78 | 2.03 | 0.9 |
| thr_score | 0.0748 | 3.55 | 1.0 | -0.72 | False | 3.41 | 0.8 | 0.0748 | 4.17 | 3.27 | 0.81 |

**chosen: reproj_med**
