# SUB1: does each pre-registered metric add to inlier count? MICRO smoke (17 Oct)

kpbench.sub1_ferranti --split micro --metric <m> [--posed]; pooled sd + cube0.5 + grid rows, 2 flights (each fit on the other flight only), so a smoke, not a verdict. Kill = dAUROC and dPR-AUC both < 0.01.

| metric | all queries dAUROC / dPR-AUC | posed only dAUROC / dPR-AUC | MICRO read |
|---|---|---|---|
| hull | 0.0054 / 0.0209 | 0.0068 / 0.0324 | adds (PR-AUC only) |
| crlb | -0.0318 / -0.0246 | -0.0424 / -0.0594 | kill on MICRO (worse than count alone) |
| inl_ratio | 0.0058 / 0.0029 | 0.0082 / -0.0022 | kill on MICRO |
| reproj_med | 0.1053 / 0.1411 | 0.1392 / 0.2459 | adds most; quality signal, not uniformity |
| depth_iqr | 0.0367 / 0.0174 | 0.0486 / 0.0185 | adds |

Reading: crlb (pose covariance trace, the plan's guard candidate) does not add to inlier count on MICRO; reproj_med adds
+0.14 AUROC on posed queries but measures fit quality, not spread, so it is a SUB1 metric only if uniform pickers lower it
(kpbench.sub1 z on DEV). Next: the same 5 runs on DEV on the Euler login node (Q187 plus --metric loop), waiting for VPN.
