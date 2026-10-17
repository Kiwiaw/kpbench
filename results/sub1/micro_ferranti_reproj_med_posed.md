# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = reproj_med (column "hull" below), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.6162 | 0.8356 | 0.0667 | 0.3925 | 0.4549 | 0.5917 | 0.1992 |
| grid | 371 | 93 | 0.7005 | 0.6959 | 0.8394 | 0.1389 | 0.5207 | 0.6046 | 0.7416 | 0.2209 |
| sd | 367 | 108 | 0.6031 | 0.7357 | 0.7758 | 0.1727 | 0.4237 | 0.69 | 0.7138 | 0.2901 |
| pooled | 1097 | 256 | 0.6668 | 0.6888 | 0.806 | 0.1392 | 0.4446 | 0.5978 | 0.6905 | 0.2459 |

**reproj_med adds over inlier count: dAUROC +0.1392, dPR-AUC +0.2459**
