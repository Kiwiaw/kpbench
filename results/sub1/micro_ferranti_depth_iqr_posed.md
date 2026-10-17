# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = depth_iqr (column "hull" below), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.685 | 0.737 | -0.0319 | 0.3925 | 0.2564 | 0.3822 | -0.0103 |
| grid | 371 | 93 | 0.7005 | 0.7802 | 0.7387 | 0.0382 | 0.5207 | 0.6471 | 0.5482 | 0.0275 |
| sd | 367 | 108 | 0.6031 | 0.7677 | 0.6908 | 0.0877 | 0.4237 | 0.6716 | 0.4715 | 0.0478 |
| pooled | 1097 | 256 | 0.6668 | 0.7752 | 0.7154 | 0.0486 | 0.4446 | 0.6157 | 0.4631 | 0.0185 |

**depth_iqr adds over inlier count: dAUROC +0.0486, dPR-AUC +0.0185**
