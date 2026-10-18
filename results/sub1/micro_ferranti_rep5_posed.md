# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = rep5 (column "hull" below), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.7608 | 0.7792 | 0.0103 | 0.3925 | 0.3446 | 0.4074 | 0.0149 |
| grid | 371 | 93 | 0.7005 | 0.7689 | 0.7159 | 0.0154 | 0.5207 | 0.5105 | 0.5167 | -0.004 |
| sd | 367 | 108 | 0.6031 | 0.7255 | 0.6279 | 0.0248 | 0.4237 | 0.4516 | 0.3982 | -0.0255 |
| pooled | 1097 | 256 | 0.6668 | 0.7585 | 0.6869 | 0.0201 | 0.4446 | 0.4547 | 0.4225 | -0.0221 |

**rep5 adds over inlier count: dAUROC +0.0201, dPR-AUC -0.0221**
