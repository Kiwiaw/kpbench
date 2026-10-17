# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.7418 | 0.7692 | 0.0003 | 0.3925 | 0.4918 | 0.4908 | 0.0983 |
| grid | 371 | 93 | 0.7005 | 0.732 | 0.7148 | 0.0143 | 0.5207 | 0.5482 | 0.5491 | 0.0284 |
| sd | 367 | 108 | 0.6031 | 0.601 | 0.5726 | -0.0305 | 0.4237 | 0.3974 | 0.4052 | -0.0185 |
| pooled | 1097 | 256 | 0.6668 | 0.684 | 0.6736 | 0.0068 | 0.4446 | 0.451 | 0.477 | 0.0324 |

**hull adds over inlier count: dAUROC +0.0068, dPR-AUC +0.0324**
