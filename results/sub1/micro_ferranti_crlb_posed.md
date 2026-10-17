# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = crlb (column "hull" below), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.6318 | 0.8091 | 0.0402 | 0.3925 | 0.3207 | 0.5023 | 0.1098 |
| grid | 371 | 93 | 0.7005 | 0.589 | 0.6639 | -0.0366 | 0.5207 | 0.3239 | 0.4437 | -0.077 |
| sd | 367 | 108 | 0.6031 | 0.3416 | 0.4909 | -0.1122 | 0.4237 | 0.2314 | 0.3301 | -0.0936 |
| pooled | 1097 | 256 | 0.6668 | 0.5218 | 0.6244 | -0.0424 | 0.4446 | 0.2716 | 0.3852 | -0.0594 |

**KILL: crlb adds < 0.01 AUROC and PR-AUC over inlier count**
