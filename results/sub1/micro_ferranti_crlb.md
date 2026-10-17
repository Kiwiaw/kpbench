# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = crlb (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.7396 | 0.882 | 0.0248 | 0.7358 | 0.5062 | 0.7839 | 0.0481 |
| grid | 393 | 115 | 0.7578 | 0.6758 | 0.7282 | -0.0296 | 0.6745 | 0.5261 | 0.6283 | -0.0462 |
| sd | 393 | 134 | 0.6801 | 0.3677 | 0.5921 | -0.088 | 0.62 | 0.3042 | 0.5588 | -0.0612 |
| pooled | 1179 | 338 | 0.7477 | 0.6211 | 0.7159 | -0.0318 | 0.6616 | 0.4333 | 0.637 | -0.0246 |

**KILL: crlb adds < 0.01 AUROC and PR-AUC over inlier count**
