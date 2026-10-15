# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.8362 | 0.8591 | 0.0019 | 0.7358 | 0.7318 | 0.7758 | 0.04 |
| grid | 393 | 115 | 0.7578 | 0.7817 | 0.7686 | 0.0108 | 0.6745 | 0.6966 | 0.6921 | 0.0176 |
| sd | 393 | 134 | 0.6801 | 0.669 | 0.6588 | -0.0213 | 0.62 | 0.5421 | 0.6138 | -0.0062 |
| pooled | 1179 | 338 | 0.7477 | 0.7558 | 0.7531 | 0.0054 | 0.6616 | 0.6373 | 0.6825 | 0.0209 |

**hull adds over inlier count: dAUROC +0.0054, dPR-AUC +0.0209**
