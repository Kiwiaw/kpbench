# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = reproj_med (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.7688 | 0.8984 | 0.0412 | 0.7358 | 0.7409 | 0.8121 | 0.0763 |
| grid | 393 | 115 | 0.7578 | 0.7586 | 0.8701 | 0.1123 | 0.6745 | 0.7301 | 0.8189 | 0.1444 |
| sd | 393 | 134 | 0.6801 | 0.7869 | 0.8193 | 0.1392 | 0.62 | 0.7812 | 0.799 | 0.179 |
| pooled | 1179 | 338 | 0.7477 | 0.7646 | 0.853 | 0.1053 | 0.6616 | 0.7413 | 0.8027 | 0.1411 |

**reproj_med adds over inlier count: dAUROC +0.1053, dPR-AUC +0.1411**
