# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = rep5 (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.8244 | 0.8635 | 0.0063 | 0.7358 | 0.54 | 0.7485 | 0.0127 |
| grid | 393 | 115 | 0.7578 | 0.7984 | 0.77 | 0.0122 | 0.6745 | 0.6151 | 0.6809 | 0.0064 |
| sd | 393 | 134 | 0.6801 | 0.7937 | 0.7019 | 0.0218 | 0.62 | 0.6467 | 0.6218 | 0.0018 |
| pooled | 1179 | 338 | 0.7477 | 0.807 | 0.7626 | 0.0149 | 0.6616 | 0.6075 | 0.6657 | 0.0041 |

**rep5 adds over inlier count: dAUROC +0.0149, dPR-AUC +0.0041**
