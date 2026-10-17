# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = depth_iqr (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.8149 | 0.8374 | -0.0198 | 0.7358 | 0.6785 | 0.7222 | -0.0136 |
| grid | 393 | 115 | 0.7578 | 0.8169 | 0.7887 | 0.0309 | 0.6745 | 0.7476 | 0.6896 | 0.0151 |
| sd | 393 | 134 | 0.6801 | 0.8267 | 0.751 | 0.0709 | 0.62 | 0.7966 | 0.6595 | 0.0395 |
| pooled | 1179 | 338 | 0.7477 | 0.8327 | 0.7844 | 0.0367 | 0.6616 | 0.7537 | 0.679 | 0.0174 |

**depth_iqr adds over inlier count: dAUROC +0.0367, dPR-AUC +0.0174**
