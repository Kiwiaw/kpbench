# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = inl_ratio (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 393 | 89 | 0.8572 | 0.8642 | 0.8583 | 0.0011 | 0.7358 | 0.7328 | 0.739 | 0.0032 |
| grid | 393 | 115 | 0.7578 | 0.794 | 0.7737 | 0.0159 | 0.6745 | 0.7001 | 0.686 | 0.0115 |
| sd | 393 | 134 | 0.6801 | 0.7052 | 0.6864 | 0.0063 | 0.62 | 0.6351 | 0.6241 | 0.0041 |
| pooled | 1179 | 338 | 0.7477 | 0.7711 | 0.7535 | 0.0058 | 0.6616 | 0.6718 | 0.6645 | 0.0029 |

**KILL: inl_ratio adds < 0.01 AUROC and PR-AUC over inlier count**
