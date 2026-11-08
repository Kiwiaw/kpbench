# SUB1 Ferranti baseline on fresh, K 400 (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = pcov (column "hull" below)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 13440 | 69 | 0.9738 | 0.938 | 0.9699 | -0.0039 | 0.3814 | 0.2368 | 0.3687 | -0.0127 |
| dpp0.5 | 13440 | 76 | 0.9271 | 0.9493 | 0.9418 | 0.0147 | 0.3607 | 0.2566 | 0.3434 | -0.0173 |
| pooled | 26880 | 145 | 0.9489 | 0.945 | 0.9487 | -0.0002 | 0.3694 | 0.2491 | 0.3533 | -0.0161 |

**KILL: pcov adds < 0.01 AUROC and PR-AUC over inlier count**
