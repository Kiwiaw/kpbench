# SUB1 Ferranti baseline on fresh, K 400 (logistic, leave-one-flight-out; fail = positive for PR-AUC)

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 13440 | 69 | 0.9738 | 0.9044 | 0.9549 | -0.0189 | 0.3814 | 0.3419 | 0.4134 | 0.032 |
| dpp0.5 | 13440 | 76 | 0.9271 | 0.8601 | 0.9234 | -0.0037 | 0.3607 | 0.2869 | 0.3453 | -0.0154 |
| pooled | 26880 | 145 | 0.9489 | 0.8801 | 0.9375 | -0.0114 | 0.3694 | 0.3222 | 0.3866 | 0.0172 |

**hull adds over inlier count: dAUROC -0.0114, dPR-AUC +0.0172**
