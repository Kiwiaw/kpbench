# SUB1 Ferranti baseline on micro (logistic, leave-one-flight-out; fail = positive for PR-AUC); metric = inl_ratio (column "hull" below), posed queries only

| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| cube0.5 | 359 | 55 | 0.7689 | 0.7696 | 0.7709 | 0.002 | 0.3925 | 0.3227 | 0.412 | 0.0195 |
| grid | 371 | 93 | 0.7005 | 0.7455 | 0.7213 | 0.0208 | 0.5207 | 0.5426 | 0.5337 | 0.013 |
| sd | 367 | 108 | 0.6031 | 0.6313 | 0.6111 | 0.008 | 0.4237 | 0.4305 | 0.4254 | 0.0017 |
| pooled | 1097 | 256 | 0.6668 | 0.697 | 0.675 | 0.0082 | 0.4446 | 0.4317 | 0.4424 | -0.0022 |

**KILL: inl_ratio adds < 0.01 AUROC and PR-AUC over inlier count**
