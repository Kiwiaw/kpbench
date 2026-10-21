# Failure prediction on fresh flat queries, raco_dpp0.5_K100_same_lg_g-dppr0.486a0.453b1.02t0.9-fresh (2075 queries, 62 fails = 2.99 %)

AURC and risk in fail % (lower = better); perfect ranking AURC 0.05. Fitted scores are leave-one-flight-out logistic. rep5 needs the true pose (measurement, not deployable).

| score | AUROC | AURC | E-AURC | risk @50 % | risk @80 % | risk @90 % | AURC - count [95 %] |
|---|---|---|---|---|---|---|---|
| rep5 | 0.864 | 0.80 | 0.75 | 0.48 | 0.60 | 1.28 | +0.43 [+0.11, +0.95] |
| count | 0.938 | 0.37 | 0.32 | 0.39 | 0.24 | 0.32 | - |
| count+rep5 | 0.913 | 0.69 | 0.64 | 0.39 | 0.24 | 0.48 | +0.32 [+0.01, +1.00] |

**KILL: best combination count+rep5 AURC +0.32 [+0.01, +1.00] pt vs inlier count, no better**
