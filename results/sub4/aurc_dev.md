# Failure prediction on dev flat queries, dpp0.5 (936 queries, 133 fails = 14.21 %)

AURC and risk in fail % (lower = better); perfect ranking AURC 1.07. Fitted scores are leave-one-flight-out logistic. rep5 needs the true pose (measurement, not deployable).

| score | AUROC | AURC | E-AURC | risk @50 % | risk @80 % | risk @90 % | AURC - count [95 %] |
|---|---|---|---|---|---|---|---|
| rep5 | 0.865 | 3.33 | 2.26 | 0.85 | 6.81 | 10.81 | +0.39 [-0.45, +1.41] |
| count | 0.885 | 2.94 | 1.87 | 0.64 | 6.14 | 9.14 | - |
| geom2d | 0.600 | 14.87 | 13.80 | 12.18 | 8.54 | 9.50 | +11.93 [+4.20, +19.14] |
| count+geom2d | 0.880 | 3.22 | 2.15 | 0.64 | 6.28 | 9.74 | +0.28 [-0.42, +1.18] |
| count+geom2d+rep5 | 0.894 | 3.00 | 1.93 | 0.64 | 5.61 | 8.91 | +0.06 [-0.81, +0.96] |

**KILL: best combination count+geom2d+rep5 AURC +0.06 [-0.81, +0.96] pt vs inlier count, no better**

Notes (engineer 2026-10-19, dispatch 13:10 SUB4): rows = 18 Oct SUB1 eval file of dpp0.5 (936 flat; 14.21 % vs 14.42 % in the gate run). rep5 AUROC 0.865 and count 0.885 match the 18 Oct abstention numbers (0.864 / 0.888). geom2d alone ranks poorly out of flight (AUROC 0.600): the per-flight base rates differ and the LOFO fit does not transfer; hull alone pooled over pickers was 0.84-0.88 (results/sub1/dev_ferranti.md). Reusable on FRESH-1 once the eval rows with metric columns exist there.
