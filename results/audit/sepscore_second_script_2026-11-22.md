# kpbench.sepscore vs an independent second script, synthetic dumps only (SUB2, tools/sepscore_check2.py)

No real picks were read and no sep value of any real run exists. 12 fake flights x 100 queries (80 flat) per arm. Cells: scorer / second script.

| set | rule 1 dAUROC, dPR-AUC, pass | rule 2 d, z, pass | rule 3 vs strong n-win-tie p; vs other; pass | rule 4 neg/pos/zero p pass | all four | agree |
|---|---|---|---|---|---|---|
| planted | +0.2956, +0.2675, yes / +0.2956, +0.2674, yes | +0.1995, 90.38, yes / +0.1995, 88.93, yes | 678-658-0 2.1e-166; 479-443-0 3e-90; yes / 678-658-0 2.1e-166; 479-443-0 3e-90; yes | 12/0/0 0.0002 yes / 12/0/0 0.0002 yes | yes / yes | yes |
| null | -0.0002, -0.0005, no / -0.0002, -0.0003, no | -0.0027, -0.68, no / -0.0027, -0.69, no | 312-153-0 0.78; 324-164-0 0.87; no / 312-153-0 0.78; 324-164-0 0.87; no | 5/7/0 0.8062 no / 5/7/0 0.8062 no | no / no | yes |
| tie | +0.2782, +0.2565, yes / +0.2782, +0.2564, yes | +0.1826, 24.85, yes / +0.1826, 24.99, yes | 678-603-57 5.4e-153; 479-402-44 9.6e-82; yes / 678-603-57 5.4e-153; 479-402-44 9.6e-82; yes | 11/0/1 0.0005 yes / 11/0/1 0.0005 yes | yes / yes | yes |

Agreement rule (fixed before the run): same pass / fail on every rule; counts identical; p within 1e-9 (rule 4: 1e-4, the scorer rounds it); dAUROC and dPR-AUC within 0.002; z within 10 %.
Sign-test thresholds by flights left after dropped zeros (one-sided p < 0.05): 5 of 5, 6 of 6, 7 of 7, 7 of 8, 8 of 9, 9 of 10, 9 of 11, 10 of 12.

**PASS: the second script agrees with kpbench.sepscore on all rules of all three sets**
