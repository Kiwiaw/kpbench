# SUB3 gate: AUROC of per-query rep5 for pose failure (kpbench.sub3_auroc)

Stored K = 100 LightGlue rows; AUROC = P(rep5 of a failed query < rep5 of a posed one); 95 % CI = 20-query block bootstrap.

| split | frames | picker | queries | fails | rep5 fail / posed | AUROC [95 %] |
|---|---|---|---|---|---|---|
| micro | flat | dpp0.5 | 393 | 89 | 0.170 / 0.414 | 0.843 [0.772, 0.913] |
| micro | flat | cube0.5 | 393 | 94 | 0.148 / 0.355 | 0.827 [0.743, 0.913] |
| micro | flat | sd | 393 | 127 | 0.130 / 0.405 | 0.837 [0.762, 0.904] |
| micro | all | dpp0.5 | 1315 | 100 | 0.184 / 0.554 | 0.899 [0.842, 0.950] |
| micro | all | cube0.5 | 1315 | 116 | 0.162 / 0.533 | 0.901 [0.840, 0.945] |
| micro | all | sd | 1315 | 154 | 0.163 / 0.573 | 0.901 [0.837, 0.950] |
| dev | flat | dpp0.5 | 936 | 135 | 0.188 / 0.496 | 0.864 [0.809, 0.910] |
| dev | flat | cube0.5 | 936 | 146 | 0.141 / 0.462 | 0.884 [0.828, 0.931] |
| dev | flat | sd | 936 | 208 | 0.188 / 0.514 | 0.842 [0.770, 0.904] |
| dev | all | dpp0.5 | 8218 | 167 | 0.201 / 0.572 | 0.921 [0.884, 0.951] |
| dev | all | cube0.5 | 8218 | 197 | 0.159 / 0.564 | 0.935 [0.902, 0.959] |
| dev | all | sd | 8218 | 297 | 0.207 / 0.592 | 0.898 [0.847, 0.940] |

Check: DEV flat dpp0.5 135 / 936 = 14.42 % fail = the DEV flat fail 14.41 % the tunes use (repo labels, 7 Oct).

Verdict (17 Oct, engineer): gate PASSES. DEV flat dpp0.5 AUROC 0.864, lower bound 0.809 >= 0.6 (MICRO 0.843 [0.772, 0.913]);
failed flat queries re-find 0.19 of their points 5 frames later, posed ones 0.50. T4 step 3 (dppr, labelled ABLATION) is unlocked.
Caveats: (1) query-level only: hull also had AUROC 0.91 at query level and no picker-level link (rho +0.24 / -0.49), so a
picker that raises mean rep5 need not lower fail; the dppr tune itself is the picker-level test. (2) rep5 uses true poses and
depth (it is a measurement, not a test-time signal); dppr must predict it from image evidence. (3) part of the link is
mechanical: few re-found points -> few matches -> no pose.
