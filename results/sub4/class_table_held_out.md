# Flight-level evidence, held_out gates (2026-11-02)

tools/class_bootstrap.py: cluster CI = flights resampled with replacement (1000 draws, seed 0), frame-weighted pooled flat-fail difference in points; block CI = the gate json (20-frame blocks); Wilcoxon = signed-rank on per-flight flat fail %, one-sided (candidate lower), tied flights dropped. Verdicts are the gate rule (kpbench.stats.gate), unchanged. [T] = rows absent here, json only.

| candidate vs incumbent | K | flat fail %% cand / inc (n flat) | block CI pt | cluster CI pt | flights b/w/t | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|
| dpp0.5 vs cube0.5 | 100 | 6.50 / 9.24 (877) | -2.74 [-4.90, -0.99] | -2.74 [-4.41, -0.74] | 6/0/9 | 0.016 | 0.016 | KEEP |
| cube0.5 vs sd | 100 | 9.24 / 9.92 (877) | -0.68 [-3.46, +1.83] | -0.68 [-3.90, +2.42] | 3/3/9 | 0.656 | 0.500 | REVERT |

Read: a cluster CI that covers 0 while the block CI does not means the frame-level gain rests on a few flights; the paper then states the gain as frame-weighted only (R22).
