# Flight-level evidence, fresh gates (2026-11-02)

tools/class_bootstrap.py: cluster CI = flights resampled with replacement (1000 draws, seed 0), frame-weighted pooled flat-fail difference in points; block CI = the gate json (20-frame blocks); Wilcoxon = signed-rank on per-flight flat fail %, one-sided (candidate lower), tied flights dropped. Verdicts are the gate rule (kpbench.stats.gate), unchanged. [T] = rows absent here, json only.

| candidate vs incumbent | K | flat fail % cand / inc (n flat) | block CI pt | cluster CI pt | flights b/w/t | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|
| dpp0.601a0.283 vs dpp0.5 | 200 | 2.51 / 2.65 (2075) | -0.14 [-0.71, +0.34] | -0.14 [-0.63, +0.26] | 2/3/8 | 0.812 | 0.406 | REVERT |
| cube0.5 vs dpp0.5 | 400 | 2.75 / 3.08 (2075) | -0.34 [-1.00, +0.40] | -0.34 [-1.05, +0.30] | 5/2/6 | 0.227 | 0.289 | REVERT |
| dppr0.5a1b2t0.8 vs dpp0.5 | 100 | 2.89 / 2.99 (2075) | -0.10 [-0.71, +0.54] | -0.10 [-0.59, +0.49] | 4/3/6 | 0.500 | 0.711 | REVERT |
| dppr0.486a0.453b1.02t0.9 vs dpp0.5 | 100 | 3.04 / 2.99 (2075) | +0.05 [-0.45, +0.61] | +0.05 [-0.57, +0.64] | 3/5/5 | 0.855 | 0.578 | REVERT |
| dppr0.317a1.6b2.01t0.9 vs dpp0.5 | 100 | 3.23 / 2.99 (2075) | +0.24 [-0.31, +0.84] | +0.24 [-0.42, +1.06] | 4/3/6 | 0.500 | 0.766 | REVERT |
| dpp0.317a1.6 vs dpp0.5 | 100 | 3.47 / 2.99 (2075) | +0.48 [-0.29, +1.25] | +0.48 [-0.26, +1.33] | 3/5/5 | 0.855 | 0.926 | REVERT |
| win23+dpp0.666a0.526 vs dpp0.5 | 100 | 3.95 / 2.99 (2075) | +0.96 [-0.22, +2.26] | +0.96 [-0.80, +3.29] | 3/5/5 | 0.855 | 0.902 | REVERT |
| cube0.5 vs sd | 100 | 4.58 / 7.13 (2075) | -2.55 [-4.80, -0.34] | -2.55 [-5.36, +0.82] | 5/3/5 | 0.363 | 0.230 | KEEP |

Read: a cluster CI that covers 0 while the block CI does not means the frame-level gain rests on a few flights; the paper then states the gain as frame-weighted only (R22).
