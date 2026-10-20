# SUB1 prereg v2 scorer on dev (8273 queries, 936 flat; uniform dpp0.5 vs strong sd, rule 3 also vs cube0.5)

| metric | dAUROC | dPR-AUC | rule 1 | d (dpp0.5 - sd) | z | rule 2 | within vs sd n / share / p | within vs cube0.5 n / share / p | rule 3 |
|---|---|---|---|---|---|---|---|---|---|
| hull | -0.0180 | +0.0224 | no | +0.0372 | 16.40 | yes | 94 / 0.7128 / 4.5e-05 | 55 / 0.6727 / 0.014 | no |
| reproj_med | -0.0072 | +0.0187 | no | +0.0478 | 2.31 | no | 94 / 0.7021 / 0.00011 | 55 / 0.6727 / 0.014 | no |
| rep5 | +0.0064 | -0.0023 | no | -0.0130 | -5.23 | no | 94 / 0.7416 / 5.7e-06 | 55 / 0.5294 / 0.78 | no |
| crlb | -0.0104 | -0.0132 | no | +0.0018 | 0.44 | no | 94 / 0.6383 / 0.0095 | 55 / 0.6909 / 0.0065 | no |
| inl_ratio | +0.0001 | -0.0062 | no | -0.0218 | -10.90 | no | 94 / 0.6277 / 0.017 | 55 / 0.7273 / 0.001 | no |
| cells | -0.0270 | -0.0036 | no | +0.0491 | 26.43 | yes | 94 / 0.6848 / 0.00051 | 55 / 0.7059 / 0.0046 | no |
| ent | -0.0228 | +0.0087 | no | +0.0363 | 19.95 | yes | 94 / 0.6809 / 0.00059 | 55 / 0.6909 / 0.0065 | no |
| offplane | +0.0265 | +0.0330 | yes | -6.5388 | -1.02 | no | 94 / 0.7473 / 2.5e-06 | 55 / 0.7222 / 0.0015 | yes |
| sel_logdet | missing | | | | | | | | |
| sel_hull | missing | | | | | | | | |

**no candidate passes all three rules -> SUB1 stays query-level (prereg kill)**

Run 2026-10-20 on the Euler login node over the 18 Oct DEV metric rows (sd, cube0.5, dpp0.5 on one node; rule 1 pooled over
the 9 picker files that carry each column). sel_logdet / sel_hull need a new evaluate --metrics run (GPU).
Reading: spread metrics move the right way under the uniform picker (cells z 26.4, ent 20.0, hull 16.4) but add nothing over
inlier count (dAUROC -0.027 / -0.023 / -0.018) and miss rule 3 vs cube0.5 (0.71 / 0.69 / 0.67 < 0.70 or p >= 0.05 not met
together). offplane passes rules 1 and 3 (dAUROC +0.027, dPR-AUC +0.033; within 0.75 / 0.72) but not rule 2: its mean d -6.54
comes from 12 sd queries with offplane > 1 (max 54026, median depth near 0); post-hoc with values clipped at 1 the uniform
picker still has LOWER offplane, d -0.008, z -4.47, so uniformity does not raise it. Kill clause "none passes rule 3" is not
met (offplane), but no candidate passes all three: no SUB1 metric from prereg v2 columns 1-8.
