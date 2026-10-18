# SUB1 pre-registration v2: eight candidate metrics, one DEV rule (18 Oct 2026, before any DEV number of the new metrics)

v1 (docs/sub1_metrics.md, 12 Oct) picked hull on DEV (cube0.5 - sd +0.018, z 8.6, AUROC 0.91). Since then: crlb and inl_ratio
add nothing over inlier count (MICRO), reproj_med adds +0.14 AUROC posed (MICRO) but is a pose-confidence signal (scout ed. 5:
Ferranti family), hull has no picker-level link (Spearman +0.24 over 60 win trials, -0.49 over 48 up/cube). v2 fixes the list
and the rule before the DEV rows with the new columns exist. The picker-level Spearman is dropped as a test (scout (f)).

## Candidates (per query, kpbench.metrics MCOLS; no-pose fill in brackets)
| # | name | role | dir | no pose |
|---|---|---|---|---|
| 1 | hull | v1 winner, query-level | + | 0 |
| 2 | reproj_med | baseline (Ferranti-family confidence) | - | 3 |
| 3 | rep5 | baseline (repeatability, needs true pose: diagnostic only) | + | 0 |
| 4 | crlb | recorded negative (MICRO: adds < 0.01) | - | DEV 99th pct |
| 5 | inl_ratio | recorded negative (MICRO: adds < 0.01) | + | 0 |
| 6 | cells | NEW: share of 8 x 6 image cells (80 px) with >= 1 inlier | + | 0 |
| 7 | ent | NEW: entropy of inlier counts over the 48 cells / log 48 | + | 0 |
| 8 | offplane | NEW: RMS distance of inliers to their own TLS plane / median depth (3D spread off the plane) | + | 0 |

## DEV rule (12 DEV flights, pickers sd, cube0.5, dpp0.5 on one node, K 100, LightGlue)
A metric is a SUB1 result only if all three hold:
1. Adds over inlier count: kpbench.sub1_ferranti leave-one-flight-out, pooled dAUROC >= 0.01 and dPR-AUC >= 0.01 (all queries).
2. Uniform lowers error: paired d_q = m(dpp0.5) - m(sd), oriented, block-bootstrap z >= 2.5 (Bonferroni ~8).
3. Picker-level link (replaces Spearman): on the flat frames where exactly one of dpp0.5 / sd fails (and separately dpp0.5 /
   cube0.5), the posed picker has the better value in >= 70 % of frames (ties dropped), sign-test p < 0.05.
Baselines: a new metric (6-8) must beat reproj_med AND rep5 on rule 1's dAUROC to be called new. Among those passing, the
largest z in rule 2 is the primary metric; exactly one goes to HELD_OUT (v1 confirmation rule, one shot).

## Kill (dispatch 2026-10-18 13:15)
No candidate reaches hull's DEV AUROC 0.91 and none passes rule 3 -> SUB1 stays query-level, hull is the paper metric.

## Smoke (laptop, 18 Oct)
- cells / ent / offplane: synthetic 60 inliers, spread 40 px vs 300 px: cells 0.083 -> 0.771, ent 0.351 -> 0.898, hull
  0.014 -> 0.723, offplane 0.050 both (points on a 1 m depth slab): columns behave; MICRO rows predate them (need Euler).
- rep5 on the stored MICRO metric rows (sd, cube0.5, grid): pooled dAUROC +0.015, dPR-AUC +0.004 (all); +0.020 / -0.022
  posed (results/sub1/micro_ferranti_rep5*.md). reproj_med posed on the same rows: +0.139 / +0.246.
