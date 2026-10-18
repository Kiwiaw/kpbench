# T4 step 3: dppr design (SUB3, 18 Oct 2026)

Design by one design review before coding (T4 rule). Context: every in-grammar rep5c tune (234 DEV trials over dpp, cube,
up, win) tops out at rep5 0.458 at flat fail 14.41 % (dpp0.5: 0.4527); per-query rep5 predicts flat failure (AUROC 0.864).

## Picker
- dppr = dpp with quality q_i = exp(-alpha * rank_i / 100) * r_i^beta, same depth gate, same 3D Gaussian kernel, same greedy MAP.
- r_i = track-count repeatability: m_i = number of the past frames q-1, q-2, q-3 (fewer at flight start) in which candidate i
  has a mutual nearest-neighbour descriptor match with cosine >= tau; r_i = (m_i + 1) / (M_eff + 1), in (0, 1].
- Past frames only: q+1..q+5 overlap the rep5 target frame q+5 (beta would be tuned on a leaked proxy), and a live drone has no
  future frames. No pose, no GT, no learned model (the LightGBM re-ranker was trained on DEV flights and would leak).
- Grammar: `dppr<sigma>a<alpha>b<beta>t<tau>` (kpbench style, no underscores), e.g. dppr0.5a1b1t0.8. beta = 0 gives dpp0.5
  exactly (r^0 = 1, same greedy).

## Tune (family dppr, study dppr_rep5c_dev1, 24 trials, maximise rep5 s.t. flat fail <= 14.41)
- sigma log [0.25, 1.0], alpha [0, 2], beta [0, 3], tau {0.7, 0.8, 0.9}; M fixed 3.
- Seeds: (0.5, 1, 0, 0.8) = dpp0.5; (0.5, 1, 1, 0.8); (0.5, 1, 2, 0.8); (0.5, 0, 1, 0.8) = rep only.

## Ablation (label: ABLATION)
- env KPBENCH_DPPR_QMODE = both | quality | rep: quality forces beta 0 (= dpp), rep forces alpha 0 (rank ignored).
- The claim holds only if 'both' beats each single mode on DEV at flat fail <= 14.41 %.

## MICRO smoke (3 pickers: beta 0, beta 1, rep only) and kill
- (a) dppr0.5a1b0t0.8 equals dpp0.5 rep5 / fail to 4 dp; (b) beta 1 changes the pick set on >= 5 % of queries and r_i is not
  degenerate (std > 0.05 per frame); (c) runtime <= 2x dpp0.5.
- Kill: any check fails -> bug or uninformative r, the 24-trial study is not queued.
- Dispatch kill (13:15): all 3 smoke rep5 <= dpp0.5 rep5 or fail above it -> drop dppr, rep5 stays a predictor only.
