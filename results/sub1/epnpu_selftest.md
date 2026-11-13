# SUB1 E-d self-test: does epnpu_t order pick sets like the true pose error? (pre-registered 2026-11-13 13:40, before the first run)

Why: 0 of 11 pre-registered SUB1 metrics passed (results/sub1/REPORT_sub1_close.md). Before a 12th metric (epnpu_t, tools/micro_epnpu_dq3.py) costs Euler time,
it must find a planted effect on synthetic data where the answer is known (dispatch 2026-11-13 13:15, SUB1 step 1).

Setup (tools/epnpu_selftest.py, no flight data): oblique view of a ground plane 3 m below the camera, 5 to 30 m away, relief +-5 cm; 400 candidate points;
three sets of 40 picks: uniform (spread over the image), cluster (one 2D clump), near (nearest depth third only).
True error = RMS camera-centre error over 200 noisy PnP solves (1 px image noise + depth noise 0.005 z^2, kpbench.bench.pnp).
Toy example: if uniform picks give 0.10 m RMS and clustered picks 0.40 m, epnpu_t must also be larger for the cluster set.

Pass rule (both needed):
1. cluster vs uniform and near vs uniform: epnpu_t points the same way as the true error in >= 9 of 10 scenes, each contrast.
2. Spearman rank correlation of epnpu_t with log10 RMS error over all 30 scene-sets >= 0.7.

Kill: either rule fails -> E-d dropped unrun, SUB1 stays KILL (final).

## Result (10 scenes x 200 trials, seed 0)

| scene | uniform epnpu_t / log10 RMS | cluster epnpu_t / log10 RMS | near epnpu_t / log10 RMS |
|---|---|---|---|
| 0 | -0.522 / -1.392 | 0.920 / -0.330 | -0.471 / -1.252 |
| 1 | -0.580 / -1.465 | 0.110 / -0.677 | -0.527 / 0.181 |
| 2 | -0.584 / -1.422 | 0.565 / -0.574 | -0.504 / -1.284 |
| 3 | -0.559 / -1.440 | 0.700 / -0.444 | -0.548 / -1.342 |
| 4 | -0.572 / -1.478 | 0.518 / -0.626 | -0.511 / -1.286 |
| 5 | -0.571 / -1.469 | -0.395 / -1.187 | -0.541 / -1.342 |
| 6 | -0.552 / -1.462 | 0.602 / -0.505 | -0.576 / -1.354 |
| 7 | -0.547 / -1.458 | 0.495 / -0.614 | -0.552 / -1.326 |
| 8 | -0.553 / -1.449 | 0.808 / -0.404 | -0.515 / -1.284 |
| 9 | -0.541 / -1.451 | 0.696 / -0.420 | -0.493 / -1.248 |
- cluster vs uniform: true error larger in 10 / 10 scenes, epnpu_t larger in 10 / 10, same direction in 10 / 10
- near vs uniform: true error larger in 10 / 10 scenes, epnpu_t larger in 8 / 10, same direction in 8 / 10
- Spearman rho(epnpu_t, log10 RMS) over 30 scene-sets: 0.869

**Reading: FAIL: kill met -> E-d dropped unrun, SUB1 stays KILL.**

Tool: tools/epnpu_selftest.py.

Note (engineer): the misses are both on near vs uniform, where the true error gap is 0.1-0.2 log10 (about 1.3-1.6x) but the epnpu_t gap only 0.01-0.08; epnpu_t under-weights near-field depth concentration, the very effect the MICRO depth-tercile table shows (sd 0.479 vs cube0.5 0.170). Rule 2 (rho 0.869) passed, rule 1 failed: the pre-registered rule needs both, so the kill stands; no euler sbatch line was added.
