# dpp2d at K200: MICRO picker smoke and fill fix (T20, SUB4 lane, 2026-10-27)

Question (dispatch 2026-10-27T13:00, SUB4): is the pixel-kernel DPP family ready for a matched-budget K200 tune
(`dpp2d_K200_dev1`) next to the 3D re-tune best 12.18 % flat fail?

Pose-level MICRO evaluate is impossible on the laptop (no MICRO flight images under local_data), so this is a
picker-level smoke on the cached MICRO candidates (393 frames, raco): `python tools/picker_smoke.py <pickers> --K 200`.

## Finding: early-stopping DPP filled K with depthless candidates

`dpp_greedy` stops when the best remaining gain is <= 1e-12 (kernel rank used up, or quality q_i = exp(-alpha rank/100)
tiny at high alpha). `fill()` then topped up to K from ALL candidates, including ones without depth. evaluate.py drops
depthless picks from map keyframes, so such a picker ran with fewer than K map points: a hidden budget handicap.

Frames (of 393 MICRO) with >= 1 depthless pick, old code:

| picker | K100 | K200 |
|---|---|---|
| dpp2d120a1 (dpp2d_dev1 best, 17.52) | 0 | 33 |
| dpp2d179a0.218 (2nd, 17.63) | 0 | 35 |
| dpp2d74a1 (seed, 18.16) | 0 | 0 |
| dpp2d115a4.76 (5th, 18.59) | 11 | 40 |
| dpp2d195a0.201 (7th, 18.80) | 0 | 36 |
| dpp2d40a1 (6th, 18.80) | 0 | 0 |
| dpp2d183a4.57 (12th, 21.37) | 27 | 40 |
| dpp0.5, dpp0.678a0.434, dpp0.601a0.283, dpp0.5a0.5 (3D) | 0 | 0 |

Mean depthless picks per frame stay small (<= 0.2), so the effect on any published K100 number is small, but at
K200 the large-sigma 2D trials would have been handicapped in exactly the matched-budget comparison the tune is for.

## Fix

pickers.py: dpp, dpp2d, dpps, dppm/dppl and dppr/dppmr now top up with `fill(..., ok)` (depth-eligible candidates only),
as nms already did. Picks are unchanged whenever the greedy does not stop early.

After the fix (K200, results/micro/picker_smoke_K200_fillfix.json): dpp2d120a1, dpp2d179a0.218, dpp2d183a4.57 fill 0 of
393 frames short; dpp0.5 picks identical (pixel spread 218.8, min pair distance 2.43 px, before and after).

K200 picker smoke (before the fix, results/micro/picker_smoke_K200.json): mean pixel spread dpp2d120a1 228.5,
dpp2d74 234.7, dpp0.601a0.283 221.8, cube0.369 219.6, dpp0.5 218.8, sd 200.3; dpp2d ~90-180 ms per frame.

## Consequences

- dpp2d_dev1 (K100) trials dpp2d115a4.76 and dpp2d183a4.57 ran with a few depthless map picks on 11 / 27 of 393 MICRO
  frames; the trial best dpp2d120a1 (17.52) and the 3D incumbent are not affected at K100. The 17.52 vs 14.42 headline
  stands; ranks 5 and 12 are not exactly reproducible with the new code.
- K200 DPP results so far (dpp_K200_dev1, dpp_K400_dev1, best 12.18 / 13.89): their 3D trials did not stop early on MICRO,
  but a DEV check needs Euler (kpbench.audit row recompute, SUB2).
- queue/_held/Qt-dpp2d_K200_dev1.json (agent repo) waits for this fix to be on Euler (euler/push.sh) and for VPN.
  Kill: best of the first 12 trials <= 13.18 flat -> 'spread must be in 3D' withdrawn.
