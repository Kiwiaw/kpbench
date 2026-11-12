# E-c: PnP fail % for dq3 (depth quota) vs cube0.5 (3D cells) on MICRO (engineer 2026-11-12, laptop GPU)

## Pre-registered reading (written 2026-11-12 16:45, before any number)
- Set: MICRO split (393 flat queries per T33/J4 notes), K 100, raco, LightGlue, AP3P-RANSAC + LM (kpbench.evaluate protocol), same frames for both arms.
- dq3 = tools/depth_quota_micro.pick_dq(3): candidates cut into equal-count depth terciles, K/3 picks each, strongest first inside a tercile.
- Numbers: flat fail % of both, paired difference dq3 - cube0.5 in points with a 20-frame block bootstrap 95 % CI (1000 draws, seed 0), discordant frames (dq3-only fails / cube-only fails), rep5 of both.
- Reading: descriptive only. A verdict is stated only if the CI excludes 0. Expected half-width ~1.6 pt (0.70 x sqrt(2075/393)), so a tie is the likely outcome and is NOT evidence that the arms are equal.
- Kill (dispatch 16:25): dq3 fail % <= cube0.5 with CI excluding 0 -> depth quota explains the gain (J4 kill, evaluation-only paper, picker spec switches to depth quota).
- Not a gate, not FRESH, no verdict change.

## Status 2026-11-12 16:55: E-c NOT RUN (blocked)
- The laptop MICRO flight folders (C:\Users\Kinga\Desktop\ETH_lab\local_data: poses for oldtown/P001, hospital/P041) are gone (14-day auto-cleanup, local_data_cleanup.ps1); the candidate cache (cache/*.npz) is there, but PnP needs the true poses. Euler unreachable (no VPN).
- Ready to run: `KPBENCH_DATA=<data root> python tools/micro_epnpu_dq3.py` (E-c + E-d in one pass, ~5 min GPU) on Euler or after a MICRO re-copy (needs Kinga's OK). The pre-registered reading above stays fixed.

## Power paragraph for a dq3 vs cube0.5 FRESH-2 arm (T4 step 2, 0 GPU-h, mirrored FRESH-1 rows, tools/dq3_power.py)

| FRESH-1 pair | flat n | fail % cand / inc | diff [block 95 %] | discordant frames | block half-width | iid half-width | inflation | flat frames for 80 % power at 1 pt |
|---|---|---|---|---|---|---|---|---|
| cube0.5 vs sd K100 | 2075 | 4.58 / 7.13 | -2.55 [-4.80, -0.34] | 6.31 % | 2.23 pt | 1.08 pt | 2.06 | 21055 |
| dpp2d120a1 vs dpp0.5 K100 | 2075 | 5.73 / 2.99 | +2.75 [+1.08, +4.90] | 4.10 % | 1.91 pt | 0.87 pt | 2.19 | 15412 |
| cube0.5 vs dpp0.5 K400 | 2075 | 2.75 / 3.08 | -0.34 [-1.00, +0.40] | 1.40 % | 0.70 pt | 0.51 pt | 1.37 | 2061 |
| cube0.5 vs sd K400 | 2075 | 2.75 / 7.13 | -4.39 [-7.01, -2.01] | 5.16 % | 2.50 pt | 0.98 pt | 2.56 | 26403 |
| dpp0.317a1.6 vs dpp0.5 K100 | 2075 | 3.47 / 2.99 | +0.48 [-0.29, +1.25] | 1.73 % | 0.77 pt | 0.57 pt | 1.36 | 2511 |
| dppr0.317 vs dpp0.5 K100 | 2075 | 3.23 / 2.99 | +0.24 [-0.31, +0.84] | 1.49 % | 0.57 pt | 0.53 pt | 1.09 | 1397 |

Needed flat frames for a 1 pt dq3 vs cube0.5 difference at 80 % power: 1397 to 26403 over the 6 mirrored pairs (median 8961). Pool: FRESH-2 ~2774, FRESH-1 + FRESH-2 4849. Pairs at or below 4849: 3 of 6.
Which row fits dq3 vs cube0.5 is not known (no picks or fail rows for dq3 yet; MICRO pick Jaccard 0.291 says the pick sets differ more than cube vs dpp, so the pairs with a score-ranked arm (vs sd) are the closer analogues).
**Reading:** kill rule "needed > 4849 -> drop the dq3 arm" applied to the median pair: MET -> drop the dq3 arm from FRESH-2.
