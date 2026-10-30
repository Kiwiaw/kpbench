# E1 power pre-check and hero CI (no compute)

31 leaderboard flights, K100, same map, LightGlue; 20-frame blocks, 1000 boots, seed 0 (`python tools/hero_ci.py`). Leaderboard rows, not same-node gate runs: a CI for the page numbers, not a gate verdict.

| pair | band | n | cand % | sd % | diff pt [95 %] | flights better/worse/tied | sign p |
|---|---|---|---|---|---|---|---|
| cube0.5 vs sd | flat | 1813 | 12.52 (227) | 16.27 (295) | -3.75 [-6.31, -1.46] | 7/6/14 | 0.5000 |
| cube0.5 vs sd | very flat | 1058 | 18.34 (194) | 24.10 (255) | -5.77 [-9.32, -2.46] | 6/6/9 | 0.6128 |
| dpp0.5 vs sd | flat | 1813 | 10.59 (192) | 16.27 (295) | -5.68 [-8.09, -3.53] | 11/3/13 | 0.0287 |
| dpp0.5 vs sd | very flat | 1058 | 16.07 (170) | 24.10 (255) | -8.03 [-11.19, -5.18] | 10/2/9 | 0.0193 |

- Hero 24.1 -> 18.3 very flat: -5.77 pt [-9.32, -2.46], CI excludes 0 (route A style), so the kill is not met.
- Caveat: the flight sign test does not back the cube number (6/6/9, p 0.61; flat 7/6/14, p 0.50). The gain sits in a few flights. dpp0.5 vs sd holds on both (very flat 10/2/9, p 0.019), but dpp0.5 was chosen on part of these flights.
- FRESH-1 count rule (sd very-flat fails < 62): not evaluable. FRESH labels and sd rows are not on the laptop; FRESH-1 has 14 flights, not 13 (13 hold flat frames).
- E1 is not in queue/: Qg-cube0.5-vs-sd-fresh was never written, so there is nothing to hold. Base rate argument stays: the incumbent's 2.99 % of 2075 is 62 frames, and a 10 % drop is about 6 frames.
