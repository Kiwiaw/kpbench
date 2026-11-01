# Hero trade-off: flat-fail gain next to the rep5 cost (2026-11-01)

Status: PROVISIONAL until T26 (row audit C25 / C37 on Euler). Table level from the gate jsons (tools/hero_tradeoff.py); rep5 = share of the K picks re-detected 5 frames on, paired on the same queries.

| gate | split | K | flat fail % (frames) cand vs inc | flat diff pt [95 %] | flights b/w/t | rel | verdict | rep5 flat diff [95 %] b/w | rep5 all diff [95 %] b/w |
|---|---|---|---|---|---|---|---|---|---|
| cube0.5 vs sd | fresh | 100 | 4.58 (95) vs 7.13 (148) of 2075 | -2.55 [-4.80,-0.34] | 5/3/5 p 0.36 | -35.8 % | KEEP | -0.1046 [-0.1288,-0.0810] 2/11 (n 2063) | -0.0334 [-0.0393,-0.0277] 0/14 (n 13377) |
| cube0.5 vs dpp0.5 | fresh | 400 | 2.75 (57) vs 3.08 (64) of 2075 | -0.34 [-1.00,+0.40] | 5/2/6 p 0.23 | -10.9 % | REVERT | -0.1063 [-0.1178,-0.0941] 0/13 (n 2063) | -0.0782 [-0.0819,-0.0742] 0/14 (n 13377) |
| dppr0.5a1b2t0.8 vs dpp0.5 | fresh | 100 | 2.89 (60) vs 2.99 (62) of 2075 | -0.10 [-0.71,+0.54] | 4/3/6 p 0.50 | -3.2 % | REVERT | +0.0034 [+0.0014,+0.0054] 11/2 (n 2063) | +0.0124 [+0.0110,+0.0138] 14/0 (n 13377) |
| dppr0.317a1.6b2.01t0.9 vs dpp0.5 | fresh | 100 | 3.23 (67) vs 2.99 (62) of 2075 | +0.24 [-0.31,+0.84] | 4/3/6 p 0.50 | +8.1 % | REVERT | +0.0363 [+0.0313,+0.0415] 13/0 (n 2063) | +0.0399 [+0.0380,+0.0420] 14/0 (n 13377) |

## Leaderboard (2026-10-23, 31 flights, K 100, lg): the repeatability column

| picker | fail % all | fail % flat | fail % very flat | rep5 |
|---|---|---|---|---|
| dpp0.5 | 1.38 | 10.6 | 16.1 | 0.5997 |
| cube0.5 | 1.71 | 12.5 | 18.3 | 0.5889 |
| grid | 1.78 | 13.6 | 20.1 | 0.5604 |
| sd | 2.38 | 16.3 | 24.1 | 0.6182 |

Read: every flat-fail winner in the table pays in rep5 and every rep5 winner pays in flat fail; the paper states the trade-off, not a fix.
