# Inlier-matched control on dev flat (936 queries): dpp0.5 vs sd

raw flat fail 14.21 vs 22.33 % (diff -8.12 pt); median inliers 51 vs 51
inlier-matched diff (deciles of own inlier count, weighted by sd): -6.13 pt [-10.67, -1.82], 0.75 of the raw gap
logistic fail ~ log(1+inliers) + picker: log-odds dpp0.5 vs sd -0.542 [-0.926, -0.150]

| inlier bin | upper edge | n dpp0.5 | fail % dpp0.5 | n sd | fail % sd |
|---|---|---|---|---|---|
| 1 | 12 | 69 | 71.0 | 107 | 76.6 |
| 2 | 24 | 106 | 31.1 | 88 | 22.7 |
| 3 | 34 | 93 | 21.5 | 92 | 23.9 |
| 4 | 42 | 89 | 14.6 | 91 | 29.7 |
| 5 | 51 | 108 | 13.0 | 89 | 31.5 |
| 6 | 60 | 99 | 3.0 | 90 | 18.9 |
| 7 | 68 | 93 | 1.1 | 88 | 4.5 |
| 8 | 76 | 99 | 0.0 | 89 | 6.7 |
| 9 | 84 | 100 | 0.0 | 92 | 2.2 |
| 10 | max | 80 | 0.0 | 110 | 0.9 |

**dpp0.5 fails less than sd at equal inlier count**

Reference arm, same script (`--cand cube0.5`): raw -7.16 pt, inlier-matched -9.58 pt [-14.67, -5.34], log-odds -1.105 [-1.489, -0.774] (cube0.5 median 41 inliers vs sd 51).

Notes (engineer 2026-10-19, dispatch 13:10 MAIN): rows are the 18 Oct SUB1 eval files (sd and dpp0.5 on one node, 936 flat); dpp0.5 flat fail here 14.21 % vs 14.42 % in the gate runs (different run of the same picker; 2 queries). Kill (picker effect covers 0 at equal inliers) not met: the uniformity gain is not explained by inlier count; dpp0.5 and sd have the same median inlier count (51).
