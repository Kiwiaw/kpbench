1328 query frames (every 3th query of 6 flights), fastest of 3 repeats, ms per frame: median [10th, 90th percentile]. Machine: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 20 logical cores, numpy 2.4.2, Python 3.14.2. Tag [P] (laptop).

| K | dpp0.5 benchmark path | dpplazy0.5 | cube0.5 | benchmark / lazy | lazy / cube0.5 |
|---|---|---|---|---|---|
| 50 | not timed | 1.30 [0.88, 2.05] | 1.82 [1.00, 2.19] | - | 0.7x |
| 100 | 40.55 [25.12, 59.13] | 3.96 [2.27, 6.45] | 1.83 [1.00, 2.21] | 10.3x | 2.2x |
| 200 | not timed | 13.30 [7.73, 23.06] | 1.84 [1.01, 2.31] | - | 7.2x |
| 400 | not timed | 46.65 [23.27, 98.82] | 1.86 [1.02, 2.36] | - | 25.1x |
