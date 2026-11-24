| flight | queries | A: differ, sequential vs fresh instance | B: differ, forward vs reverse order | C: differ, lazy-sorted vs benchmark |
|---|---|---|---|---|
| hospital/P041 | 530 | 0 | 0 | 0 |
| oldtown/P000 | 711 | 0 | 0 | 0 |
| oldtown/P001 | 794 | 0 | 0 | 0 |
| oldtown/P002 | 835 | 0 | 0 | 0 |
| oldtown/P006 | 453 | 0 | 0 | 0 |
| oldtown/P007 | 657 | 0 | 0 | 0 |
| total | 3980 | 0 | 0 | 0 |

1328 timed query frames (every 3th), K100, fastest of 3 repeats, ms per frame median [10th, 90th percentile]; Machine: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 20 logical cores, numpy 2.4.2. Tag [P] (laptop).

| picker | ms per frame | / cube0.5 |
|---|---|---|
| dppr benchmark | 64.05 [32.03, 83.44] | 22.3x |
| dppr lazy | 25.24 [12.64, 31.25] | 8.8x |
| cube0.5 | 2.87 [1.01, 3.15] | 1.0x |

## Read (SUB3, engineer 2026-11-24, dispatch 2026-11-24T13:15 SUB3; tools/t33_dppr_replay.py) [P]
- Replay identity: **0 of 3980** query frames differ in A (sequential vs fresh instance), B (forward vs reverse order) and C (lazy prefix on quality-sorted candidates vs benchmark dppr). The past-frame cache holds no hidden state; the dumped dppr picks are the picker's output. Not checked: against stored Euler picks (none exist on the laptop: the FRESH dppr run has has_picks False and other flights) -> owed with the DEV dump.
- Time K100: dppr benchmark 64.05 ms, dppr lazy 25.24 ms (2.5x faster), cube0.5 2.87 ms in this run. Lazy cuts only the greedy; the rest (~20 ms) is track_count (descriptor mutual-NN over 3 past frames x 1000 candidates).
- Kill ("dppr0.317 > 10x cube0.5 1.1 ms at K100"): **met** on the registered threshold (25.24 ms > 11 ms, 23x; also 13.8x of the 1.83 ms cube0.5 of the T55 run). Only the same-run ratio is below 10x (8.8x), because cube0.5 ran slow here (2.87 vs 1.83 ms). Conservative reading: dppr0.317 is NOT offered as the practical variant.
