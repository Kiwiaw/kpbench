# Selection time per frame, hospital/P041 (raco candidates, laptop CPU) [P]

tools/select_time.py: picker call only (no detector, matcher or pose), 530 queries, fastest of 3 repeats per query, pickers and K interleaved per query. Candidates per query: median 2000 (min 2000, max 2000), with depth median 2000. Machine: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 20 logical cores, numpy 2.4.2, Python 3.14.2.

| K | dpp0.5 median ms | dpp0.5 p90 ms | cube0.5 median ms | cube0.5 p90 ms | ratio of medians dpp0.5 / cube0.5 |
|---|---|---|---|---|---|
| 50 | 5.46 | 7.17 | 1.75 | 2.07 | 3.1x |
| 100 | 34.54 | 39.25 | 1.75 | 1.89 | 19.7x |
| 200 | 176.22 | 197.53 | 1.77 | 2.10 | 99.6x |
| 400 | 464.80 | 845.59 | 1.80 | 2.51 | 257.8x |

Reading (engineer 2026-11-22, dispatch 15:25): at K100 the dpp0.5 selection costs about 35 ms per frame against about 1.8 ms for cube0.5 (20x); cube time does not grow with K, dpp0.5 grows faster than K (5.5 -> 35 -> 176 -> 465 ms, the greedy update is O(N K) per frame plus Python overhead; N = 2000 candidates). Toy: at 10 frames/s a frame has 100 ms, so dpp0.5 fits at K100 (35 ms) and not at K200 (176 ms) in this implementation.
Limits: one flight (530 queries), one laptop CPU, pure numpy implementation of both pickers (no GPU, no compiled code), every frame has the full 2000 candidates with depth; the K400 90th percentile (846 ms) shows machine noise. Not a fail number, no pose read. Tag [P].
