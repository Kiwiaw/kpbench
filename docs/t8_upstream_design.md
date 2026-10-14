# T8: upstream-of-picker tune (MAIN), design 14 Oct 2026

Why: three 30-trial picker tunes (dpp_dev2, pd_dev1, cube_dev1) bottom at 13.66-13.87 % DEV flat fail; the MAIN gate needs
<= 12.97 % (10 % below the incumbent dpp0.5, 14.41 % on 937 DEV flat frames). The next lever is the candidate set the picker sees.

1. Which knobs. The cache holds the strongest 2000 RaCo candidates per frame in score order, already NMS'd at extraction.
   - Cache-only, applied in this order: border margin m px (drop u < m, u > W - m, v < m, v > H - m); extra greedy NMS radius r px
     in score order (only r above RaCo's native radius changes anything); top-N pool (first N after border + NMS). A score
     threshold is the same knob as N on a score-ordered cache, so it is dropped.
   - Need re-extraction: pyramid scale, sub-pixel refinement. Deferred until the cache-only tune has a verdict.
   - Downstream fixed: dpp0.5, K = 100.
2. Search space (3 dims, 30 trials): m in {0, 4, 8, 12, 16, 24, 32}; r int 0-12 (0 = off); N log int 150-2000.
   If the filter leaves fewer than K usable points, the picker pads from the suppressed points in score order (depth first),
   never returns fewer than K when the frame has K.
3. Grammar: up[b<m>][r<r>]+dpp<sigma>[a<alpha>][p<N>], omitted token = off; upb0r0+dpp0.5p2000 == dpp0.5.
4. Seeds: identity, r6, b16, p500, b8 r6 p800, r4 (near-native radius, no-op floor check).
5. Kill: no trial <= 13.66 % by trial 15 -> the lever is not in the cache. Ticket kill: best trial not >= 2 pt below dpp0.5 DEV
   flat -> MAIN stays dpp0.5. Smoke sanity: identity reproduces dpp0.5's picks exactly; r = 12 cuts the mean candidate count
   >= 1.5x with almost no padded frames.
6. More DEV power without touching HELD_OUT: pair each flat query with more keyframes and add a near-flat stratum (relaxed
   planarity threshold) from the same 12 flights, scored as paired per-frame discordance vs dpp0.5; non-flat frames only as a
   non-regression guard. 937 frames give SE ~1.1 pt, so the 1.44 pt gap is ~1.3 SE unpaired; pairing buys the power.
   Not implemented in step 1 (a later T8 step).

Design by one Fable consult (engineer session 14 Oct); step 1 implements 1-5.
