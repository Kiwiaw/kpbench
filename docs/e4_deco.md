# E4b: deco picker, a ranking-only control (MAIN/SUB4, 29 Oct 2026)

Why: the closest prior work (DECO, arXiv 2608.22289) ranks UAV keypoints by saliency x co-visibility before PnP, with no
spatial spread. The leaderboard has no ranker row at the same K (K100 flat fail: dpp0.5 10.6, cube0.5 12.5, grid 13.6, sd 16.3 %).

## Picker
- `deco[b<beta>][t<tau>]`, default b1 t0.9: q_i = (s_i / s_max) * c_i^beta over the candidates with depth, top K by q,
  fill by rank (depth only). No spatial term.
- c_i = co-visibility estimate = dppr's track count: (m_i + 1) / (M + 1), m_i = past frames q-1..q-3 where candidate i has a
  mutual nearest-neighbour descriptor match with cosine >= tau. Toy numbers: seen in 2 of 3 past frames -> c = 3/4.
- Why not GT-pose co-visibility (frustum + depth test into the map keyframes): it needs the query's true pose, i.e. an oracle
  (needs_ctx, refused by kpbench.evaluate). The past-frame estimate uses no pose and no future frame, like dppr.
- beta = 0 is sd (checked: 0 of 393 MICRO frames differ).

## Smoke (laptop, picks only: the laptop has no TartanAir poses, results/main/deco_smoke.md)
- PASS: b0 = sd on 393/393 frames; b1 changes the pick set on 393/393 frames, per-frame std of c 0.210; runtime 0.39x dppr0.317.
- Pick spread (mean 3D nearest-neighbour distance): sd 0.217 m, deco 0.210 m, cube0.5 0.319 m, dpp0.5 0.330 m: deco stays
  in the ranking class, as intended.

## Euler run and kill (pre-registered before any fail % exists)
- DEV gate deco vs dpp0.5, K100, lg (queue Qg-deco-dev). The 31-flight leaderboard row waits for the held_out frames on Euler.
- deco DEV flat fail <= cube0.5 DEV flat fail -> 'spread class > ranking' headline dies; between grid and sd -> confirmed.
