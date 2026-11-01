# SUB3 r-term pick change on MICRO (2026-11-01, K 100)

Source: tools/rterm_pickchange.py on cache/*_raco.npz, 393 MICRO flat frames (oldtown/P001, hospital/P041). No poses on the laptop: pick change only, no fail % or rep5.
change = 1 - |picks(dppr) & picks(dpp)| / K per frame, dppr vs its sigma-matched plain DPP. Bootstrap CI over frames (frames of one flight are not independent, so read the per-flight means too).

| r-term picker | plain DPP | mean change | 95 % CI | median | frames changed | per flight | r std | beta 0 = dpp (frames differing) | s dppr / dpp |
|---|---|---|---|---|---|---|---|---|---|
| dppr0.5a1b2t0.8 | dpp0.5 | 0.253 | [0.243, 0.263] | 0.260 | 392 / 393 | P001 0.225, P041 0.297 | 0.255 | 0 | 30.3 / 18.1 |
| dppr0.317a1.6b2.01t0.9 | dpp0.317a1.6 | 0.205 | [0.199, 0.212] | 0.210 | 392 / 393 | P001 0.197, P041 0.218 | 0.210 | 0 | 33.6 / 18.4 |

## Pre-registered test (dispatch 2026-11-01 15:20)

- ratio mean change sigma 0.317 / sigma 0.5 = 0.205 / 0.253 = 0.81 (prediction >= 2.0): NOT MET
- paired per-frame difference (0.317 - 0.5) = -0.048 [-0.054, -0.041]
- kill (change at 0.5 >= change at 0.317): MET -> sigma-saturation hypothesis dead; DEV study sweeps beta/t at fixed sigma 0.5
- sanity beta 0 = plain DPP on every frame: PASS

Read: the FRESH rep5 gains are +0.0034 (sigma 0.5) and +0.0113 (sigma 0.317); if the pick change ratio is far from the gain ratio 3.3, the r-term moves picks that the
repeatability score does not reward, i.e. the gain is bounded by what the kernel scale leaves free, not by how many picks the r-term can touch.
