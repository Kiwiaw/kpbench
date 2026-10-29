# E4b deco picker: MICRO pick-level smoke (2026-10-29, K 100)

Source: tools/deco_smoke.py on cache/*_raco.npz, 393 MICRO flat frames (oldtown/P001, hospital/P041). No poses on the laptop, so no fail %;
fail % comes from the Euler DEV gate.

| check | number | pass |
|---|---|---|
| (a) deco b0 = sd | 0 / 393 frames differ | True |
| (b) deco b1 vs sd | 393 / 393 frames differ (100.0 %), mean per-frame std of r 0.210 | True |
| (c) runtime vs dppr0.317 | 0.39x (5.9 s vs 15.2 s) | True |

Mean 3D nearest-neighbour distance between picks (m, camera frame; higher = more spread):

| picker | nn dist m | pick s |
|---|---|---|
| sd | 0.217 | 0.0 |
| deco | 0.210 | 5.9 |
| decob0 | 0.217 | 0.0 |
| cube0.5 | 0.319 | 0.5 |
| dpp0.5 | 0.330 | 9.0 |
| dppr0.317a1.6b2.01t0.9 | 0.285 | 15.2 |

Verdict: PASS (a, b, c)
