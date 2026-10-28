# Determinism probe, K200, MICRO cache (28 Oct 2026, SUB2)

Question: dpp1a2 fails 165 vs 167 of 936 DEV flat frames in dpp_K200_dev1 vs dpp_K200_dev2 (same picker, same K). Picker
nondeterminism or something downstream?

Run: `python tools/determinism_probe.py dpp1a2 dpp0.601a0.283 dpp0.5 --K 200` (kpbench HEAD 629610f8 + this tool), 393 MICRO
frames of cached raco candidates, three fresh processes: A, B default BLAS threads, C 1 thread.

| picker | frames | differ A vs B | differ A vs C (1 thread) |
|---|---|---|---|
| dpp1a2 | 393 | 0 | 0 |
| dpp0.601a0.283 | 393 | 0 | 0 |
| dpp0.5 | 393 | 0 | 0 |

Raw: results/audit/determinism_probe_K200.json.

Reading: the DPP pickers are deterministic at K200 (also across BLAS thread counts). The 2-frame drift is not in the picks.
Left as causes: (1) pose side (LightGlue on a different GPU / RANSAC) -- not testable on the laptop (no MICRO flight
images), (2) commit drift between the two studies (629610f8 changed DPP fill at K200: depthless top-ups, dpp2d mostly,
3D dpp unaffected on MICRO). Pre-registered kill of the dispatch step: picks identical -> the 0.21 pt "seed floor" is not
a picker floor; per-flight localisation of the 2 frames (analyst) is needed before it is used as a gate margin.
