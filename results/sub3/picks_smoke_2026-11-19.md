# T33 picks-dump plumbing smoke, 5 arms K100 (SUB3, engineer 2026-11-19, dispatch 2026-11-19T13:15)

Command: `python tools/picks_smoke.py` (hospital/P041, 589 frames, LightGlue, laptop GPU, 5 min 52 s). Result: **PASS**.

- What it proves: `kpbench.run --save-picks --tag picks` writes rows that `gate.require_picks` accepts and `kpbench.pickarms` reads, for all 5 registered arms (dpp0.5, dppr0.317a1.6b2.01t0.9, cube0.5, grid, cube0.359).
- Per arm: meta has_picks True, 530 rows = 530 onplane entries, all 530 with 5 fields and 100 per-pick labels + 100 per-pick depths, 153 of 153 flat queries read, off / offn / dspan filled on 153 / 153 / 153; paired read-out vs dpp0.5 has n 153 per field for each of the 4 arms.
- What it does NOT prove: the laptop has the feature cache but no raw flight folder, so poses are placeholders (identity). Pose errors and fail flags are meaningless, and no off / offn / dspan / sep value was printed or stored (temp dir deleted). No number from this smoke is a result.
- Still owed: the same one-flight smoke on the Euler login node with real poses (VPN was down 13:10), then the DEV dump itself (5 arms, 12 flights, about 4 GPU-h, FREEZE: needs Kinga's yes). Held item: keypoint-gap-agent queue/_held/Qx-picksdump-dev.json.
