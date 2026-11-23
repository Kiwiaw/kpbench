# T33 picks-dump contract smoke, 3 arms K100 (SUB3 + SUB1, engineer 2026-11-23, dispatch 2026-11-23T13:15)

Command: `python tools/picks_smoke.py --arms dpp0.5,cube0.5,dppr0.317a1.6b2.01t0.9` (hospital/P041, 589 frames, 530 queries, 153 flat, LightGlue, laptop GPU, placeholder poses). Result: **PASS**. Format check only: no off / offn / dspan / sep value and no fail number was printed or stored.

| arm | has_picks | queries with K = 100 picks = n_sel = 100 depth entries | dumped pixels = direct picker call on the cache | off / offn / dspan finite (flat queries) | file size per query |
|---|---|---|---|---|---|
| dpp0.5 | True | 530 of 530 | 530 of 530 | 153 of 153 | 0.71 kB |
| cube0.5 | True | 530 of 530 | 530 of 530 | 153 of 153 | 0.70 kB |
| dppr0.317a1.6b2.01t0.9 | True | 530 of 530 | not checked (picker reads past frames) | 153 of 153 | 0.70 kB |

- New in tools/picks_smoke.py (`dump_checks`, `--arms`): per query the dump holds exactly n_sel picks and as many per-pick depths; for pickers without frame context the dumped pixel list equals a fresh picker call on the cached candidates (so the dump is the picker's output, not a re-ordered or truncated copy); the three SUB1 / SUB3 fields are finite on every flat query.
- SUB1 kill ("dump lacks per-pick depth or plane id") is NOT met: per-pick on-plane label and depth are present for every pick, so the sep read is computable from picks as designed.
- Dispatch asked "picks reproduce the stored rep5 for that flight". Not possible on the laptop: rep5 needs the flight's poses and depth maps, which only exist on Euler (poses here are identity placeholders). Replaced by the pixel-identity check above; the rep5 check stays owed for the Euler one-flight smoke.
- Size for the full dump: 0.70 kB per query and arm -> the 8273 DEV queries (12 flights, 936 flat) take about 5.8 MB per arm, about 29 MB for the 5 registered arms. Toy: 530 queries x 0.70 kB = 0.37 MB for this flight.
- Still owed (VPN off today): the same smoke on the Euler login node with real poses, then the DEV dump (about 4 GPU-h; FREEZE, needs Kinga's yes; held item keypoint-gap-agent queue/_held/Qx-picksdump-dev.json).
