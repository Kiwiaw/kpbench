# C42: kpbench.pickstats on synthetic picks files with a known answer (16 Nov 2026, laptop, no flight data)

Verdict: **PASS** (3/3 file-level cases + 3/3 earlier unit cases, `python -m pytest tests/test_pickstats_file.py tests/test_pickstats.py`).

Setup (tests/test_pickstats_file.py): two fake runs written as real `--save-picks` files (meta.json + `<flight>.json.gz` with rows, picks, onplane),
2 flights x 10 queries; queries 0-5 flat (plane_frac 0.9), 6-9 not flat (0.5). Fail = error 300 cm (> 25 cm), pass = 5 cm.

| case | planted | expected | got |
|---|---|---|---|
| 1 known answer | cand fails q0, off 0.30, depths 10/20/30 m; inc fails q0-2, off 0.10, depths 10/10/10 m | n 20, flat 12, rescued 4 (q1, q2 x 2 flights); off diff +0.2 on 12/12 flat and 4/4 rescued; dstd 8.165 m, dspan 0.8; kill False | same |
| 2 kill | cand off 0.05 | rescued off median -0.05, 4 down; kill True | same |
| 3 no picks | cand meta has_picks False | GateError (refused) | refused |

What it covers: file reading, flat filter from labels, rescued set (cand posed, inc failed), paired median/mean/up/down, T33 kill flag.
Not covered: real label files and real `bench.onplane` output on flight data (unit test `test_onplane_share_matches_planted_split` covers onplane on a synthetic frame).
Consequence: pickstats outputs may now be written (SUB3 MICRO dry run, J6 regression inputs).
