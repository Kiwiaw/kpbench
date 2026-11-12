# r30: dpp2d120a1 vs dpp0.5 FRESH-1 K100 gate table recomputed from raw rows (engineer 2026-11-12, laptop, 0 GPU-h)

Source: results/runs/raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh and results/runs/raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh (local mirror, 14 flights, .json.gz read directly), results/labels/<flight>.json, rule re-implemented here (flat plane_frac >= 0.85; fail = no pose / > 25 cm / > 2 deg; 20-frame blocks). No kpbench.gate / stats code imported. Compared to results/gates/raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh__vs__raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh__fresh.json.

| number | recomputed | published | check |
|---|---|---|---|
| flat frames | 2075 | 2075 | PASS |
| queries | 13440 | 13440 | PASS |
| flat fails cand (dpp2d120a1) | 119 | 119 | PASS |
| flat fails inc (dpp0.5) | 62 | 62 | PASS |
| inc flat fail % | 2.99 | 2.99 | PASS |
| flat diff, block CI seed 0 | +2.75 [+1.08, +4.90] | +2.75 [+1.08, +4.90] | PASS |
| all diff, block CI seed 0 | +0.37 [+0.05, +0.76] | +0.37 [+0.05, +0.76] | PASS |
| flights better / worse / tied (cand) | 3 / 5 / 5 | 3 / 5 / 5 | PASS |
| sign p one-sided (cand lower) | 0.8555 | 0.8555 | PASS |
| flight-cluster CI 2000 x seed 22 | +2.75 [+0.05, +5.58] | +2.75 [+0.05, +5.58] (10 Nov note) | PASS |
| rep5 flat n | 2063 | 2063 | PASS |
| rep5 flat diff, block CI seed 0 | -0.0833 [-0.0977, -0.0701] | -0.0833 [-0.0977, -0.0701] | PASS |

Seed check: flat diff block CI with seed 1 = +2.75 [+1.00, +4.98] (seed 0 is the gate default).
Orientation: gate line 3 / 5 / 5 = candidate (pixel dpp2d) better / worse / tied; "5 / 3 / 5 for 3D" in the 10 Nov note is the same split.

**Verdict: PASS** (12 / 12 numbers). The [P] tag on +2.75 [+1.08, +4.90] can be lifted at table+row level.
