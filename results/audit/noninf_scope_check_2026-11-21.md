# T48 non-inferiority projection, second method (SUB2, tools/noninf_scope_check.py)

Rows: 13 flights, 2075 flat, fails cand 67 vs inc 62, diff +0.24 pt (T48: 13 flights, 2075, 67 vs 62, +0.24).
FRESH-1 flight-cluster CI (20000 weight draws, seed 7): [-0.39, +1.05] vs T48 [-0.39, +1.05] -> agree. Linearised flight SE 0.382 pt.

| flat | A pass chance | T48 | diff | A median upper | T48 | diff | B upper (blocks) | T48 scaled | diff | C normal approx (not in rule) | flights per set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2774 | 61.4 % | 60 % | +1.4 | +0.88 | +0.88 | -0.00 | +0.76 | +0.76 | -0.00 | 63.2 % | 18.0 |
| 3500 | 70.8 % | 71 % | -0.2 | +0.81 | +0.80 | +0.01 | +0.71 | +0.70 | +0.01 | 73.3 % | 22.6 |
| 4849 | 82.3 % | 84 % | -1.7 | +0.72 | +0.72 | -0.00 | +0.64 | +0.63 | +0.01 | 86.0 % | 31.0 |

Agreement rule: pass chance within 5 points, bounds within 0.1 pt. Worst gaps: chance 1.7 points, median upper 0.01 pt, block upper 0.01 pt.
**PASS -> T48 numbers [T]** (outer 4000, inner 1000, seed 7)
