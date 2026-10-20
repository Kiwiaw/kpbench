# SUB3 objective fcap: min DEV flat fail s.t. rep5 >= 0.4527 (T4, engineer 2026-10-20)

Study name with `_fcap` (kpbench/tune.py): value = fail_flat if rep5 >= REP5_FLOOR (env KPBENCH_REP5_FLOOR, default 0.4527 =
dpp0.5 DEV flat rep5), else fail_flat + 100 + 100 x (floor - rep5). Summaries now flag the Pareto front (lower fail, higher rep5).
Laptop has no flight data, so the check re-ranks stored DEV trials (results/tune/*.json, no new runs):

| study | trials | feasible (rep5 >= 0.4527) | plain best fail / rep5 | fcap best fail / rep5 | Pareto front |
|---|---|---|---|---|---|
| dppr_rep5c_dev1 | 24 | 12 | dppr0.609a0.731b2.91t0.9 13.66 / 0.4464 | dppr0.367a0.744b2.34t0.9 13.77 / 0.4600 | 4 |
| dppr_dev1 | 12 | 7 | dppr0.486a0.453b1.02t0.9 13.23 / 0.4390 | dppr0.5a1b2t0.8 14.30 / 0.4543 | 7 |
| dpp_rep5c_dev1 | 24 | 14 | dpp0.545a0.395 13.34 / 0.4196 | dpp0.368a1.09 14.41 / 0.4580 | 7 |
| win_dev1 | 30 | 0 | win23+dpp0.666a0.526 13.23 / 0.3702 | win11+dpp0.436a1.29 14.30 / 0.4454 (infeasible) | 5 |

Regression: fcap on dppr_rep5c_dev1 picks the stored SUB3 best 13.77 / 0.460. The floor binds in all 4 studies (ranking differs
from plain fail_flat), so the kill "constraint never binds" is not met. Reading: in every stored family the lowest-fail trial
breaks the rep5 floor; under the floor the best fail is 13.77 (dppr), above the 12.97 MAIN bar.
Next (Euler): {"kind": "tune", "study": "dppr_fcap_dev1", "trials": 30}.
