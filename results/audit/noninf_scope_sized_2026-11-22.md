# T48 pass chances at the registered sizes (SUB2, tools/noninf_scope_sized.py)

Rows: 13 FRESH-1 flights, 2075 flat, fails cand 67 vs inc 62 (dppr0.317a1.6b2.01t0.9 vs dpp0.5, K100). Pass = flight-bootstrap upper bound < +1.0 pt. Outer 4000 sets, inner 1000 draws, seed 7 per column.

| registered flat | over (T49 rule) chance / realised flat | T49 printed | exact chance / median upper / realised | under chance / realised flat | exact - T49 |
|---|---|---|---|---|---|
| 2774 | 61.4 % / 2881 | 61.4 % | **62.0 %** / +0.88 / 2774 | 58.0 % / 2668 | +0.6 |
| 3500 | 69.2 % / 3607 | 70.8 % | **70.3 %** / +0.82 / 3500 | 67.5 % / 3393 | -0.5 |
| 4849 | 82.1 % / 4955 | 82.3 % | **81.0 %** / +0.73 / 4849 | 80.9 % / 4743 | -1.3 |

Monte Carlo error of one chance: about +-1.5 points (95 %, 4000 sets).

Reading (engineer 2026-11-22, dispatch 15:25): at exactly the registered sizes the pass chance is 62.0 / 70.3 / 81.0 % (T49 printed 61.4 / 70.8 / 82.3 %); the differences (+0.6 / -0.5 / -1.3 points) are inside the Monte Carlo error of +-1.5 points, so the about 105-query overshoot did not inflate the printed chances by a measurable amount. Even with the last flight dropped (smaller sets, 2668 flat) the chance at the first size is 58.0 %. Dispatch kill (size-matched chance at 2774 under 50 %) NOT met: the SUB3 non-inferiority claim stays open with about 62 % at 2774 flat.
The 'over' column repeats the T49 rule with another random stream: 61.4 / 69.2 / 82.1 vs 61.4 / 70.8 / 82.3, i.e. the seed-to-seed spread is up to 1.6 points. Print the chances as 'about 62 / 70 / 81 %, +-1.5'. Same method as T48 / T49 (nested flight resampling of the 13 FRESH-1 flights): this fixes the size, it is not a new method.
