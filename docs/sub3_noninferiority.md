# SUB3 claim as non-inferiority (pre-registration, 2026-10-24)

SUB3 claims "more repeatable keypoints at no pose cost". The MAIN gate (>= 10 % relative drop, 1.44 pt at 14.42) tests
something else: that the picker is *better* for pose. This sheet fixes the test for "no pose cost" before any new run.

## Rule (fixed now)

- Margin: **+0.50 pt** absolute flat fail. Toy: dpp0.5 fails on 14.4 of 100 flat frames; the candidate may fail on at most
  14.9 of 100 and still count as "no cost".
- Why 0.50: keep at least half of the incumbent's own effect, taken at its conservative end. dpp0.5 vs cube0.5 on held-out
  flat is -2.74 [-4.90, -0.99]; lower bound 0.99 pt x 0.5 = 0.50 pt. Only that reference interval is used, not the DEV rows
  below. (0.69 = 1/4 of the point estimate was rejected: no convention, ignores the reference uncertainty.)
- Non-inferior iff the upper 95 % bound of (candidate - dpp0.5) flat fail, paired flight-block bootstrap, is **< +0.50 pt**
  AND the paired rep5 CI lower bound is **> 0**.
- Confirmatory split: **held-out flat (877 frames)**, one shot. DEV is the selection split (21 dppr/dpp tunes) and cannot
  confirm. FRESH-1 (2075 flat, base ~3 %, ~60 fails) is too low-base to see a cost; report it as a descriptive replicate only.
- One picker, named now: **dppr0.317a1.6b2.01t0.9** (best rep5, 12/0 flights, tightest DEV flat interval).
- Power stated in advance: the held-out dpp0.5-vs-cube0.5 interval had half-width ~2 pt, so at 877 frames the rule passes
  only if the candidate is truly ~1.5 pt better. A fail reads "non-inferiority not shown", never "inferior", and is not
  reworded afterwards.

## DEV rows (exploratory: selection split, margin chosen after these intervals were seen)

936 DEV flat frames, 12 flights; diff = candidate - dpp0.5 (14.42 %).

| picker | rep5 vs 0.4519 | flat diff pt [95 %] | upper vs +0.50 |
|---|---|---|---|
| dppr0.317a1.6b2.01t0.9 | 0.4756, +0.0236 [+0.0209, +0.0263], 12/0 | -0.75 [-1.86, +0.31] | would clear |
| dppr0.388a1.91b2.85t0.9 | 0.4749, +0.0229 [+0.0200, +0.0259], 11/1 | -0.11 [-1.38, +1.07] | would not |
| dppr0.329a0.701b2.02t0.9 | not tabled | -0.75 [-2.43, +0.81] | would not |

No non-inferiority verdict is drawn from this table; it is why row 1 is the pre-registered held-out candidate.

## R6: does the repeatability term earn a place in the paper (pre-registered, scout ed. 11)

Q211 (beta = 0 control dpp0.317a1.6, same alpha, no track term) decides it on the 12 DEV flights:
- rep5(beta = 0) < 0.4519 + 0.0209 = 0.4728 -> the track term causes the gain: the rep5 variant goes in the paper.
- else (beta = 0 keeps the gain) -> the gain comes from alpha alone: the variant is cut to an appendix note.

## Not decided here

Running the held-out shot goes against the program rule "no held-out gate without a DEV KEEP" (MAIN gate), so it needs
Kinga's OK as a separate SUB3 test before it is queued. Until then SUB3 reads "rep5 up, pose cost not shown".
Design: Fable, 2026-10-24.
