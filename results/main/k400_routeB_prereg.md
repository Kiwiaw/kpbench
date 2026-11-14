# Pre-registration: K400 route B, cube0.5 vs dpp0.5 (written 2026-11-14, before any FRESH-2 row exists)

Why now: tools/k_interaction.py on FRESH-1 (results/main/k_interaction_fresh1.md) gives picker x K interaction
(cube0.5 - dpp0.5 at K400) - (same at K100) = -40 of 2075 flat = -1.93 pt, flight-cluster CI [-3.35, -0.66], 7/0/6 flights,
exact sign-flip p2 0.016. So the winner depends on K: dpp0.5 leads at K100 (62 vs 95), cube0.5 leads at K400 (57 vs 64).
Per the 14 Nov dispatch kill rule (CI below 0) this prereg is mandatory before any K400 claim.

Note: the 14 Nov dispatch quoted K100 as "67 vs 62"; 67 is dppr0.317 (K100 FRESH gate), not cube0.5. The cube0.5 K100 FRESH-1
count is 95 (E1 run). All six r28 counts reproduce (tools/fresh_counts.py), so this is a label slip, not a SUB2 fail.

## Test (one shot)
- Frames: FRESH-2 only (frames never read by any gate); not FRESH-1 (read 10x) and not HELD_OUT (one shot per picker, reserved).
- Arms: cand raco cube0.5 K400 same lg vs inc raco dpp0.5 K400 same lg, same job, same device type for both arms (C38 device note).
- Rule: kpbench/stats.py gate rule unchanged; the claim is carried by route B = flight sign test on flat fail %, tied flights
  dropped, one-sided binomial p <= 0.025 (needs >= 6/0, 8/1, 9/1, 11/2 ...), plus guards rel drop >= 10 % and all-frames <= +0.2 pt.
- Secondary (descriptive, no claim): the same interaction at K100 on FRESH-2; K200 not run.

## Reading, fixed now
- WIN at K400 -> wording "cube0.5 fails less than dpp0.5 at K400; dpp0.5 fails less at K100"; incumbent stays dpp0.5 (the benchmark
  budget is K100); paper scope gets a K paragraph (SUB4).
- No win -> "the K400 lead is not confirmed"; claims stay scoped to K <= 100.
- Either way the FRESH-1 interaction (-1.93 [-3.35, -0.66]) is reported, labelled "FRESH-1, read 10x, descriptive".

## Not queued
Freeze holds (decision.md default since 13 Nov 12:00) and FRESH-2 needs Kinga's go; this file only fixes the test.
