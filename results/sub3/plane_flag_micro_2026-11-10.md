# Plane-fit flag from cached 3D detections vs the frame label (raco, 3980 labelled queries of 6 flights, MICRO 393, 2026-11-10 13:36)

Flag = RANSAC dominant-plane inlier share over the strongest 300 candidates with depth >= 0.85; label = depth-map plane_frac >= 0.85 (results/labels). Flights: hospital_P041, oldtown_P000, oldtown_P001, oldtown_P002, oldtown_P006, oldtown_P007.

| set | n | label flat | flag flat | both (TP) | label only (FN) | flag only (FP) | neither (TN) | agreement | flat recall | flat precision |
|---|---|---|---|---|---|---|---|---|---|---|
| all labelled queries | 3980 | 814 | 616 | 550 | 264 | 66 | 3100 | 0.917 | 0.676 | 0.893 |
| MICRO 393 (label-flat by construction) | 393 | 393 | 318 | 318 | 75 | 0 | 0 | 0.809 | 0.809 | 1.000 |

Agreement on all labelled queries by flag threshold (label fixed at 0.85): 0.60 -> 0.864, 0.70 -> 0.924, 0.75 -> 0.939, 0.80 -> 0.936, 0.85 -> 0.917, 0.90 -> 0.890, 0.95 -> 0.871
Spearman rank correlation of plane_frac_hat with the label plane_frac: 0.846; median |hat - label| 0.083.

## Pre-registered reading

- flag agreement on all labelled queries 0.917 (condition >= 0.9: met); on MICRO the flag recalls 0.809 of the 393 label-flat frames
- the detection-based flag is usable for a gated cube (cube0.5 when flagged, dpp0.5 otherwise); still a spec: the oracle composite guards (SUB2 r24_strata_fresh1.md) must clear first, then a gate on a gate-capable set
