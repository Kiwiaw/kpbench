# E-d: EPnPU-style pose covariance trace per pick set on MICRO (engineer 2026-11-12, laptop GPU)

## Pre-registered reading (written 2026-11-12 16:45, before any number)
- Pickers: sd, grid, cube0.5, dpp0.5 (+ dq3 for SUB3), K 100, MICRO flat queries, kpbench.evaluate protocol.
- Metric epnpu_t (new, differs from pcov v3 which used the estimated pose and inliers and was None on no-pose queries): at the TRUE query pose, take the matched points whose true-pose reprojection error < 3 px; per point 2D covariance S_i = I (1 px) + J_X S_X J_X^T with isotropic 3D std s_X^2 = (z/f)^2 + (0.005 z^2)^2 (1 px keyframe noise back-projected + depth noise growing with z^2; Vakhitov et al. CVPR 2021 style uncertainty propagation, simplified to isotropic); information = sum J_i^T S_i^-1 J_i; epnpu_t = log10 sqrt(trace of camera-centre block) in metres. < 6 true inliers or singular -> +inf (worst). Defined on every query, including failures.
- Readout: AUROC of epnpu_t for PnP failure on flat queries (pooled over the 4 pickers and per picker), and the median epnpu_t per picker.
- Kill (dispatch 16:25): pooled AUROC <= 0.5, or median order not sd > grid > cube0.5 > dpp0.5 -> SUB1 stays KILL for good, no executor run. Pass on MICRO only enables the one FRESH-1 K100 + K400 sub1 run (rule 1-3 unchanged).

## Status 2026-11-12 16:55: E-d NOT RUN (blocked)
- MICRO poses gone from the laptop (local_data removed by the 14-day auto-cleanup); the candidate cache alone cannot give PnP outcomes or the true pose. Euler unreachable (no VPN).
- Tool ready: tools/micro_epnpu_dq3.py (py_compile OK; E-d + E-c in one pass). Run on Euler (`KPBENCH_DATA` set) or after a MICRO re-copy (needs Kinga's OK). Pre-registered reading above unchanged; SUB1 stays KILL until it runs.
