"""SUB1 candidate per-query metrics (pre-registered in docs/sub1_metrics.md before any number was computed).

Raw values from one PnP result; None when there is no pose. The no-pose values of the pre-registration (caps) are applied at
analysis time, not here. n_inl is already a bench column, so it is not repeated.
  crlb        log10 camera-centre std in cm at 1 px iid noise over the inliers: Sigma = (J^T J)^-1, J = 2n x 6 Jacobian of the
              pixel reprojection w.r.t. (small rotation, camera centre) at the estimated pose
  inl_ratio   n_inl / n_match
  hull        convex-hull area of the inlier pixels / image area
  reproj_med  median inlier reprojection error at the refined pose, px
  log_t       log10(max(position error cm, 0.1))
  log_r       log10(max(rotation error deg, 0.01))
  thr_score   mean over tau in (1, 2, 5, 10, 25) cm of [t_err <= tau and r_err <= tau / 12.5 deg]
  depth_iqr   interquartile range of the inliers' camera depth, m (for the depth-spread check, boring explanation B)
"""
import numpy as np
from scipy.spatial import ConvexHull
from .data import FX, FY, CX, CY, W, H

MCOLS = ['crlb', 'inl_ratio', 'hull', 'reproj_med', 'log_t', 'log_r', 'thr_score', 'depth_iqr']
TAUS = (1, 2, 5, 10, 25)


def crlb_cm(Xc, R):
    """camera-centre std in cm at 1 px noise. Xc = inlier points in the camera frame (n x 3); parametrisation
    Xc = exp(w) R (X - c): dXc/dw = -[Xc]x, dXc/dc = -R"""
    x, y, z = Xc.T
    dpu = np.stack([FX / z, 0 * z, -FX * x / z ** 2], 1); dpv = np.stack([0 * z, FY / z, -FY * y / z ** 2], 1)
    J = []
    for dp in (dpu, dpv):
        dw = np.cross(Xc, dp)                         # dp . (-[Xc]x dw) = (Xc x dp) . dw
        dc = -dp @ R
        J.append(np.concatenate([dw, dc], 1))
    J = np.concatenate(J)
    try:
        S = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    tr = np.trace(S[3:, 3:])
    return float(100 * np.sqrt(tr)) if tr > 0 else None


def query_metrics(obj, img, n_match, err, rot, sol):
    """obj (n x 3 world), img (n x 2) = the PnP input; sol = (R, tvec, inlier indices) from bench.pnp(full=True) or None"""
    if sol is None or err is None:
        return [None] * len(MCOLS)
    R, t, inl = sol
    X = np.asarray(obj, float).reshape(-1, 3)[inl]; uv = np.asarray(img, float).reshape(-1, 2)[inl]
    Xc = X @ R.T + t
    c = crlb_cm(Xc, R)
    proj = np.stack([FX * Xc[:, 0] / Xc[:, 2] + CX, FY * Xc[:, 1] / Xc[:, 2] + CY], 1)
    try:
        hull = ConvexHull(uv).volume / (W * H)
    except Exception:
        hull = 0.0
    q75, q25 = np.percentile(Xc[:, 2], [75, 25])
    thr = np.mean([err <= tau and rot <= tau / 12.5 for tau in TAUS])
    out = [np.log10(c) if c else None, len(inl) / max(n_match, 1), hull, float(np.median(np.linalg.norm(proj - uv, axis=1))),
           np.log10(max(err, 0.1)), np.log10(max(rot, 0.01)), thr, q75 - q25]
    return [None if v is None else round(float(v), 4) for v in out]
