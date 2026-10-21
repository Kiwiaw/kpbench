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
  cells       share of the 8 x 6 image cells (80 px) holding >= 1 inlier (prereg v2, results/sub1/prereg.md)
  ent         entropy of the inlier counts over those 48 cells / log 48 (0 = one cell, 1 = even spread)
  offplane    RMS distance of the inliers (camera frame) to their total-least-squares plane / their median depth
  sel_logdet  pre-solve (scout E3, dispatch 2026-10-20 14:45): log10 det of the 2 x 2 covariance of the SELECTED query keypoints
              in image-normalised coordinates (u / W, v / H); needs no pose, so it is filled for no-pose queries too
  sel_hull    pre-solve: convex-hull area of the selected query keypoints / image area (hull is the same on the inliers)
  pcov        prereg v3 (dispatch 2026-10-21 13:10, T10): log10 sqrt(trace) of the full 6 x 6 pose covariance at 1 px noise,
              scale-free: rotation in rad, camera centre divided by the inliers' median depth (so a 2x bigger scene gives the
              same value); crlb is its centre-only part in cm
"""
import numpy as np
from scipy.spatial import ConvexHull
from .data import FX, FY, CX, CY, W, H

MCOLS = ['crlb', 'inl_ratio', 'hull', 'reproj_med', 'log_t', 'log_r', 'thr_score', 'depth_iqr', 'cells', 'ent', 'offplane', 'sel_logdet', 'sel_hull', 'pcov']
TAUS = (1, 2, 5, 10, 25)


def pose_cov(Xc, R):
    """6 x 6 pose covariance (rotation rad, centre m) at 1 px iid noise, None if singular. Xc = inlier points in the camera
    frame (n x 3); parametrisation Xc = exp(w) R (X - c): dXc/dw = -[Xc]x, dXc/dc = -R"""
    x, y, z = Xc.T
    dpu = np.stack([FX / z, 0 * z, -FX * x / z ** 2], 1); dpv = np.stack([0 * z, FY / z, -FY * y / z ** 2], 1)
    J = []
    for dp in (dpu, dpv):
        dw = np.cross(Xc, dp)                         # dp . (-[Xc]x dw) = (Xc x dp) . dw
        dc = -dp @ R
        J.append(np.concatenate([dw, dc], 1))
    J = np.concatenate(J)
    try:
        return np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None


def crlb_cm(Xc, R):
    """camera-centre std in cm at 1 px noise"""
    S = pose_cov(Xc, R)
    tr = np.trace(S[3:, 3:]) if S is not None else 0
    return float(100 * np.sqrt(tr)) if tr > 0 else None


def pcov(Xc, R):
    """log10 sqrt(trace) of the pose covariance with the centre in units of the median inlier depth (scale-free)"""
    S = pose_cov(Xc, R)
    if S is None:
        return None
    zm2 = max(float(np.median(Xc[:, 2])), 1e-6) ** 2
    tr = np.trace(S[:3, :3]) + np.trace(S[3:, 3:]) / zm2
    return round(float(0.5 * np.log10(tr)), 4) if tr > 0 else None


def presolve(sel_uv):
    """[sel_logdet, sel_hull] of the selected query pixels (k x 2); None / 0 below 3 points or for a degenerate set"""
    P = np.asarray(sel_uv, float).reshape(-1, 2) / (W, H)
    if len(P) < 3:
        return [None, 0.0]
    dt = np.linalg.det(np.cov(P.T))
    try:
        hl = ConvexHull(P).volume
    except Exception:
        hl = 0.0
    return [round(float(np.log10(dt)), 4) if dt > 0 else None, round(float(hl), 4)]


def query_metrics(obj, img, n_match, err, rot, sol, sel_uv=None):
    """obj (n x 3 world), img (n x 2) = the PnP input; sol = (R, tvec, inlier indices) from bench.pnp(full=True) or None;
    sel_uv (k x 2) = the picked query keypoints (pre-solve columns, None -> None)"""
    pre = presolve(sel_uv) if sel_uv is not None else [None, None]
    if sol is None or err is None:
        return [None] * (len(MCOLS) - 3) + pre + [None]
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
    cnt = np.bincount((np.clip(uv[:, 1] // 80, 0, 5) * 8 + np.clip(uv[:, 0] // 80, 0, 7)).astype(int), minlength=48)
    pr = cnt[cnt > 0] / cnt.sum(); D = Xc - Xc.mean(0)
    off = float(np.sqrt(np.linalg.eigvalsh(D.T @ D / len(Xc))[0]) / max(np.median(Xc[:, 2]), 1e-6)) if len(Xc) >= 4 else None
    out = [np.log10(c) if c else None, len(inl) / max(n_match, 1), hull, float(np.median(np.linalg.norm(proj - uv, axis=1))),
           np.log10(max(err, 0.1)), np.log10(max(rot, 0.01)), thr, q75 - q25, (cnt > 0).mean(), float(-(pr * np.log(pr)).sum() / np.log(48)), off]
    return [None if v is None else round(float(v), 4) for v in out] + pre + [pcov(Xc, R)]
