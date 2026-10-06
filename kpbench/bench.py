"""The task: relocalise every non-keyframe frame against a local map built from ground-truth depth and poses.

Protocol (the C23 protocol, unchanged):
  keyframes      every 10th frame; a query uses the keyframes within +-20 frames (usually 4)
  map            'same': each keyframe contributes the SAME picker's K points (with depth), lifted to 3D with its true pose
                 'dense': each keyframe contributes its strongest 300 points with depth
  matching       mutual nearest neighbour on SuperPoint descriptors (query picks vs all map points)
  solver         cv2.solvePnPRansac, 3 px, 2000 iterations, confidence 0.9999, >= 6 inliers
  errors         rotation (deg) and camera position (cm) against the true pose; fail thresholds are applied later (export/page)
  repeatability  share of kept points (with depth) re-found at +1/+5/+10 frames: lifted with true depth and pose, projected into
                 the later frame, within 3 px of one of its strongest 300 candidates whose depth agrees within 5 %
Output rows per query: flight, q, err_cm, rot_deg, n_sel, n_match, n_inl, rep1, rep5, rep10 (see docs/data_spec.md).
"""
import os, time
import numpy as np, cv2
from multiprocessing import Pool
from scipy.spatial import cKDTree
from .data import Flight, to3d, DEPTH_MAX, KMAT, FX, FY, CX, CY, W, H
from .cache import Cands
from .pickers import parse as parse_picker

REP_LAGS = (1, 5, 10)
COLS = ['flight', 'q', 'err_cm', 'rot_deg', 'n_sel', 'n_match', 'n_inl', 'rep1', 'rep5', 'rep10']


def match_mutual(dA, dB):
    sim = dA.astype(np.float32) @ dB.astype(np.float32).T
    m1 = sim.argmax(1); m0 = sim.argmax(0)
    mu = np.flatnonzero(m0[m1] == np.arange(len(m1)))
    return mu, m1[mu]


def pnp(obj, img, Rq, tq):
    """-> (err_cm, rot_deg, n_inl); err/rot None when no pose"""
    if len(obj) < 6:
        return None, None, 0
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), KMAT, None,
                                                 reprojectionError=3.0, iterationsCount=2000, confidence=0.9999,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None, None, 0
    if not ok or inl is None or len(inl) < 6:
        return None, None, 0
    R, _ = cv2.Rodrigues(rvec); R_gt = Rq.T; t_gt = -Rq.T @ tq
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1))))
    err = float(np.linalg.norm(tvec.ravel() - t_gt) * 100)
    return round(err, 3), round(rot, 4), int(len(inl))


def _worker(job):
    q, obj, img, Rq, tq = job
    return q, pnp(obj, img, Rq, tq)


def repeat(F, C, q, idx, trees):
    """repeatability of candidates idx of frame q at the REP_LAGS; trees caches (xy KD-tree, z) of the strongest 300 per frame"""
    u, v, s, z, _ = C.frame(q)
    ok = idx[z[idx] < DEPTH_MAX]
    out = []
    if not len(ok):
        return [None] * len(REP_LAGS)
    Rq, tq = F.cam_pose(q); Xw = to3d(u[ok], v[ok], z[ok]) @ Rq.T + tq
    for L in REP_LAGS:
        f = q + L
        if f >= F.N:
            out.append(None); continue
        if f not in trees:
            u2, v2, s2, z2, _ = C.frame(f); j = np.flatnonzero(z2 < DEPTH_MAX)[:300]
            trees[f] = (cKDTree(np.stack([u2[j], v2[j]], 1)) if len(j) else None, z2[j])
        tree, z2 = trees[f]
        if tree is None:
            out.append(0.0); continue
        Rf, tf = F.cam_pose(f); Xc = (Xw - tf) @ Rf; zc = Xc[:, 2]; good = zc > 0.05
        pu = FX * Xc[:, 0] / np.where(good, zc, 1) + CX; pv = FY * Xc[:, 1] / np.where(good, zc, 1) + CY
        d, nn = tree.query(np.stack([pu, pv], 1), distance_upper_bound=3.0)
        hit = good & np.isfinite(d)
        hit[hit] &= np.abs(z2[nn[hit]] - zc[hit]) <= 0.05 * zc[hit]
        out.append(round(float(hit.mean()), 4))
    return out


def run_flight(fid, det, picker, K, mapmode, workers=None, save_picks=False, flight_index=0, verbose=True):
    F = Flight(fid); C = Cands(F.slug, det); pk = parse_picker(picker); t0 = time.time()
    if C.N != F.N:
        raise RuntimeError('cache has %d frames, flight has %d' % (C.N, F.N))
    sel = {}
    for f in range(F.N):
        u, v, s, z, _ = C.frame(f); sel[f] = np.asarray(pk(u, v, s, z, K), int)
    MAP = {}
    for k in F.keyframes:
        u, v, s, z, de = C.frame(k)
        i = sel[k] if mapmode == 'same' else np.flatnonzero(z < DEPTH_MAX)[:300]
        i = i[z[i] < DEPTH_MAX]; Rk, tk = F.cam_pose(k)
        MAP[k] = (to3d(u[i], v[i], z[i]) @ Rk.T + tk, de[i])
    jobs, meta, picks = [], {}, {}
    for q in F.queries:
        near = F.near(q); Rq, tq = F.cam_pose(q)
        P3 = np.concatenate([MAP[k][0] for k in near]); DM = np.concatenate([MAP[k][1] for k in near])
        u, v, s, z, de = C.frame(q); i = sel[q]; mq, mm = match_mutual(de[i], DM)
        jobs.append((q, P3[mm], np.stack([u[i][mq], v[i][mq]], 1), Rq, tq))
        meta[q] = dict(n_sel=int(len(i)), n_match=int(len(mq)))
        if save_picks:
            picks[str(q)] = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(u[i], v[i])]
    if verbose:
        print(fid, det, picker, K, mapmode, 'matching done, %d queries, %.0fs' % (len(jobs), time.time() - t0), flush=True)
    nw = workers or int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or max(1, (os.cpu_count() or 2) - 1)
    cv2.setNumThreads(1)
    if nw > 1:
        with Pool(nw) as P:
            res = dict(P.map(_worker, jobs, chunksize=16))
    else:
        res = dict(map(_worker, jobs))
    trees = {}; rows = []
    for q in F.queries:
        err, rot, ninl = res[q]; rep = repeat(F, C, q, sel[q], trees)
        for f in [f for f in trees if f < q]:
            del trees[f]
        rows.append([flight_index, q, err, rot, meta[q]['n_sel'], meta[q]['n_match'], ninl] + rep)
    if verbose:
        fails = np.mean([r[2] is None or r[2] > 25 or r[3] > 5 for r in rows])
        print(fid, det, picker, K, mapmode, 'fail@25cm %.2f %%  %.0fs' % (100 * fails, time.time() - t0), flush=True)
    return rows, picks
