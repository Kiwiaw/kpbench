"""The task: relocalise every non-keyframe frame against a local map built from ground-truth depth and poses.

Protocol (the C23 protocol, unchanged):
  keyframes      every 10th frame; a query uses the keyframes within +-20 frames (usually 4)
  map            'same': each keyframe contributes the SAME picker's K points (with depth), lifted to 3D with its true pose;
                 keyframes = those within +-20 frames of the query (temporal window = perfect retrieval, includes future frames)
                 'ret<k>': same keyframe points, but the database is ALL keyframes of the flight and a query is matched against
                 the top-k by image retrieval (VLAD over the SuperPoint descriptors of each frame's 300 strongest candidates,
                 64-word codebook learnt on the flight's keyframes). No time information is used: the localisation setting.
                 'dense': each keyframe contributes its strongest 300 points with depth
  matching       'nn': mutual nearest neighbour on SuperPoint descriptors (query picks vs all map points)
                 'lg': LightGlue (SuperPoint weights) between the query picks and each keyframe's map points in turn;
                       a query point keeps its highest-scoring match over the keyframes
  solver         cv2.solvePnPRansac, 3 px, 2000 iterations, confidence 0.9999, >= 6 inliers
  errors         rotation (deg) and camera position (cm) against the true pose; fail thresholds are applied later (export/page)
  repeatability  share of kept points (with depth) re-found at +1/+5/+10 frames: lifted with true depth and pose, projected into
                 the later frame, within 3 px of one of its strongest 300 candidates whose depth agrees within 5 %
Output rows per query: flight, q, err_cm, rot_deg, n_sel, n_match, n_inl, rep1, rep5, rep10 (see docs/data_spec.md).
"""
import os, time
import re
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


_LG = None


def lightglue():
    """LightGlue with SuperPoint weights, loaded once per process (DEV env: cuda | cpu, default cuda if available)"""
    global _LG
    if _LG is None:
        import torch
        from lightglue import LightGlue
        dev = os.environ.get('DEV') or ('cuda' if torch.cuda.is_available() else 'cpu')
        _LG = (LightGlue(features='superpoint').eval().to(dev), dev)
    return _LG


def match_lightglue(uvq, dq, kfs):
    """query points (uv, desc) against each keyframe's (uv, desc) with LightGlue; kfs = [(uv_k, desc_k), ...]
    -> (query indices, (keyframe index, point index) per match): the best-scoring match of every query point"""
    import torch
    LG, dev = lightglue()
    best = {}
    with torch.no_grad():
        fq = dict(keypoints=torch.from_numpy(uvq.astype(np.float32))[None].to(dev),
                  descriptors=torch.from_numpy(dq.astype(np.float32))[None].to(dev),
                  image_size=torch.tensor([[W, H]], dtype=torch.float32, device=dev))
        for ki, (uvk, dk) in enumerate(kfs):
            if len(uvk) < 2:
                continue
            fk = dict(keypoints=torch.from_numpy(uvk.astype(np.float32))[None].to(dev),
                      descriptors=torch.from_numpy(dk.astype(np.float32))[None].to(dev),
                      image_size=torch.tensor([[W, H]], dtype=torch.float32, device=dev))
            out = LG({'image0': fq, 'image1': fk})
            m = out['matches'][0].cpu().numpy(); sc = out['scores'][0].cpu().numpy()
            for (a, b), s in zip(m, sc):
                if a not in best or s > best[a][0]:
                    best[a] = (float(s), ki, int(b))
    qi = np.array(sorted(best), int)
    return qi, [(best[a][1], best[a][2]) for a in qi]


def pnp(obj, img, Rq, tq):
    """-> (err_cm, rot_deg, n_inl); err/rot None when no pose.
    RANSAC with the AP3P minimal solver (3 px, 2000 it, conf 0.9999), then Levenberg-Marquardt refinement on all inliers,
    as in hloc / visuallocalization.net. The OpenCV RNG is seeded per query, so a run is reproducible exactly.
    Errors as in the visual localisation benchmarks: position = distance between the estimated and true camera CENTRES
    in metres (c = -R^T t), rotation = angle of R_est R_true^T in degrees."""
    if len(obj) < 6:
        return None, None, 0
    obj = np.ascontiguousarray(obj.reshape(-1, 1, 3), dtype=np.float64); img = np.ascontiguousarray(img.reshape(-1, 1, 2), dtype=np.float64)
    try:
        cv2.setRNGSeed(0)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, KMAT, None, reprojectionError=3.0, iterationsCount=2000,
                                                 confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < 6:
            return None, None, 0
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], KMAT, None, rvec, tvec)
    except cv2.error:
        return None, None, 0
    R, _ = cv2.Rodrigues(rvec); R_gt = Rq.T
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1))))
    c_est = -R.T @ tvec.ravel()                       # estimated camera centre in the world frame
    err = float(np.linalg.norm(c_est - tq) * 100)     # tq = true camera centre
    return round(err, 3), round(rot, 4), int(len(inl))


def vlad_codebook(C, frames, k=64, seed=0, cap=60000):
    """k-means codebook over the descriptors of the 300 strongest candidates of the given frames"""
    from scipy.cluster.vq import kmeans2
    D = np.concatenate([C.frame(f)[4][:300] for f in frames]).astype(np.float32)
    if len(D) > cap:
        D = D[np.random.default_rng(seed).choice(len(D), cap, replace=False)]
    cb, _ = kmeans2(D, k, minit='++', seed=seed)
    return cb


def vlad(desc, cb):
    """VLAD global descriptor of one frame: residuals to the nearest word, intra-normalised, signed sqrt, L2"""
    d = desc[:300].astype(np.float32)
    a = np.argmax(d @ cb.T - 0.5 * (cb ** 2).sum(1)[None], 1)
    V = np.zeros_like(cb); np.add.at(V, a, d - cb[a])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    v = V.ravel(); v = np.sign(v) * np.sqrt(np.abs(v))
    return v / (np.linalg.norm(v) + 1e-9)


def retrieval(C, F, mapmode):
    """mapmode 'ret<k>' -> function q -> top-k keyframes by VLAD similarity; otherwise F.near (temporal window)"""
    m = re.fullmatch(r'ret([0-9]+)', mapmode)
    if not m:
        return F.near
    k = int(m.group(1)); cb = vlad_codebook(C, F.keyframes)
    KV = np.stack([vlad(C.frame(f)[4], cb) for f in F.keyframes])
    def near(q):
        sim = KV @ vlad(C.frame(q)[4], cb)
        return [F.keyframes[i] for i in np.argsort(-sim)[:k]]
    return near


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


def run_flight(fid, det, picker, K, mapmode, workers=None, save_picks=False, flight_index=0, verbose=True, matcher='nn'):
    F = Flight(fid); C = Cands(F.slug, det); pk = parse_picker(picker); t0 = time.time()
    if C.N != F.N:
        raise RuntimeError('cache has %d frames, flight has %d' % (C.N, F.N))
    sel = {}
    for f in range(F.N):
        u, v, s, z, de = C.frame(f)
        kw = {}
        if getattr(pk, 'needs_ctx', False): kw['ctx'] = (F, C, f)       # oracle pickers see the flight
        if getattr(pk, 'needs_desc', False): kw['de'] = de               # descriptor-based pickers
        sel[f] = np.asarray(pk(u, v, s, z, K, **kw), int)
    MAP = {}
    for k in F.keyframes:
        u, v, s, z, de = C.frame(k)
        i = sel[k] if mapmode != 'dense' else np.flatnonzero(z < DEPTH_MAX)[:300]
        i = i[z[i] < DEPTH_MAX]; Rk, tk = F.cam_pose(k)
        MAP[k] = (to3d(u[i], v[i], z[i]) @ Rk.T + tk, de[i], np.stack([u[i], v[i]], 1))
    jobs, meta, picks = [], {}, {}; near_of = retrieval(C, F, mapmode)
    for q in F.queries:
        near = near_of(q); Rq, tq = F.cam_pose(q)
        P3 = np.concatenate([MAP[k][0] for k in near]); DM = np.concatenate([MAP[k][1] for k in near])
        u, v, s, z, de = C.frame(q); i = sel[q]
        if matcher == 'lg':
            mq, pairs = match_lightglue(np.stack([u[i], v[i]], 1), de[i], [(MAP[k][2], MAP[k][1]) for k in near])
            obj = np.array([MAP[near[ki]][0][pi] for ki, pi in pairs]).reshape(-1, 3)
        else:
            mq, mm = match_mutual(de[i], DM); obj = P3[mm]
        jobs.append((q, obj, np.stack([u[i][mq], v[i][mq]], 1), Rq, tq))
        meta[q] = dict(n_sel=int(len(i)), n_match=int(len(mq)))
        if save_picks:
            picks[str(q)] = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(u[i], v[i])]
    if verbose:
        print(fid, det, picker, K, mapmode, matcher, 'matching done, %d queries, %.0fs' % (len(jobs), time.time() - t0), flush=True)
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
        fails = np.mean([r[2] is None or r[2] > 25 or r[3] > 2 for r in rows])
        print(fid, det, picker, K, mapmode, matcher, 'fail@25cm/2deg %.2f %%  %.0fs' % (100 * fails, time.time() - t0), flush=True)
    return rows, picks
