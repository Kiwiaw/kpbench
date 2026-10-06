"""Per-query scene labels (picker independent): planarity, depth statistics, map coverage.

Output: results/labels/<flight_slug>.json  {q: {plane_frac, plane_frac_grid, med_depth, missing_depth, coverage}}
plane_frac      share of the strongest 300 SuperPoint candidates (with depth) that lie on one plane (RANSAC, 30 cm),
                exactly as the earlier C22/C23 reports, so bins stay comparable.
plane_frac_grid detector-free version on a 20 x 15 pixel grid of the depth map (sanity check for the above).
med_depth       median depth (m) of those 300 candidates.
missing_depth   share of the strongest 100 SuperPoint candidates without depth (sky).
coverage        share of the query's pixels (8 px grid, with depth) seen by at least one map keyframe within +-20 frames
                (inside that keyframe's image and not occluded: projected depth within 10 % of the keyframe's depth).
"""
import os, sys, json, time
import numpy as np
from .data import Flight, to3d, DEPTH_MAX, W, H, FX, FY, CX, CY, parse_flights
from .cache import Cands, REPO

OUT_DIR = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))


def plane_frac(P, tol=0.3, iters=300, seed=0):
    """largest share of points within tol of a plane through 3 random points (300 trials, fixed seed)"""
    if len(P) < 10:
        return float('nan')
    rng = np.random.default_rng(seed); best = 0
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False); a, b, c = P[i]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; best = max(best, int((np.abs((P - a) @ n) < tol).sum()))
    return best / len(P)


def grid_points(dp, step=32):
    vv, uu = np.mgrid[step // 2:H:step, step // 2:W:step]
    z = dp[vv, uu]; ok = z < DEPTH_MAX
    return to3d(uu[ok].astype(float), vv[ok].astype(float), z[ok])


def coverage(F, q, depths, step=8):
    """share of q's pixels with depth that some keyframe within +-20 frames sees"""
    dq = depths[q]
    vv, uu = np.mgrid[step // 2:H:step, step // 2:W:step]
    z = dq[vv, uu]; ok = z < DEPTH_MAX
    if not ok.any():
        return float('nan')
    Rq, tq = F.cam_pose(q)
    Xw = to3d(uu[ok].astype(float), vv[ok].astype(float), z[ok]) @ Rq.T + tq
    seen = np.zeros(len(Xw), bool)
    for k in F.near(q):
        Rk, tk = F.cam_pose(k); Xc = (Xw - tk) @ Rk
        zc = Xc[:, 2]; good = zc > 0.05
        pu = FX * Xc[:, 0] / np.where(good, zc, 1) + CX; pv = FY * Xc[:, 1] / np.where(good, zc, 1) + CY
        inside = good & (pu >= 0) & (pu <= W - 1) & (pv >= 0) & (pv <= H - 1)
        if not inside.any():
            continue
        dk = depths[k][np.clip(np.rint(pv[inside]).astype(int), 0, H - 1), np.clip(np.rint(pu[inside]).astype(int), 0, W - 1)]
        vis = np.abs(dk - zc[inside]) <= 0.10 * zc[inside]
        idx = np.flatnonzero(inside)[vis]; seen[idx] = True
    return float(seen.mean())


def run_flight(fid):
    F = Flight(fid); out = os.path.join(OUT_DIR, 'labels'); os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, F.slug + '.json')
    if os.path.isfile(fn):
        print('labels exist', fn); return
    C = Cands(F.slug, 'sp'); t0 = time.time()
    depths = {}
    def dep(f):
        if f not in depths:
            depths[f] = F.depth(f)
        return depths[f]
    lab = {}
    for q in F.queries:
        u, v, s, z, _ = C.frame(q)
        ok = np.flatnonzero(z < DEPTH_MAX)[:300]
        P = to3d(u[ok], v[ok], z[ok])
        dq = dep(q)
        for k in F.near(q):
            dep(k)
        lab[q] = dict(plane_frac=plane_frac(P), plane_frac_grid=plane_frac(grid_points(dq)),
                      med_depth=float(np.median(z[ok])) if len(ok) else None,
                      missing_depth=float(np.mean(z[:100] >= DEPTH_MAX)) if len(z) else None,
                      coverage=coverage(F, q, depths))
        for f in list(depths):          # keep only what the next queries need
            if f < q - 2 * 10 - 1:
                del depths[f]
        if q % 100 == 1:
            print(F.id, q, F.N, '%.0fs' % (time.time() - t0), flush=True)
    json.dump({str(q): {k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in d.items()} for q, d in lab.items()},
              open(fn, 'w'))
    print(F.id, 'labels done', len(lab), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--flights', default='all'); a = ap.parse_args()
    for fid in parse_flights(a.flights):
        run_flight(fid)
