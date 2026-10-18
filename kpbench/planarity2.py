"""Second planarity definition (T11 / MAIN, dispatch 2026-10-18 13:15): plane-fit residual instead of RANSAC inlier share.

  python -m kpbench.planarity2 [--flights all]    # results/labels_resid/<slug>.json + agreement table on stdout

plane_resid = RMS orthogonal distance of the strongest 300 SuperPoint candidates with depth (camera frame, the same points as
plane_frac) to their total-least-squares plane, divided by their median depth (scale-free: 0 = a perfect plane).
flat2 = plane_resid <= RESID_FLAT. RESID_FLAT is fixed once (calibrated on MICRO + contaminated DEV flights so the flat2 count
equals the plane_frac >= 0.85 count there) and never retuned. Labels from kpbench.labels are not touched; needs only the sp
candidate cache, no poses. Gates and the leaderboard keep plane_frac; this is the robustness check of reviewer objection
'the flat bin is one definition'.
"""
import os, json
import numpy as np
from .data import to3d, DEPTH_MAX, slug
from .cache import Cands, REPO
from .splits import FLAT

OUT = os.path.join(REPO, 'results', 'labels_resid')
RESID_FLAT = 0.14074   # fixed 18 Oct 2026: flat2 count = plane_frac >= 0.85 count (814) over the 6 flights with sp caches on the laptop


def plane_resid(P):
    """RMS distance to the total-least-squares plane / median depth; nan for < 10 points"""
    if len(P) < 10:
        return float('nan')
    X = P - P.mean(0)
    return float(np.sqrt(np.linalg.eigvalsh(X.T @ X / len(P))[0]) / max(np.median(P[:, 2]), 1e-6))


def run_flight(fid):
    C = Cands(slug(fid), 'sp'); lab = json.load(open(os.path.join(REPO, 'results', 'labels', slug(fid) + '.json')))
    out = {}
    for q in lab:
        u, v, s, z, _ = C.frame(int(q)); ok = np.flatnonzero(z < DEPTH_MAX)[:300]
        r = plane_resid(to3d(u[ok], v[ok], z[ok])); out[q] = None if np.isnan(r) else round(r, 5)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, slug(fid) + '.json'), 'w'))
    return {q: (lab[q].get('plane_frac'), out[q]) for q in lab}


def main(argv=None):
    import argparse
    from .data import FLIGHTS
    ap = argparse.ArgumentParser(); ap.add_argument('--flights', default='all'); a = ap.parse_args(argv)
    fl = FLIGHTS if a.flights == 'all' else a.flights.split(',')
    rows = []
    for fid in fl:
        try:
            rows += [(fid, pf, r) for pf, r in run_flight(fid).values() if pf is not None and r is not None]
        except FileNotFoundError:
            print('skip', fid, '(no sp cache)')
    pf = np.array([x[1] for x in rows]); r = np.array([x[2] for x in rows]); flat = pf >= FLAT
    thr = RESID_FLAT if RESID_FLAT is not None else float(np.sort(r)[max(int(flat.sum()) - 1, 0)])
    flat2 = r <= thr; rho = np.corrcoef(np.argsort(np.argsort(pf)), np.argsort(np.argsort(r)))[0, 1]
    print(json.dumps(dict(n=len(rows), flights=sorted({x[0] for x in rows}), n_flat=int(flat.sum()), resid_flat=round(thr, 5),
                          n_flat2=int(flat2.sum()), both=int((flat & flat2).sum()), agree=round(float((flat == flat2).mean()), 4),
                          jaccard_flat=round(float((flat & flat2).sum() / max((flat | flat2).sum(), 1)), 4), spearman=round(float(rho), 4))))


if __name__ == '__main__':
    main()
