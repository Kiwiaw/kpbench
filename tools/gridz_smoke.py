"""Synthetic smoke of the gridz family (no images, poses or GPU): cells cover equal plane area, clamp gives no NaN.

  python tools/gridz_smoke.py [--out results/micro/gridz_smoke.json]

Uniform random points on a plane (fronto-parallel and tilted 30 / 60 deg) are projected to u, v, z; the picker's own cell
key is rebuilt and the interior cells' point counts compared: equal plane area -> equal counts, up to Poisson noise
(area_cv = sqrt(max(0, cv^2 - 1/mean count))). PASS = area_cv <= 0.10, K picks returned, distinct, finite.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench import pickers
from kpbench.data import to3d, W, H, FX, FY, CX, CY

ROOT = os.path.join(os.path.dirname(__file__), '..')


def plane(tilt, n, seed):
    """uniform on a plane through (0, 0, 8) rotated by tilt about camera x, only points inside the image and in front"""
    r = np.random.default_rng(seed); t = np.radians(tilt)
    a, b = r.uniform(-30, 30, (2, n)); P = np.stack([a, b * np.cos(t), 8 + b * np.sin(t)], 1)
    P = P[P[:, 2] > 0.5]; u = FX * P[:, 0] / P[:, 2] + CX; v = FY * P[:, 1] / P[:, 2] + CY
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok], v[ok], P[ok, 2]


def area_cv(Q, size):
    key = np.floor(Q / size).astype(int); _, inv, c = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    occ = {tuple(k) for k in np.unique(key, axis=0)}; uk = np.unique(key, axis=0)
    # interior = all 8 neighbours occupied, so the image border does not cut the cell
    inner = np.array([all((k[0] + i, k[1] + j) in occ for i in (-1, 0, 1) for j in (-1, 0, 1)) for k in uk])
    x = c[inner].astype(float)
    return dict(cells=int(inner.sum()), mean=round(float(x.mean()), 1) if len(x) else 0,
                area_cv=round(float(np.sqrt(max(0.0, x.var() / x.mean() ** 2 - 1 / x.mean()))), 4) if len(x) > 1 else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); a = ap.parse_args()
    rows = []; ok = True
    for tilt in (0, 30, 60):
        u, v, z = plane(tilt, 400000, tilt)
        P = to3d(u, v, z)
        for name, Q in (('gridz', P[:, :2]), ('gridzr', pickers.plane_coords(P))):
            r = dict(tilt=tilt, family=name, **area_cv(Q, 0.5))
            sel = pickers.parse('%s0.5m1' % name)(u[:3000], v[:3000], np.arange(3000, 0, -1.0), z[:3000], 100)
            r['picks'] = len(sel); r['distinct'] = len(set(sel.tolist()))
            r['pass'] = r['area_cv'] is not None and r['area_cv'] <= 0.10 and r['picks'] == 100 == r['distinct']
            ok = ok and r['pass']; rows.append(r); print(r)
    # clamp: tiny depths give finite cells, K picks (a NaN key would raise in astype or merge everything)
    u, v = np.random.default_rng(1).uniform(0, W, 500), np.random.default_rng(2).uniform(0, H, 500); z = np.full(500, 1e-6)
    for n in ('gridz0.5m1', 'gridzr0.5m1'):
        sel = pickers.parse(n)(u, v, np.arange(500, 0, -1.0), z, 100); c = len(sel) == 100 == len(set(sel.tolist()))
        rows.append(dict(clamp=n, picks=len(sel), **{'pass': c})); ok = ok and c; print(rows[-1])
    print('PASS' if ok else 'FAIL')
    if a.out:
        json.dump(dict(ok=ok, rows=rows), open(os.path.join(ROOT, a.out), 'w'), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
