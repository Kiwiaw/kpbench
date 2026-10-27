"""Picker-level smoke of any picker names on the cached MICRO candidates (no images, poses or GPU needed).

  python tools/picker_smoke.py dpp2d120a1 dpp0.5 sd --K 200 [--det raco] [--out results/micro/picker_smoke_K200.json]

Checks per MICRO frame (splits.json 'micro', candidates from cache/<slug>_<det>.npz):
  fill     the picker returns min(K, #candidates with depth) distinct indices, all with depth
  runs     no exception
Reports per picker: fill fails, mean pixel min pair distance, mean pixel spread (sqrt of the trace of the pick covariance),
and ms per frame. Same picks -> same PnP, so a picker that passes here only lacks the pose rows (Euler).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench import pickers
from kpbench.data import DEPTH_MAX

ROOT = os.path.join(os.path.dirname(__file__), '..')


def min_dist(u, v):
    d = (u[:, None] - u[None]) ** 2 + (v[:, None] - v[None]) ** 2
    d[np.diag_indices(len(u))] = np.inf
    return float(np.sqrt(d.min())) if len(u) > 1 else np.inf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pickers', nargs='+'); ap.add_argument('--det', default='raco')
    ap.add_argument('--K', type=int, default=200); ap.add_argument('--out', default=None)
    a = ap.parse_args(); K = a.K
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    pk = {p: pickers.parse(p) for p in a.pickers}
    res = {p: dict(fill_fail=0, error=0, mindist=[], spread=[], ms=0.0) for p in a.pickers}
    frames = 0
    for fid, fr in micro.items():
        c = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (fid.replace('/', '_'), a.det)))
        for f in fr:
            n = int(c['n'][f]); u, v, s, z = (c[k][f, :n].astype(float) for k in 'uvsz')
            want = min(K, int((z < DEPTH_MAX).sum())); frames += 1
            for p, fn in pk.items():
                r = res[p]; t = time.perf_counter()
                try:
                    i = np.asarray(fn(u, v, s, z, K))
                except Exception:
                    r['error'] += 1; continue
                r['ms'] += 1e3 * (time.perf_counter() - t)
                if len(i) != want or len(set(i.tolist())) != len(i) or not (z[i] < DEPTH_MAX).all():
                    r['fill_fail'] += 1
                r['mindist'].append(min_dist(u[i], v[i]))
                r['spread'].append(float(np.sqrt(np.var(u[i]) + np.var(v[i]))))
    for p, r in res.items():
        r['mindist'] = round(float(np.mean(r['mindist'])), 2) if r['mindist'] else None
        r['spread'] = round(float(np.mean(r['spread'])), 1) if r['spread'] else None
        r['ms'] = round(r['ms'] / max(frames, 1), 1)
        r['pass'] = r['fill_fail'] == 0 and r['error'] == 0
    out = dict(det=a.det, K=K, frames=frames, pickers=res)
    p = os.path.join(ROOT, a.out or 'results/micro/picker_smoke_K%d.json' % K); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
