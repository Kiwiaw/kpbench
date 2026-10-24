"""Picker-level smoke of nms<r> on the cached MICRO candidates (no images, poses or GPU needed).

  python tools/nms_smoke.py [--det raco] [--K 100] [--out results/micro/nms_smoke.json]

Checks per MICRO frame (splits.json 'micro', candidates from cache/<slug>_<det>.npz):
  identity   nms0 picks == sd picks (same indices, same order)
  fill       nms16 picks K distinct candidates with depth whenever sd does
  spread     the NMS part of nms16 (before the score fill) has no pair closer than 16 px
Reports per radius the share of frames whose K picks are all NMS survivors (no fill) and the mean min pair distance.
Pose fail flags follow from the picks (same picks -> same PnP), so identity of picks is identity of rows.
"""
import argparse, json, os, sys
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
    ap = argparse.ArgumentParser(); ap.add_argument('--det', default='raco'); ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--radii', default='0,4,8,16'); ap.add_argument('--out', default='results/micro/nms_smoke.json')
    a = ap.parse_args(); K = a.K; radii = [int(r) for r in a.radii.split(',')]
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    sd = pickers.parse('sd'); nms = {r: pickers.parse('nms%d' % r) for r in radii}
    out = dict(det=a.det, K=K, radii=radii, flights={}, frames=0, identity_fail=0, fill_fail=0, spread_fail=0,
               nofill={r: 0 for r in radii}, mindist={r: [] for r in radii})
    for fid, frames in micro.items():
        c = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (fid.replace('/', '_'), a.det)))
        for f in frames:
            n = int(c['n'][f]); u, v, s, z = (c[k][f, :n].astype(float) for k in 'uvsz')
            ok = np.flatnonzero(z < DEPTH_MAX); want = min(K, len(ok)); i_sd = sd(u, v, s, z, K)
            out['frames'] += 1
            for r in radii:
                i = nms[r](u, v, s, z, K)
                if r == 0 and i.tolist() != i_sd.tolist():
                    out['identity_fail'] += 1
                if len(i) != want or len(set(i.tolist())) != len(i) or not (z[i] < DEPTH_MAX).all():
                    out['fill_fail'] += r == max(radii)
                core = ok[pickers.nms_order(u[ok], v[ok], r)][:K]
                if r > 0 and min_dist(u[core], v[core]) < r:
                    out['spread_fail'] += 1
                out['nofill'][r] += len(core) >= K
                out['mindist'][r].append(min_dist(u[i], v[i]))
        out['flights'][fid] = len(frames)
    out['nofill'] = {r: round(out['nofill'][r] / out['frames'], 4) for r in radii}
    out['mindist'] = {r: round(float(np.mean(out['mindist'][r])), 2) for r in radii}
    out['pass'] = out['identity_fail'] == 0 and out['fill_fail'] == 0 and out['spread_fail'] == 0
    p = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1); print(json.dumps(out))


if __name__ == '__main__':
    main()
