"""Pick-level smoke of the deco picker (E4b) on the cached MICRO candidates (no images, poses or GPU needed).

  python tools/deco_smoke.py [--K 100] [--out results/main/deco_smoke.md]

Checks: (a) deco b0 picks = sd picks (beta 0 removes the co-visibility term); (b) deco (b1) changes the pick set vs sd on
>= 5 % of frames and its co-visibility r is not degenerate (mean per-frame std > 0.05); (c) runtime <= 2x dppr0.317a1.6b2.01t0.9
(same track count). Also prints the 3D spread of the picks (mean camera-frame nearest-neighbour distance, m) for sd, deco,
cube0.5 and dpp0.5: a ranker should stay near sd, the spread class above it.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scipy.spatial import cKDTree
from kpbench import pickers
from kpbench.cache import Cands
from kpbench.data import to3d

ROOT = os.path.join(os.path.dirname(__file__), '..')


class Fl:
    def __init__(self, fid):
        self.slug = fid.replace('/', '_')


def nn_spread(u, v, z, i):
    P = to3d(u[i], v[i], z[i])
    return float(cKDTree(P).query(P, 2)[0][:, 1].mean()) if len(P) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'main', 'deco_smoke.md')); a = ap.parse_args()
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    names = ['sd', 'deco', 'decob0', 'cube0.5', 'dpp0.5', 'dppr0.317a1.6b2.01t0.9']
    fns = {p: pickers.parse(p) for p in names}; T = dict.fromkeys(names, 0.0); spread = {p: [] for p in names}
    diff_b1 = diff_b0 = n = 0
    for fid, fr in micro.items():
        F = Fl(fid); C = Cands(F.slug, a.det)
        for q in fr:
            u, v, s, z, de = C.frame(q); sel = {}
            for p, fn in fns.items():
                kw = {}
                if getattr(fn, 'needs_past', False): kw['ctx'] = (F, C, q)
                if getattr(fn, 'needs_desc', False): kw['de'] = de
                t0 = time.time(); sel[p] = np.sort(np.asarray(fn(u, v, s, z, a.K, **kw), int)); T[p] += time.time() - t0
                spread[p].append(nn_spread(u, v, z, sel[p]))
            n += 1; diff_b1 += not np.array_equal(sel['deco'], sel['sd']); diff_b0 += not np.array_equal(sel['decob0'], sel['sd'])
    rstd = float(np.mean(fns['deco'].rstd)); ratio = T['deco'] / max(T['dppr0.317a1.6b2.01t0.9'], 1e-9)
    ok = dict(a=diff_b0 == 0, b=diff_b1 >= 0.05 * n and rstd > 0.05, c=ratio <= 2.0)
    L = ['# E4b deco picker: MICRO pick-level smoke (%s, K %d)' % (time.strftime('%Y-%m-%d'), a.K), '',
         'Source: tools/deco_smoke.py on cache/*_%s.npz, %d MICRO flat frames (%s). No poses on the laptop, so no fail %%;' % (a.det, n, ', '.join(micro)),
         'fail % comes from the Euler DEV gate.', '',
         '| check | number | pass |', '|---|---|---|',
         '| (a) deco b0 = sd | %d / %d frames differ | %s |' % (diff_b0, n, ok['a']),
         '| (b) deco b1 vs sd | %d / %d frames differ (%.1f %%), mean per-frame std of r %.3f | %s |' % (diff_b1, n, 100.0 * diff_b1 / n, rstd, ok['b']),
         '| (c) runtime vs dppr0.317 | %.2fx (%.1f s vs %.1f s) | %s |' % (ratio, T['deco'], T['dppr0.317a1.6b2.01t0.9'], ok['c']), '',
         'Mean 3D nearest-neighbour distance between picks (m, camera frame; higher = more spread):', '',
         '| picker | nn dist m | pick s |', '|---|---|---|'] + \
        ['| %s | %.3f | %.1f |' % (p, np.mean(spread[p]), T[p]) for p in names] + \
        ['', 'Verdict: %s' % ('PASS (a, b, c)' if all(ok.values()) else 'FAIL ' + ','.join(k for k, v in ok.items() if not v))]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
