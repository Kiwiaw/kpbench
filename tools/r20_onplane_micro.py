"""R20 schema check on the cached MICRO candidates (no images, poses or GPU): on-plane share of the picks per picker.

  python tools/r20_onplane_micro.py [--K 100] [--pickers sd,grid,cube0.5,dpp0.5] [--out results/sub3/r20_schema.md]

Per frame and picker: bench.onplane = share of the K picks (with depth) within 30 cm of the frame's dominant plane (RANSAC over
the strongest 300 candidates with depth, the labels.plane_frac fit). Flat frames = plane_frac >= 0.85 from results/labels.
Reported: mean share over all and over flat frames with a frame bootstrap 95 % CI, and the paired difference to sd per frame.
Prediction (ed. 19 contribution 2): spread pickers keep more off-plane points on flat frames, so share(sd) > share(cube0.5), share(dpp0.5).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kpbench import pickers
from kpbench.bench import onplane
from kpbench.cache import Cands
from kpbench.splits import FLAT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--pickers', default='sd,grid,cube0.5,dpp0.5'); ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'r20_schema.md'))
    a = ap.parse_args(); names = a.pickers.split(','); fns = {p: pickers.parse(p) for p in names}
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    share = {p: [] for p in names}; flat = []; fit = []; n = 0
    for fid, fr in micro.items():
        slug = fid.replace('/', '_'); C = Cands(slug, a.det); lab = json.load(open(os.path.join(ROOT, 'results', 'labels', slug + '.json')))
        for q in fr:
            u, v, s, z, de = C.frame(q)
            pf = lab[str(q)].get('plane_frac'); flat.append(pf is not None and pf >= FLAT)
            for p, fn in fns.items():
                kw = {'de': de} if getattr(fn, 'needs_desc', False) else {}
                r = onplane(u, v, z, np.asarray(fn(u, v, s, z, a.K, **kw), int))[0]
                share[p].append(np.nan if r[0] is None else r[0])
                if p == names[0]:
                    fit.append(np.nan if r[2] is None else r[2])
            n += 1
    flat = np.array(flat, bool); S = {p: np.array(v, float) for p, v in share.items()}
    L = ['# R20 schema: on-plane share of the picks on MICRO (%s, %d frames, %d flat, K %d)' % (time.strftime('%Y-%m-%d'), n, int(flat.sum()), a.K), '',
         'tools/r20_onplane_micro.py on the cached candidates (no poses, no fail %%). on-plane share = picks with depth within 30 cm of the frame\'s dominant plane '
         '(RANSAC, strongest 300 with depth); the same number is now stored per query by kpbench.run / kpbench.evaluate (bench.onplane). Plane share of the fit set: mean %.3f all, %.3f flat.' % (np.nanmean(fit), np.nanmean(np.array(fit)[flat])), '',
         '| picker | on-plane share all [95 %%] | flat [95 %%] | flat diff vs %s [95 %%] | frames with lower share than %s (flat) |' % (names[0], names[0]), '|---|---|---|---|---|']
    out = {}
    for p in names:
        x = S[p]; d = x - S[names[0]]; xf, df = x[flat], d[flat]; ok = np.isfinite(xf); okd = np.isfinite(df)
        out[p] = dict(all=float(np.nanmean(x)), all_ci=boot_ci(x[np.isfinite(x)]), flat=float(np.nanmean(xf)), flat_ci=boot_ci(xf[ok]),
                      diff_flat=float(np.nanmean(df)), diff_ci=boot_ci(df[okd]), lower=int((df[okd] < 0).sum()), n_flat=int(okd.sum()))
        o = out[p]
        L.append('| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %+.3f [%+.3f, %+.3f] | %d of %d |' % (p, o['all'], *o['all_ci'], o['flat'], *o['flat_ci'], o['diff_flat'], *o['diff_ci'], o['lower'], o['n_flat']))
    L += ['', 'Read: a spread picker with a lower on-plane share than sd on flat frames keeps more off-plane points; whether that lowers the fail rate is the R20 question for the DEV/FRESH rows (needs a rerun with the new field, no rows carry it yet).']
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump(dict(n=n, n_flat=int(flat.sum()), K=a.K, pickers=out), open(a.out[:-3] + '.json', 'w'), indent=1)
    print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
