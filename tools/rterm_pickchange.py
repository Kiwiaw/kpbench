"""SUB3 mechanism check on the cached MICRO candidates (no images, poses or GPU): how many K100 picks does the r-term change?

  python tools/rterm_pickchange.py [--K 100] [--out results/sub3/rterm_pickchange_micro_<date>.md]

Pairs (r-term picker vs its sigma-matched plain DPP): dppr0.5a1b2t0.8 vs dpp0.5 (= dpp0.5a1) and dppr0.317a1.6b2.01t0.9 vs
dpp0.317a1.6. Per frame: change = 1 - |picks_dppr & picks_dpp| / K. Pre-registered prediction (dispatch 2026-11-01 15:20):
mean change at sigma 0.317 >= 2 x mean change at sigma 0.5 (the FRESH rep5 gain ratio +0.0113 / +0.0034). Kill: change at
sigma 0.5 >= change at 0.317. Sanity: beta 0 (dppr..b0..) reproduces the plain DPP picks on every frame.
Same picks -> same PnP, so pick change bounds what the r-term can move; it does not say which way.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench import pickers
from kpbench.cache import Cands

ROOT = os.path.join(os.path.dirname(__file__), '..')
PAIRS = [('dppr0.5a1b2t0.8', 'dpp0.5', 'dppr0.5a1b0t0.8'), ('dppr0.317a1.6b2.01t0.9', 'dpp0.317a1.6', 'dppr0.317a1.6b0t0.9')]


class Fl:
    def __init__(self, fid):
        self.slug = fid.replace('/', '_')


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'rterm_pickchange_micro_%s.md' % time.strftime('%Y-%m-%d')))
    a = ap.parse_args(); K = a.K
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    names = sorted({p for pr in PAIRS for p in pr})
    fns = {p: pickers.parse(p) for p in names}
    change = {pr[0]: [] for pr in PAIRS}; flight = {pr[0]: [] for pr in PAIRS}; b0_bad = {pr[0]: 0 for pr in PAIRS}; T = dict.fromkeys(names, 0.0); n = 0
    for fid, fr in micro.items():
        F = Fl(fid); C = Cands(F.slug, a.det)
        for q in fr:
            u, v, s, z, de = C.frame(q); sel = {}
            for p, fn in fns.items():
                kw = {}
                if getattr(fn, 'needs_past', False): kw['ctx'] = (F, C, q)
                if getattr(fn, 'needs_desc', False): kw['de'] = de
                t0 = time.time(); sel[p] = np.sort(np.asarray(fn(u, v, s, z, K, **kw), int)); T[p] += time.time() - t0
            n += 1
            for r, d, b0 in PAIRS:
                k = max(len(sel[d]), 1)
                change[r].append(1.0 - len(np.intersect1d(sel[r], sel[d])) / k); flight[r].append(fid)
                b0_bad[r] += not np.array_equal(sel[b0], sel[d])
    rows = {}
    for r, d, b0 in PAIRS:
        c = np.array(change[r]); fl = np.array(flight[r])
        rows[r] = dict(dpp=d, mean=float(c.mean()), ci=boot_ci(c), median=float(np.median(c)), any=int((c > 0).sum()),
                       per_flight={f: float(c[fl == f].mean()) for f in micro}, rstd=float(np.mean(fns[r].rstd)) if fns[r].rstd else None,
                       b0_bad=b0_bad[r], secs=T[r], secs_dpp=T[d])
    r5, r3 = rows['dppr0.5a1b2t0.8'], rows['dppr0.317a1.6b2.01t0.9']
    ratio = r3['mean'] / max(r5['mean'], 1e-9)
    diff = np.array(change['dppr0.317a1.6b2.01t0.9']) - np.array(change['dppr0.5a1b2t0.8']); dci = boot_ci(diff)
    pred = ratio >= 2.0; kill = r5['mean'] >= r3['mean']
    L = ['# SUB3 r-term pick change on MICRO (%s, K %d)' % (time.strftime('%Y-%m-%d'), K), '',
         'Source: tools/rterm_pickchange.py on cache/*_%s.npz, %d MICRO flat frames (%s). No poses on the laptop: pick change only, no fail %% or rep5.' % (a.det, n, ', '.join(micro)),
         'change = 1 - |picks(dppr) & picks(dpp)| / K per frame, dppr vs its sigma-matched plain DPP. Bootstrap CI over frames (frames of one flight are not independent, so read the per-flight means too).', '',
         '| r-term picker | plain DPP | mean change | 95 % CI | median | frames changed | per flight | r std | beta 0 = dpp (frames differing) | s dppr / dpp |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in (r5, r3):
        L.append('| %s | %s | %.3f | [%.3f, %.3f] | %.3f | %d / %d | %s | %s | %d | %.1f / %.1f |' % (
            [k for k, v in rows.items() if v is r][0], r['dpp'], r['mean'], *r['ci'], r['median'], r['any'], n,
            ', '.join('%s %.3f' % (f.split('/')[1], m) for f, m in r['per_flight'].items()), '%.3f' % r['rstd'] if r['rstd'] is not None else '-', r['b0_bad'], r['secs'], r['secs_dpp']))
    L += ['', '## Pre-registered test (dispatch 2026-11-01 15:20)', '',
          '- ratio mean change sigma 0.317 / sigma 0.5 = %.3f / %.3f = %.2f (prediction >= 2.0): %s' % (r3['mean'], r5['mean'], ratio, 'MET' if pred else 'NOT MET'),
          '- paired per-frame difference (0.317 - 0.5) = %+.3f [%+.3f, %+.3f]' % (float(diff.mean()), *dci),
          '- kill (change at 0.5 >= change at 0.317): %s' % ('MET -> sigma-saturation hypothesis dead; DEV study sweeps beta/t at fixed sigma 0.5' if kill else 'not met'),
          '- sanity beta 0 = plain DPP on every frame: %s' % ('PASS' if not any(v['b0_bad'] for v in rows.values()) else 'FAIL (%s)' % {k: v['b0_bad'] for k, v in rows.items()}), '',
          'Read: the FRESH rep5 gains are +0.0034 (sigma 0.5) and +0.0113 (sigma 0.317); if the pick change ratio is far from the gain ratio 3.3, the r-term moves picks that the',
          'repeatability score does not reward, i.e. the gain is bounded by what the kernel scale leaves free, not by how many picks the r-term can touch.']
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
