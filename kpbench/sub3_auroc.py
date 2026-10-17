"""SUB3 gate for a repeatability-aware picker (T4): does per-query repeatability predict pose failure?

  python -m kpbench.sub3_auroc [--pickers dpp0.5,cube0.5] [--splits micro,dev] [--out results/sub3/rep5_auroc.md]

Reads the stored leaderboard rows results/runs/raco_<picker>_K100_same_lg/<flight>.json.gz (no GPU, no flight data).
AUROC = P(rep5 of a failed query < rep5 of a posed query), ties count 1/2 (0.5 = no signal; > 0.5 = low repeatability goes with
failure). 95 % CI: bootstrap over 20-query blocks (kpbench.stats BLOCK), 2000 resamples. Flat = plane_frac >= FLAT.
Gate (dispatch 2026-10-17): build the dppr picker only if AUROC >= 0.6 (lower bound on the frames that matter).
"""
import os, json, gzip
import numpy as np
from .cache import REPO
from .splits import load as load_splits, FLAT
from .stats import fail_flags, BLOCK
from .bench import COLS
from .data import slug


def auroc(x, y):
    """x = score (low = predicted fail), y = 1 for fail; AUROC of -x for y via ranks (ties 1/2)"""
    from scipy.stats import rankdata
    r = rankdata(-x); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')


def rows_for(picker, flights):
    out = []
    for fid in flights:
        fn = os.path.join(REPO, 'results', 'runs', 'raco_%s_K100_same_lg' % picker, slug(fid) + '.json.gz')
        lab = json.load(open(os.path.join(REPO, 'results', 'labels', slug(fid) + '.json')))
        for r in json.load(gzip.open(fn, 'rt'))['rows']:
            pf = lab[str(r[1])].get('plane_frac')
            out.append((r, pf is not None and pf >= FLAT))
    return out


def gate(picker, flights, flat_only, nboot=2000, seed=0):
    rr = [(r, fl) for r, fl in rows_for(picker, flights) if (fl or not flat_only) and r[COLS.index('rep5')] is not None]
    rows = [r for r, _ in rr]; y = fail_flags(rows); x = np.array([r[COLS.index('rep5')] for r in rows], float)
    blk = np.array(['%s:%d' % (r[0], r[1] // BLOCK) for r in rows]); ub, inv = np.unique(blk, return_inverse=True)
    idx = [np.flatnonzero(inv == b) for b in range(len(ub))]; rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        j = np.concatenate([idx[b] for b in rng.integers(0, len(ub), len(ub))]); bs.append(auroc(x[j], y[j]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=len(rows), fails=int(y.sum()), auroc=auroc(x, y), lo=lo, hi=hi,
                rep5_fail=float(x[y == 1].mean()) if y.any() else None, rep5_ok=float(x[y == 0].mean()))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickers', default='dpp0.5,cube0.5,sd'); ap.add_argument('--splits', default='micro,dev')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'sub3', 'rep5_auroc.md'))
    a = ap.parse_args(argv); sp = load_splits()
    L = ['# SUB3 gate: AUROC of per-query rep5 for pose failure (kpbench.sub3_auroc)', '',
         'Stored K = 100 LightGlue rows; AUROC = P(rep5 of a failed query < rep5 of a posed one); 95 % CI = 20-query block bootstrap.', '',
         '| split | frames | picker | queries | fails | rep5 fail / posed | AUROC [95 %] |', '|---|---|---|---|---|---|---|']
    for s in a.splits.split(','):
        fl = list(sp[s]) if s == 'micro' else sp[s]
        for flat_only in (True, False):
            for p in a.pickers.split(','):
                g = gate(p, fl, flat_only)
                L.append('| %s | %s | %s | %d | %d | %.3f / %.3f | %.3f [%.3f, %.3f] |' % (
                    s, 'flat' if flat_only else 'all', p, g['n'], g['fails'], g['rep5_fail'], g['rep5_ok'], g['auroc'], g['lo'], g['hi']))
                print(L[-1])
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
