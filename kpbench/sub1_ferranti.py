"""SUB1 baseline (Ferranti et al., ICPR 2020): does hull add to the inlier count as a predictor of PnP failure?

  python -m kpbench.sub1_ferranti --split dev     -> results/sub1/<split>_ferranti.md + .json
Inputs: results/eval/<split>/raco_<picker>_K100_lg_m.json.gz (every picker present; euler/sub1.sbatch writes them).
Per picker and pooled: logistic fail ~ log(1 + n_inl), fail ~ hull, fail ~ log(1 + n_inl) + hull, each fitted
leave-one-flight-out (predictions on a flight come from a fit without it), scored by AUROC and PR-AUC (average
precision, fail = positive). delta = count+hull minus count alone. Kill rule (dispatch 2026-10-15): delta < 0.01 in
both AUROC and PR-AUC on DEV => hull is Ferranti's coverage restated, SUB1 moves to metric #2 (crlb).
No-pose queries: n_inl 0 and hull 0 (as kpbench.sub1 NOPOSE).
--metric m tests another pre-registered column the same way (crlb, inl_ratio, reproj_med, depth_iqr; no-pose fill as
docs/sub1_metrics.md: crlb = pooled 99th percentile, depth_iqr = 0; prereg v2 results/sub1/prereg.md adds rep5, cells, ent,
offplane, no-pose fill 0). --posed keeps only queries with a pose (fail = > 25 cm or
> 2 deg), so the no-pose fill cannot carry the signal. Output <split>_ferranti[_<m>][_posed].md + .json.
"""
import os, json, gzip, glob
import numpy as np
from .cache import REPO
from .stats import fail_flags
from .sub1 import auroc, OUT, NOPOSE

MODELS = {'count': ('n_inl',), 'hull': ('hull',), 'count+hull': ('n_inl', 'hull')}
FILL = dict(NOPOSE, depth_iqr=0.0, cells=0.0, ent=0.0, offplane=0.0, rep5=0.0, sel_logdet=-6.0, sel_hull=0.0)   # prereg v2 fills (results/sub1/prereg.md)


def load(split, metric='hull', posed=False):
    """{picker: (flights, X dict, fail)} for every saved metrics file of the split; X['hull'] holds the tested metric"""
    T = {}
    for fn in sorted(glob.glob(os.path.join(REPO, 'results', 'eval', split, 'raco_*_K100_lg_m.json.gz'))):
        p = os.path.basename(fn)[len('raco_'):-len('_K100_lg_m.json.gz')]
        d = json.load(gzip.open(fn, 'rt')); c = d['cols']; rows = d['rows']
        if posed:
            rows = [r for r in rows if r[2] is not None]
        col = lambda m: np.array([np.nan if r[c.index(m)] is None else r[c.index(m)] for r in rows], float)
        X = dict(n_inl=np.log1p(np.nan_to_num(col('n_inl'))), hull=col(metric))
        T[p] = (np.array([str(r[0]) for r in rows]), X, fail_flags(rows))
    cap = np.nanpercentile(np.concatenate([X['hull'] for _, X, _ in T.values()]), 99) if metric == 'crlb' else FILL[metric]
    for _, X, _ in T.values():
        X['hull'][np.isnan(X['hull'])] = cap
    return T


def logit_fit(X, y, l2=1e-3, it=50):
    """Newton (IRLS) logistic regression with intercept; X standardised by the caller"""
    A = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w)); g = A.T @ (p - y) + l2 * w
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(len(w)); step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def lofo(fl, X, y):
    """leave-one-flight-out predicted fail probability"""
    out = np.zeros(len(y))
    for f in np.unique(fl):
        tr = fl != f; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        w = logit_fit((X[tr] - mu) / sd, y[tr]); A = np.hstack([np.ones(((~tr).sum(), 1)), (X[~tr] - mu) / sd])
        out[~tr] = 1 / (1 + np.exp(-A @ w))
    return out


def avg_prec(score, y):
    """average precision, y = 1 positive, higher score = more likely positive"""
    o = np.argsort(-score, kind='stable'); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1))[y == 1].mean()) if y.sum() else None


def score(fl, X, y):
    r = {}
    for name, cols in MODELS.items():
        s = lofo(fl, np.stack([X[c] for c in cols], 1), y)
        a = auroc(-s, y)                                         # sub1.auroc: P(success scores higher); -s = success score
        r[name] = dict(auroc=None if a is None else round(a, 4), prauc=None if a is None else round(avg_prec(s, y), 4))
    if r['count']['auroc'] is not None:
        r['delta'] = dict(auroc=round(r['count+hull']['auroc'] - r['count']['auroc'], 4),
                          prauc=round(r['count+hull']['prauc'] - r['count']['prauc'], 4))
    return r


def analyse(split, metric='hull', posed=False):
    T = load(split, metric, posed); res = {}
    for p, (fl, X, y) in T.items():
        res[p] = dict(n=len(y), fails=int(y.sum()), **score(fl, X, y))
    fl = np.concatenate([T[p][0] for p in T]); y = np.concatenate([T[p][2] for p in T])
    X = {m: np.concatenate([T[p][1][m] for p in T]) for m in ('n_inl', 'hull')}
    res['pooled'] = dict(n=len(y), fails=int(y.sum()), **score(fl, X, y))
    d = res['pooled'].get('delta', {})
    kill = bool(d) and d['auroc'] < 0.01 and d['prauc'] < 0.01
    return dict(split=split, metric=metric, posed=posed, pickers=list(T), results=res,
                verdict=('KILL: %s adds < 0.01 AUROC and PR-AUC over inlier count' % metric if kill else
                         '%s adds over inlier count: dAUROC %+.4f, dPR-AUC %+.4f' % (metric, d.get('auroc', 0), d.get('prauc', 0))))


def write(out):
    os.makedirs(OUT, exist_ok=True); m = out.get('metric', 'hull')
    base = os.path.join(OUT, out['split'] + '_ferranti' + ('' if m == 'hull' else '_' + m) + ('_posed' if out.get('posed') else ''))
    json.dump(out, open(base + '.json', 'w'), indent=1)
    L = ['# SUB1 Ferranti baseline on %s (logistic, leave-one-flight-out; fail = positive for PR-AUC)%s' % (
        out['split'], '' if m == 'hull' else '; metric = %s (column "hull" below)' % m) + (', posed queries only' if out.get('posed') else ''), '',
         '| picker | queries | fails | AUROC count | AUROC hull | AUROC count+hull | dAUROC | PR-AUC count | PR-AUC hull | PR-AUC count+hull | dPR-AUC |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for p, r in out['results'].items():
        g = lambda m, k: r[m][k] if m in r else None
        L.append('| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            p, r['n'], r['fails'], g('count', 'auroc'), g('hull', 'auroc'), g('count+hull', 'auroc'), g('delta', 'auroc'),
            g('count', 'prauc'), g('hull', 'prauc'), g('count+hull', 'prauc'), g('delta', 'prauc')))
    L += ['', '**%s**' % out['verdict']]
    open(base + '.md', 'w').write('\n'.join(L) + '\n')
    return base + '.md'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev')
    ap.add_argument('--metric', default='hull', choices=['hull', 'crlb', 'inl_ratio', 'reproj_med', 'depth_iqr', 'rep5', 'cells', 'ent', 'offplane', 'sel_logdet', 'sel_hull'])
    ap.add_argument('--posed', action='store_true'); a = ap.parse_args(argv)
    out = analyse(a.split, a.metric, a.posed); print(write(out)); print(out['verdict'])


if __name__ == '__main__':
    main()
