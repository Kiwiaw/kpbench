"""SUB1 pre-registration v2 scorer (results/sub1/prereg.md): all three DEV rules for every candidate metric, one table.

  python -m kpbench.sub1_prereg --split dev        -> results/sub1/<split>_prereg.md + .json
Inputs: results/eval/<split>/raco_<picker>_K100_lg_m.json.gz (euler/sub1.sbatch, Q187: sd, cube0.5, dpp0.5).
Rule 1 adds over inlier count: kpbench.sub1_ferranti pooled leave-one-flight-out dAUROC >= 0.01 and dPR-AUC >= 0.01.
Rule 2 uniform lowers error: d_q = dir x (m(uniform) - m(strong)) per query, no-pose fill as sub1_ferranti, 20-frame block
  bootstrap z >= 2.5.
Rule 3 picker-level link: flat frames where exactly one of uniform / strong fails (and uniform / other): the posed picker
  has the better oriented value in >= 70 % (ties dropped), sign-test p < 0.05, in both pairs.
A new metric (cells, ent, offplane, sel_logdet, sel_hull) must also beat reproj_med and rep5 on rule-1 dAUROC. Primary = largest rule-2 z among
the non-baseline metrics passing all rules (reproj_med and rep5 are baselines: reported, never primary). Columns missing from the rows are reported as 'missing', never as a pass or a kill.
"""
import os, json, gzip
import numpy as np
from scipy.stats import binomtest
from .cache import REPO
from .stats import fail_flags
from .sub1 import OUT, blocks, boot_se
from . import sub1_ferranti as fe

DIR = dict(hull=1, reproj_med=-1, rep5=1, crlb=-1, inl_ratio=1, cells=1, ent=1, offplane=1, sel_logdet=1, sel_hull=1)   # prereg v2 table, '+' = higher is better
NEW = ('cells', 'ent', 'offplane', 'sel_logdet', 'sel_hull')   # sel_*: pre-solve, scout E3 (prereg addendum 2026-10-20)
BASE = ('reproj_med', 'rep5')


def load(split, p):
    """rows sorted by (flight, q), their columns and flat flags"""
    d = json.load(gzip.open(os.path.join(REPO, 'results', 'eval', split, 'raco_%s_K100_lg_m.json.gz' % p), 'rt'))
    o = sorted(range(len(d['rows'])), key=lambda i: (str(d['rows'][i][0]), d['rows'][i][1]))
    return [d['rows'][i] for i in o], d['cols'], np.array(d['flat'], bool)[o]


def col(rows, cols, m, fill):
    x = np.array([np.nan if r[cols.index(m)] is None else r[cols.index(m)] for r in rows], float)
    x[np.isnan(x)] = fill
    return x


def within(fa, fb, xa, xb, flat):
    """flat frames where exactly one side fails: n, wins of the posed side (higher oriented value), ties, sign-test p"""
    one = flat & (fa != fb); good = np.where(fa[one], xb[one], xa[one]); bad = np.where(fa[one], xa[one], xb[one])
    n = int(one.sum()); win = int((good > bad).sum()); tie = int((good == bad).sum()); m = n - tie
    return dict(n=n, win=win, tie=tie, share=round(win / m, 4) if m else None, p=float(binomtest(win, m).pvalue) if m else None)


def analyse(split, uniform='dpp0.5', strong='sd', other='cube0.5'):
    D = {p: load(split, p) for p in (uniform, strong, other)}
    keys = [[(str(r[0]), r[1]) for r in D[p][0]] for p in D]
    assert all(k == keys[0] for k in keys), 'pickers must share the query set'
    ru, cu, flat = D[uniform]; inv = blocks(ru); F = {p: fail_flags(D[p][0]) for p in D}
    res = {}
    for m in DIR:
        if not all(m in D[p][1] for p in D):
            res[m] = dict(status='missing'); continue
        r1 = fe.analyse(split, m)['results']['pooled'].get('delta', {})
        allx = np.concatenate([col(D[p][0], D[p][1], m, np.nan) for p in D])
        fill = float(np.nanpercentile(allx, 99)) if m == 'crlb' else fe.FILL[m]
        X = {p: DIR[m] * col(D[p][0], D[p][1], m, fill) for p in D}
        d = X[uniform] - X[strong]; z = float(d.mean() / boot_se(d, inv))
        w = {q: within(F[uniform], F[q], X[uniform], X[q], flat) for q in (strong, other)}
        ok1 = bool(r1) and r1['auroc'] >= 0.01 and r1['prauc'] >= 0.01
        ok3 = all(v['share'] is not None and v['share'] >= 0.70 and v['p'] < 0.05 for v in w.values())
        res[m] = dict(status='scored', d_auroc=r1.get('auroc'), d_prauc=r1.get('prauc'), rule1=ok1, d_mean=round(float(d.mean()), 4),
                      z=round(z, 2), rule2=z >= 2.5, within=w, rule3=ok3)
    for m in NEW:
        if res[m]['status'] == 'scored':
            b = [res[k]['d_auroc'] for k in BASE if res[k]['status'] == 'scored']
            res[m]['beats_baselines'] = bool(b) and res[m]['d_auroc'] > max(b)
    ok = [m for m, r in res.items() if r['status'] == 'scored' and r['rule1'] and r['rule2'] and r['rule3'] and r.get('beats_baselines', True)]
    cand = [m for m in ok if m not in BASE]; prim = max(cand, key=lambda m: res[m]['z']) if cand else None
    return dict(split=split, uniform=uniform, strong=strong, other=other, n=len(ru), n_flat=int(flat.sum()), results=res, passing=ok,
                primary=prim, verdict=('primary metric %s (z %.2f) -> one held-out shot' % (prim, res[prim]['z'])) if prim else
                'no candidate passes all three rules -> SUB1 stays query-level (prereg kill)')


def write(out):
    os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, out['split'] + '_prereg')
    json.dump(out, open(base + '.json', 'w'), indent=1)
    u, s, o = out['uniform'], out['strong'], out['other']
    L = ['# SUB1 prereg v2 scorer on %s (%d queries, %d flat; uniform %s vs strong %s, rule 3 also vs %s)' % (out['split'], out['n'], out['n_flat'], u, s, o), '',
         '| metric | dAUROC | dPR-AUC | rule 1 | d (%s - %s) | z | rule 2 | within vs %s n / share / p | within vs %s n / share / p | rule 3 |' % (u, s, s, o),
         '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda w: '%d / %s / %s' % (w['n'], w['share'], 'nan' if w['p'] is None else '%.2g' % w['p'])
    for m, r in out['results'].items():
        if r['status'] != 'scored':
            L.append('| %s | missing | | | | | | | | |' % m); continue
        L.append('| %s | %+.4f | %+.4f | %s | %+.4f | %.2f | %s | %s | %s | %s |' % (m, r['d_auroc'], r['d_prauc'], 'yes' if r['rule1'] else 'no', r['d_mean'], r['z'],
                 'yes' if r['rule2'] else 'no', f(r['within'][s]), f(r['within'][o]), 'yes' if r['rule3'] else 'no'))
    L += ['', '**%s**' % out['verdict']]
    open(base + '.md', 'w').write('\n'.join(L) + '\n')
    return base + '.md'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev')
    ap.add_argument('--uniform', default='dpp0.5'); ap.add_argument('--strong', default='sd'); ap.add_argument('--other', default='cube0.5')
    a = ap.parse_args(argv); out = analyse(a.split, a.uniform, a.strong, a.other); print(open(write(out)).read())


if __name__ == '__main__':
    main()
