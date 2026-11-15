"""C43: second-method recompute of the picker x K interaction on FRESH-1 flat (SUB2, dispatch 2026-11-15 16:05).

  python tools/c43_k_check.py [--boot 20000] [--seed 7] [--out results/audit/c43_k_interaction.md]

Independent of tools/k_interaction.py: no kpbench.gate / class_bootstrap / stats imports. Reads the raw row files
(results/runs/<id>/<flight>.json.gz), the label json and splits_fresh.json directly, joins the four arms by query id (dict,
not row order), fail = err_cm None or > 25 or rot_deg > 2, flat = plane_frac >= 0.85. Exact two-sided sign-flip p from the
distribution of sum(+-d_f) built by convolution; flight-cluster CI by multinomial flight weights. Expected (14 Nov):
K100 cube0.5/dpp0.5 95/62, K400 57/64 of 2075, interaction -40 = -1.93 pt, CI [-3.35,-0.66], 7/0/6, p 0.0156.
"""
import argparse, gzip, json, os, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RUNS = dict(c100='raco_cube0.5_K100_same_lg_g-cube0.5-fresh', i100='raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh',
            c400='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', i400='raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400')
EXPECT = dict(n=2075, c100=95, i100=62, c400=57, i400=64, inter=-40, lo=-3.35, hi=-0.66, neg=7, pos=0, p=0.0156)


def fails(run, flight):
    """{query id: 0/1 fail} of one flight from the raw row file"""
    d = json.load(gzip.open(os.path.join(ROOT, 'results', 'runs', run, flight.replace('/', '_') + '.json.gz'), 'rt'))
    return {int(r[1]): int(r[2] is None or r[2] > 25 or r[3] > 2) for r in d['rows']}


def exact_p(d):
    """exact two-sided sign-flip p: P(|sum s_f d_f| >= |sum d_f|) over the 2^m sign patterns, by convolution of counts"""
    dist = Counter({0: 1})
    for x in (abs(int(v)) for v in d if v):
        nd = Counter()
        for s, c in dist.items():
            nd[s + x] += c; nd[s - x] += c
        dist = nd
    tot = sum(dist.values()); s0 = abs(int(sum(d)))
    return sum(c for s, c in dist.items() if abs(s) >= s0) / tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=20000); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'audit', 'c43_k_interaction.md')); a = ap.parse_args()
    flights = json.load(open(os.path.join(ROOT, 'splits_fresh.json')))['fresh']; T = []
    for f in flights:
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))
        F = {k: fails(r, f) for k, r in RUNS.items()}
        qs = set(F['c100']); assert all(set(v) == qs for v in F.values()), f
        flat = [q for q in qs if (lab.get(str(q), {}).get('plane_frac') or -1) >= 0.85]
        c = {k: sum(F[k][q] for q in flat) for k in RUNS}
        if flat:
            T.append((f, len(flat), c['c400'], c['i400'], c['c100'], c['i100'], (c['c400'] - c['i400']) - (c['c100'] - c['i100'])))
    A = np.array([t[1:] for t in T]); n, d = A[:, 0], A[:, 5]; N = int(n.sum())
    tot = dict(c400=int(A[:, 1].sum()), i400=int(A[:, 2].sum()), c100=int(A[:, 3].sum()), i100=int(A[:, 4].sum()))
    w = np.random.default_rng(a.seed).multinomial(len(T), np.full(len(T), 1 / len(T)), a.boot)
    bs = 100 * (w @ d) / (w @ n); lo, hi = np.percentile(bs, [2.5, 97.5])
    got = dict(n=N, **tot, inter=int(d.sum()), lo=round(float(lo), 2), hi=round(float(hi), 2), neg=int((d < 0).sum()),
               pos=int((d > 0).sum()), p=round(exact_p(d), 4))
    chk = []
    for k, e in EXPECT.items():
        tol = 0.15 if k in ('lo', 'hi') else 0.0005 if k == 'p' else 0
        chk.append((k, e, got[k], abs(got[k] - e) <= tol))
    verdict = 'PASS' if all(c[3] for c in chk) else 'FAIL'
    L = ['# C43 K interaction, second method (%s): %s' % (time.strftime('%Y-%m-%d %H:%M'), verdict), '',
         'Script tools/c43_k_check.py (raw gz rows + label json, join by query id, convolution sign-flip, multinomial-weight bootstrap '
         '%d, seed %d; CI tolerance 0.15 pt, counts exact). First method: tools/k_interaction.py -> results/main/k_interaction_fresh1.md.' % (a.boot, a.seed), '',
         '| quantity | expected (14 Nov) | recomputed | match |', '|---|---|---|---|']
    L += ['| %s | %s | %s | %s |' % (k, e, g, 'yes' if ok else 'NO') for k, e, g, ok in chk]
    L += ['', 'Interaction %+d of %d = %+.2f pt; cube0.5 - dpp0.5 at K100 %+d, at K400 %+d.' % (got['inter'], N, 100 * got['inter'] / N,
          tot['c100'] - tot['i100'], tot['c400'] - tot['i400']), '',
          '| flight | n flat | K400 c / i | K100 c / i | d |', '|---|---|---|---|---|']
    L += ['| %s | %d | %d / %d | %d / %d | %+d |' % t for t in T]
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L[:20])); print('->', a.out)


if __name__ == '__main__':
    main()
