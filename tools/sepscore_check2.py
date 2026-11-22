"""Second-script check of kpbench.sepscore on synthetic dumps (SUB2, dispatch 2026-11-22 15:25). No real picks, no sep read.

  python tools/sepscore_check2.py [--out results/audit/sepscore_second_script_2026-11-22.md]

The four rules are re-coded here from the rule text (sep_rule_prereg + addendum), reading the dump files directly, with other
tools than the scorer: scikit-learn logistic regression / AUROC / average precision (scorer: own Newton fit and rank
formulas), own block bootstrap with another seed, binomial tails from math.comb (scorer: scipy binomtest). Only the dump
WRITER of tests/test_sepscore.py is shared (12 fake flights x 100 queries, q < 80 flat). Sets:
  planted : off drives failure and the uniform arm has more off -> all four rules must pass in both scripts;
  null    : off is noise -> all four rules must fail in both;
  tie     : planted, but in flight 0 every arm has off = 0.25 on every query -> that flight's sep_f is exactly 0 (dropped by
            rule 4, 11 flights left: 10 of 11 needed, p 0.0059; 9 of 11 = 0.033 would also pass) and its one-fails pairs are
            ties (dropped by rule 3).
Agreement, fixed before the run: same pass / fail on every rule of every set; counts (n, wins, ties, flights neg / pos / zero)
identical; p values within 1e-9; dAUROC and dPR-AUC within 0.002; bootstrap z within 10 %.
Toy rule 4: 12 flights, 10 with failed frames more on-plane than posed ones -> p = P(X >= 10 | 12, 1/2) = 79 / 4096 = 0.019.
"""
import argparse, gzip, json, math, os, sys, tempfile
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tests'))

ARMS = ['uni', 'strong', 'other']; BASE = (0.30, 0.10, 0.20)


def fname(f):
    return f.replace('/', '_') + '.json.gz'


def read(root, rid, flights, flat_of):
    """-> {(flight, q): (fail, off or None, n_inl, flat)} straight from the files"""
    T = {}
    for f in flights:
        j = json.load(gzip.open(os.path.join(root, 'runs', rid, fname(f)), 'rt'))
        for r in j['rows']:
            q = int(r[1]); e = j['onplane'].get(str(q)); off = None if not e or e[0] is None else 1.0 - e[0]
            T[(f, q)] = (r[2] is None or r[2] > 25 or r[3] > 2, off, float(r[6] or 0), flat_of(q))
    return T


def tail_ge(k, m):
    return sum(math.comb(m, i) for i in range(k, m + 1)) / 2 ** m if m else 1.0


def two_sided(k, m):
    """exact two-sided binomial p at 1/2: outcomes no more likely than the observed one"""
    return min(1.0, sum(math.comb(m, i) for i in range(m + 1) if math.comb(m, i) <= math.comb(m, k)) / 2 ** m) if m else None


def rule1(T, keys):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    fl = np.array([k[0] for k in keys] * 3); y = np.array([t[k][0] for t in T for k in keys], int)
    X = np.array([[math.log1p(t[k][2]), t[k][1]] for t in T for k in keys])
    out = {}
    for name, cols in (('count', [0]), ('both', [0, 1])):
        p = np.zeros(len(y))
        for f in np.unique(fl):
            tr = fl != f; A = X[:, cols]; mu = A[tr].mean(0); sd = A[tr].std(0) + 1e-12
            p[~tr] = LogisticRegression(C=1e6, max_iter=1000).fit((A[tr] - mu) / sd, y[tr]).predict_proba((A[~tr] - mu) / sd)[:, 1]
        out[name] = (roc_auc_score(y, p), average_precision_score(y, p))
    da, dp = out['both'][0] - out['count'][0], out['both'][1] - out['count'][1]
    return dict(d_auroc=da, d_prauc=dp, ok=round(da, 4) >= 0.01 and round(dp, 4) >= 0.01)


def rule2(T, keys, seed=12345, n=4000):
    d = np.array([T[0][k][1] - T[1][k][1] for k in keys]); B = {}
    for k, v in zip(keys, d):
        B.setdefault((k[0], k[1] // 20), []).append(v)
    s = np.array([sum(v) for v in B.values()]); c = np.array([len(v) for v in B.values()], float)
    pick = np.random.default_rng(seed).integers(0, len(s), (n, len(s))); se = float(np.std(s[pick].sum(1) / c[pick].sum(1)))
    z = float(d.mean() / se) if se > 0 else 0.0
    return dict(d_mean=float(d.mean()), z=z, ok=round(z, 2) >= 2.5)


def rule3(T, keys):
    res = {}
    for name, B in (('strong', T[1]), ('other', T[2])):
        n = win = tie = 0
        for k in keys:
            fa, xa, _, flat = T[0][k]; fb, xb = B[k][0], B[k][1]
            if flat and fa != fb:
                good, bad = (xb, xa) if fa else (xa, xb); n += 1; win += good > bad; tie += good == bad
        m = n - tie; res[name] = dict(n=n, win=int(win), tie=int(tie), share=round(win / m, 4) if m else None, p=two_sided(int(win), m))
    return dict(pairs=res, ok=all(v['share'] is not None and v['share'] >= 0.70 and v['p'] < 0.05 for v in res.values()))


def rule4(T, keys):
    S = {}
    for k in keys:
        bad, off, _, flat = T[0][k]
        if flat:
            S.setdefault(k[0], {True: [], False: []})[bool(bad)].append(off)
    sep = [sum(v[True]) / len(v[True]) - sum(v[False]) / len(v[False]) for v in S.values() if len(v[True]) >= 5 and len(v[False]) >= 5]
    neg = sum(x < 0 for x in sep); pos = sum(x > 0 for x in sep); p = tail_ge(neg, neg + pos)
    return dict(flights=len(sep), neg=neg, pos=pos, zero=len(sep) - neg - pos, p=p, ok=neg + pos >= 5 and p < 0.05)


def make(root, kind):
    import test_sepscore as ts                                     # dump writer only
    for k, (a, b) in enumerate(zip(ARMS, BASE)):
        ts.write_arm(root, a, b, kind != 'null', seed=k + 1)
    if kind == 'tie':
        for a in ARMS:
            p = os.path.join(root, 'runs', a, fname(ts.FLIGHTS[0])); j = json.load(gzip.open(p, 'rt'))
            for e in j['onplane'].values():
                e[0] = 0.75
            with gzip.open(p, 'wt') as fh:
                json.dump(j, fh)
    return ts.FLIGHTS


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); a = ap.parse_args(argv)
    from kpbench import sepscore, gate
    flat_of = lambda q: q < 80; lab = {str(q): dict(plane_frac=0.9 if q < 80 else 0.5) for q in range(100)}
    L = ['# kpbench.sepscore vs an independent second script, synthetic dumps only (SUB2, tools/sepscore_check2.py)', '',
         'No real picks were read and no sep value of any real run exists. 12 fake flights x 100 queries (80 flat) per arm. '
         'Cells: scorer / second script.', '',
         '| set | rule 1 dAUROC, dPR-AUC, pass | rule 2 d, z, pass | rule 3 vs strong n-win-tie p; vs other; pass | rule 4 neg/pos/zero p pass | all four | agree |',
         '|---|---|---|---|---|---|---|']
    allok = True; y = lambda b: 'yes' if b else 'no'
    for kind in ('planted', 'null', 'tie'):
        with tempfile.TemporaryDirectory() as root:
            flights = make(root, kind); gate.RESULTS = sepscore.RESULTS = root; sepscore.labels = lambda f: lab
            Ts, ks, miss = sepscore.validate(ARMS, flights); r = sepscore.score(Ts, ks, ARMS)
            T = [read(root, x, flights, flat_of) for x in ARMS]; keys = [k for k in sorted(T[0]) if all(t[k][1] is not None for t in T)]
        m1, m2, m3, m4 = rule1(T, keys), rule2(T, keys), rule3(T, keys), rule4(T, keys); q = r['flight']
        w = [r['within']['strong'], r['within']['other']]; v = [m3['pairs']['strong'], m3['pairs']['other']]
        ok = (len(keys) == r['n'] and miss == 0
              and (m1['ok'], m2['ok'], m3['ok'], m4['ok']) == (r['rule1'], r['rule2'], r['rule3'], r['rule4'])
              and abs(m1['d_auroc'] - r['d_auroc']) <= 0.002 and abs(m1['d_prauc'] - r['d_prauc']) <= 0.002
              and abs(m2['d_mean'] - r['d_mean']) <= 1e-4 and abs(m2['z'] - r['z']) <= 0.1 * max(abs(r['z']), 1.0)
              and all((a_['n'], a_['win'], a_['tie'], a_['share']) == (b_['n'], b_['win'], b_['tie'], b_['share']) and abs(a_['p'] - b_['p']) <= 1e-9
                      for a_, b_ in zip(w, v))
              and (q['flights'], q['neg'], q['pos'], q['zero']) == (m4['flights'], m4['neg'], m4['pos'], m4['zero']) and abs(q['p'] - m4['p']) <= 1e-4)
        allok = allok and ok; p4 = bool(m1['ok'] and m2['ok'] and m3['ok'] and m4['ok'])
        f3 = lambda x: '%d-%d-%d %.2g' % (x['n'], x['win'], x['tie'], x['p'])
        L.append('| %s | %+.4f, %+.4f, %s / %+.4f, %+.4f, %s | %+.4f, %.2f, %s / %+.4f, %.2f, %s | %s; %s; %s / %s; %s; %s | %d/%d/%d %.4f %s / %d/%d/%d %.4f %s | %s / %s | %s |'
                 % (kind, r['d_auroc'], r['d_prauc'], y(r['rule1']), m1['d_auroc'], m1['d_prauc'], y(m1['ok']),
                    r['d_mean'], r['z'], y(r['rule2']), m2['d_mean'], m2['z'], y(m2['ok']),
                    f3(w[0]), f3(w[1]), y(r['rule3']), f3(v[0]), f3(v[1]), y(m3['ok']),
                    q['neg'], q['pos'], q['zero'], q['p'], y(r['rule4']), m4['neg'], m4['pos'], m4['zero'], m4['p'], y(m4['ok']),
                    y(r['passed']), y(p4), 'yes' if ok else 'NO'))
    L += ['', 'Agreement rule (fixed before the run): same pass / fail on every rule; counts identical; p within 1e-9 (rule 4: 1e-4, the scorer '
          'rounds it); dAUROC and dPR-AUC within 0.002; z within 10 %.', 'Sign-test thresholds by flights left after dropped zeros (one-sided p < 0.05): '
          + ', '.join('%d of %d' % (next(k for k in range(m + 1) if tail_ge(k, m) < 0.05), m) for m in range(5, 13)) + '.',
          '', '**%s**' % ('PASS: the second script agrees with kpbench.sepscore on all rules of all three sets' if allok else
                         'FAIL: disagreement -> no sep read until resolved')]
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)
    return allok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
