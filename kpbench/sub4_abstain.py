"""SUB4 objection (f): is low repeatability an actionable failure signal? Abstention curve + within-query test.

  python -m kpbench.sub4_abstain [--picker dpp0.5] [--others cube0.5,sd] [--out results/sub4/abstention_dev.md]

Reads the stored DEV rows results/runs/raco_<picker>_K100_same_lg (no GPU, no flight data), flat queries (plane_frac >= FLAT).
(i) Abstention: reject the lowest-scoring x % of flat queries, report fail % of the kept ones. Scores: rep5 (a MEASUREMENT with
true pose and depth, the upper reference), n_inl and n_match (test-time, Ferranti-family confidence). reproj_med needs the
--metrics rows (SUB1 job 17066289, on Euler) and is not in the stored rows.
(ii) Within query: on the picker-dependent flat frames (exactly one of picker / other fails), does the posed picker have the
higher score on that same frame? Share of frames + sign-test p (ties dropped). This is the per-frame link a query-level AUROC
(0.864) does not show.
"""
import os
import numpy as np
from scipy.stats import binomtest
from .cache import REPO
from .splits import load as load_splits
from .stats import fail_flags
from .bench import COLS
from .sub3_auroc import rows_for, auroc

SCORES = ('rep5', 'n_inl', 'n_match')
RATES = (0, 5, 10, 20, 30)


def flat_rows(picker, flights):
    """(flight id, q) -> row of the flat queries (row[0] is a per-file flight index, not unique across files)"""
    return {(fid, r[1]): r for fid in flights for r, fl in rows_for(picker, [fid]) if fl}


def abstain(rows, key, rates=RATES):
    x = np.array([np.nan if r[COLS.index(key)] is None else r[COLS.index(key)] for r in rows], float); y = fail_flags(rows)
    x = np.where(np.isnan(x), -np.inf, x); o = np.argsort(x, kind='stable'); out = []
    for p in rates:
        keep = o[int(round(len(o) * p / 100)):]
        out.append(100 * y[keep].mean())
    return out, auroc(np.where(np.isinf(x), -1, x), y)


def within(A, B, key):
    """frames in both where exactly one fails: share where the posed side has the higher key, sign-test p (ties dropped)"""
    ks = sorted(set(A) & set(B)); ra = [A[k] for k in ks]; rb = [B[k] for k in ks]
    fa, fb = fail_flags(ra), fail_flags(rb); win = tie = n = 0
    for a, b, xa, xb in zip(fa, fb, ra, rb):
        if a == b:
            continue
        n += 1; sa, sb = xa[COLS.index(key)] or 0, xb[COLS.index(key)] or 0
        ok, bad = (sb, sa) if a else (sa, sb)
        win += ok > bad; tie += ok == bad
    m = n - tie
    return n, win, tie, (binomtest(win, m).pvalue if m else float('nan'))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--picker', default='dpp0.5'); ap.add_argument('--others', default='cube0.5,sd')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'sub4', 'abstention_dev.md'))
    a = ap.parse_args(argv); dev = load_splits()['dev']
    A = flat_rows(a.picker, dev); rows = list(A.values())
    L = ['# SUB4 objection (f): abstention and within-query test (kpbench.sub4_abstain)', '',
         'DEV flat queries, stored K = 100 LightGlue rows of %s (%d queries, %d fails). rep5 uses true pose + depth (measurement, '
         'upper reference); n_inl / n_match are test-time.' % (a.picker, len(rows), int(fail_flags(rows).sum())), '',
         '## (i) Abstention: flat fail % of the kept queries after rejecting the lowest x %', '',
         '| score | AUROC | ' + ' | '.join('reject %d %%' % p for p in RATES) + ' | rel. drop at 10 % |', '|---' * (len(RATES) + 3) + '|']
    for k in SCORES:
        f, au = abstain(rows, k)
        L.append('| %s | %.3f | %s | %.1f %% |' % (k, au, ' | '.join('%.2f' % v for v in f), 100 * (1 - f[2] / f[0])))
    L += ['', '## (ii) Within query: picker-dependent flat frames (exactly one picker fails)', '',
          '| pair | frames | score | posed side higher | ties | sign-test p |', '|---|---|---|---|---|---|']
    for o in a.others.split(','):
        B = flat_rows(o, dev)
        for k in SCORES:
            n, w, t, p = within(A, B, k)
            L.append('| %s vs %s | %d | %s | %d (%.2f) | %d | %.2g |' % (a.picker, o, n, k, w, w / max(n - t, 1), t, p))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
