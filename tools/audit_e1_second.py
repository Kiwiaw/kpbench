"""Second-method check of E1 and the FRESH-1 K100 close-out line (SUB2, dispatch 2026-11-27 13:20). [P] post hoc.

  python tools/audit_e1_second.py [--boot 100000] [--seed 1] [--out results/audit/2026-11-27_e1_second_method.md]

Independent of tools/e1_equal_compute.py and tools/dpp_vs_cube_k100.py: reads the run .json.gz files and label files directly
(no kpbench.gate / kpbench.stats / tools imports), pairs rows by query id (inner join, refuses on a mismatch), applies the fail
rule itself (no pose, > 25 cm or > 2 deg), flat = plane_frac >= 0.85. Interval 1: flight-cluster percentile bootstrap (ratio of
sums), other seed and 10x the draws. Interval 2: delta-method CI of the ratio estimator over flights (no resampling at all).
Sign-flip p: full enumeration of the 2^m sign patterns of the non-zero per-flight count diffs.
"""
import argparse, gzip, itertools, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RUNS = dict(c100='raco_cube0.5_K100_same_lg_g-cube0.5-fresh', c400='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400',
            i100='raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', i400='raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400')
PAIRS = [('E1: cube0.5 K400 - dpp0.5 K100', 'c400', 'i100', (-0.24, -1.59, 1.45, 0.8828)),
         ('dpp0.5 K100 - cube0.5 K100 (close-out)', 'i100', 'c100', (-1.59, -3.46, -0.25, 0.0508)),
         ('cube0.5 K400 - dpp0.5 K400', 'c400', 'i400', (-0.34, -1.03, 0.32, 0.5781))]
TOL = 0.05


def read(run):
    d = os.path.join(ROOT, 'results', 'runs', run); out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json.gz'):
            j = json.load(gzip.open(os.path.join(d, fn), 'rt')); out[fn[:-8]] = {int(r[1]): r for r in j['rows']}
    return out


def failed(r):
    return r[2] is None or r[3] is None or r[2] > 25 or r[3] > 2


def counts(R, a, b):
    """-> flights, fails a, fails b, n flat (paired on query id)"""
    fl, fa, fb, nn = [], [], [], []
    for f in sorted(R[a]):
        A, B = R[a][f], R[b][f]
        if set(A) != set(B):
            raise SystemExit('%s: query sets differ between %s and %s' % (f, a, b))
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f + '.json')))
        qs = [q for q in A if (lab[str(q)].get('plane_frac') or -1) >= 0.85]
        if qs:
            fl.append(f); fa.append(sum(failed(A[q]) for q in qs)); fb.append(sum(failed(B[q]) for q in qs)); nn.append(len(qs))
    return fl, np.array(fa, float), np.array(fb, float), np.array(nn, float)


def boot_ci(d, n, boot, seed):
    rng = np.random.default_rng(seed); m = len(n); lo = []
    for s in range(0, boot, 10000):
        i = rng.integers(0, m, (min(10000, boot - s), m)); lo.append(100 * d[i].sum(1) / n[i].sum(1))
    return tuple(np.percentile(np.concatenate(lo), [2.5, 97.5]))


def delta_ci(d, n):
    m = len(n); r = d.sum() / n.sum(); e = d - r * n
    se = np.sqrt(m / (m - 1) * (e ** 2).sum()) / n.sum()
    return 100 * (r - 1.96 * se), 100 * (r + 1.96 * se)


def signflip(d):
    d = [x for x in d if x != 0]; obs = abs(sum(d)); hit = 0
    for s in itertools.product((1, -1), repeat=len(d)):
        hit += abs(sum(a * b for a, b in zip(s, d))) >= obs - 1e-9
    return hit / 2 ** len(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=100000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'audit', '2026-11-27_e1_second_method.md')); a = ap.parse_args()
    R = {k: read(v) for k, v in RUNS.items()}
    L = ['# E1 + FRESH-1 K100 close-out line: second method (SUB2, 2026-11-27) [P]', '',
         'Script: tools/audit_e1_second.py (own reader, own fail rule, query-id join; boot %d seed %d; delta-method CI; full sign-flip'
         ' enumeration). First method: tools/e1_equal_compute.py, tools/dpp_vs_cube_k100.py (boot 10000 seed 0). Tolerance %.2f pt.' % (a.boot, a.seed, TOL), '',
         '| contrast | flights | flat | fails a / b | diff pt | boot CI (2nd) | delta CI | flights neg/pos/zero | sign p | 1st method | verdict |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    allok = True
    for name, x, y, ref in PAIRS:
        fl, fa, fb, n = counts(R, x, y); d = fa - fb; pt = 100 * d.sum() / n.sum()
        lo, hi = boot_ci(d, n, a.boot, a.seed); dlo, dhi = delta_ci(d, n); p = signflip(d.tolist())
        ok = abs(pt - ref[0]) <= TOL and abs(lo - ref[1]) <= TOL and abs(hi - ref[2]) <= TOL and abs(p - ref[3]) < 1e-3 \
            and np.sign(lo) == np.sign(ref[1]) and np.sign(hi) == np.sign(ref[2])
        allok &= ok
        L.append('| %s | %d | %d | %d / %d | %+.2f | [%+.2f, %+.2f] | [%+.2f, %+.2f] | %d / %d / %d | %.4f | %+.2f [%+.2f, %+.2f], p %.4f | %s |' % (
            name, len(fl), n.sum(), fa.sum(), fb.sum(), pt, lo, hi, dlo, dhi, (d < 0).sum(), (d > 0).sum(), (d == 0).sum(), p,
            ref[0], ref[1], ref[2], ref[3], 'PASS' if ok else 'FAIL'))
    L += ['', 'Verdict: **%s** (diff and both bootstrap bounds within %.2f pt, same signs, sign p equal to 3 decimals).' % ('PASS' if allok else 'FAIL', TOL),
          'Delta CI is a second interval shape (normal approximation on 13 flights), shown for context, not part of the tolerance test.']
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
