"""Second-method check of the T48 non-inferiority projection (SUB2, dispatch 2026-11-21): dppr0.317a1.6b2.01t0.9 vs dpp0.5,
K100, FRESH-1 flat fail diff (candidate - incumbent, pt), margin +1.0 pt.

  python tools/noninf_scope_check.py [--outer 4000] [--inner 1000] [--seed 7] [--out results/audit/noninf_scope_check_2026-11-21.md]

Published (T48, keypoint-gap-agent results/sub3/noninf_scope.md): flight-cluster CI on FRESH-1 [-0.39, +1.05]; chance that the
upper bound is < +1.0: 60 % at 2774 flat, 71 % at 3500, 84 % at pooled 4849; median upper +0.88 at 2774; query-CI scaling
upper +0.76 at 2774. Agreement rule (dispatch 2026-11-20T14:45): within 5 points of pass chance and 0.1 pt of bound.

Fixed before any number of this script was seen:
  A (decides the tag of 60 / 71 / 84 % and +0.88): own nested flight resampling. Rows read here (own fail rule: err None or
    > 25 cm or > 2 deg; flat = plane_frac >= 0.85). Outer: flights drawn one at a time with replacement until the flat count
    reaches the target (last flight kept whole). Inner: flight bootstrap by multinomial WEIGHTS on the drawn flights (T48 draws
    indices, 400 draws; here 1000), upper = 97.5th percentile of sum(w d) / sum(w n). Other seed (7).
  B (decides the tag of +0.76): no sqrt scaling. 20-frame query blocks (as the gate) are drawn with replacement until the flat
    count reaches the target; upper = 97.5th percentile of the resampled diff (an m-out-of-n block bootstrap).
  C (shown, never part of the rule): normal approximation with the linearised ratio SE of the 13 flights,
    chance = Phi((1.0 - diff) / se_n - 1.96), se_n = se_13 x sqrt(2075 / n). It ignores that the bound itself varies from set
    to set, so it is expected to sit above A.
Toy: flights (n, d) = (100, +1), (100, -1), (200, +2): diff = 2 / 400 = +0.5 pt.
"""
import argparse, gzip, json, math, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CAND = 'raco_dppr0.317a1.6b2.01t0.9_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh'
INC = 'raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh'
MARGIN = 1.0
PUB = {2774: (60, 0.88), 3500: (71, 0.80), 4849: (84, 0.72)}          # T48: pass chance %, median upper
PUB_SCALED = {2774: 0.76, 3500: 0.70, 4849: 0.63}                      # T48: query-CI scaling upper
PUB_CI = (-0.39, 1.05)


def load():
    """-> list of (flight, q, d) over flat queries, d = fail(cand) - fail(inc) in {-1, 0, +1}; and the two fail totals"""
    Q = []; tot = [0, 0]
    dc, di = (os.path.join(ROOT, 'results', 'runs', r) for r in (CAND, INC))
    for fn in sorted(os.listdir(dc)):
        if not fn.endswith('.json.gz'):
            continue
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', fn[:-8] + '.json')))
        bad = []
        for d in (dc, di):
            rows = json.load(gzip.open(os.path.join(d, fn), 'rt'))['rows']
            bad.append({int(r[1]): (r[2] is None) or (r[2] > 25) or (r[3] > 2) for r in rows})
        assert bad[0].keys() == bad[1].keys(), fn
        for q in sorted(bad[0]):
            if (lab.get(str(q), {}).get('plane_frac') or -1) >= 0.85:
                Q.append((fn[:-8], q, int(bad[0][q]) - int(bad[1][q]))); tot[0] += bad[0][q]; tot[1] += bad[1][q]
    return Q, tot


def sums(Q, key):
    """group (flight, q, d) by key -> arrays n, d per group"""
    G = {}
    for f, q, d in Q:
        g = G.setdefault(key(f, q), [0, 0]); g[0] += 1; g[1] += d
    v = np.array([G[k] for k in sorted(G)], float)
    return v[:, 0], v[:, 1]


def draw_until(n, target, rng):
    """indices of groups drawn one at a time with replacement until their n sums to >= target"""
    idx = []; tot = 0
    while tot < target:
        i = int(rng.integers(len(n))); idx.append(i); tot += n[i]
    return np.array(idx)


def method_a(n, d, target, outer, inner, rng):
    """nested flight resampling -> (pass chance %, median upper, mean flights per set)"""
    up = np.empty(outer); m = 0; tot = 0.0
    for k in range(outer):
        idx = draw_until(n, target, rng); m += len(idx); tot += n[idx].sum()
        w = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), inner).astype(float)
        up[k] = np.percentile(100 * (w @ d[idx]) / (w @ n[idx]), 97.5)
    return 100 * float((up < MARGIN).mean()), float(np.median(up)), m / outer, tot / outer


def method_b(bn, bd, target, outer, rng):
    """m-out-of-n block bootstrap -> 97.5th percentile of the diff at the target size"""
    r = np.empty(outer)
    for k in range(outer):
        idx = draw_until(bn, target, rng); r[k] = 100 * bd[idx].sum() / bn[idx].sum()
    return float(np.percentile(r, 97.5))


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--outer', type=int, default=4000); ap.add_argument('--inner', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); rng = np.random.default_rng(a.seed)
    Q, tot = load(); n, d = sums(Q, lambda f, q: f); bn, bd = sums(Q, lambda f, q: (f, q // 20))
    N = int(n.sum()); diff = 100 * d.sum() / N
    L = ['# T48 non-inferiority projection, second method (SUB2, %s)' % 'tools/noninf_scope_check.py', '',
         'Rows: %d flights, %d flat, fails cand %d vs inc %d, diff %+.2f pt (T48: 13 flights, 2075, 67 vs 62, +0.24).' % (len(n), N, tot[0], tot[1], diff)]
    ok_in = (len(n), N, tot[0], tot[1]) == (13, 2075, 67, 62)
    # FRESH-1 flight-cluster CI, own weights bootstrap
    w = rng.multinomial(len(n), np.full(len(n), 1.0 / len(n)), 20000).astype(float); b = 100 * (w @ d) / (w @ n)
    lo, hi = np.percentile(b, [2.5, 97.5]); ok_ci = abs(lo - PUB_CI[0]) <= 0.1 and abs(hi - PUB_CI[1]) <= 0.1
    e = d - (d.sum() / N) * n; se13 = 100 * math.sqrt(len(n) / (len(n) - 1) * (e ** 2).sum()) / N
    L += ['FRESH-1 flight-cluster CI (20000 weight draws, seed %d): [%+.2f, %+.2f] vs T48 [%+.2f, %+.2f] -> %s. Linearised flight SE %.3f pt.'
          % (a.seed, lo, hi, *PUB_CI, 'agree' if ok_ci else 'DISAGREE', se13), '',
          '| flat | A pass chance | T48 | diff | A median upper | T48 | diff | B upper (blocks) | T48 scaled | diff | C normal approx (not in rule) | flights per set | realised flat |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    ok = ok_in and ok_ci; worst = [0.0, 0.0, 0.0]
    for t, (pc, pu) in PUB.items():
        ch, med, m, real = method_a(n, d, t, a.outer, a.inner, rng); ub = method_b(bn, bd, t, a.outer * 5, rng)
        c = 100 * phi((MARGIN - diff) / (se13 * math.sqrt(N / t)) - 1.96)
        x = (abs(ch - pc), abs(med - pu), abs(ub - PUB_SCALED[t])); worst = [max(u, v) for u, v in zip(worst, x)]
        ok = ok and x[0] <= 5 and x[1] <= 0.1 and x[2] <= 0.1
        L.append('| %d | %.1f %% | %d %% | %+.1f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.1f %% | %.1f | %.0f |'
                 % (t, ch, pc, ch - pc, med, pu, med - pu, ub, PUB_SCALED[t], ub - PUB_SCALED[t], c, m, real))
    L += ['', 'Agreement rule: pass chance within 5 points, bounds within 0.1 pt. Worst gaps: chance %.1f points, median upper %.2f pt, block upper %.2f pt.' % tuple(worst),
          '**%s** (outer %d, inner %d, seed %d)' % ('PASS -> T48 numbers [T] as REPLICATED' if ok else 'FAIL -> T48 numbers stay [P]', a.outer, a.inner, a.seed), '',
          'What this PASS is (review 2026-11-21): A re-codes the published nested flight resampling with other code and seed, so it rules out a coding or seed error, not a flaw of the method. Only C is a different method; it sits 2-3 points above A. A and T48 both keep the last flight whole, so the sets are larger than the target (column realised flat): the pass chances are slightly optimistic for the nominal sizes. B is a different way to the scaled bound (resampling, no sqrt rule).']
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)
    return ok


if __name__ == '__main__':
    main()
