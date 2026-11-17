"""J6 proxy (SUB3, dispatch 2026-11-17 20:10): is the picker gain over grid only depth coverage?

  python tools/j6_proxy.py --arms raco_dpp0.5_K100_same_lg_picks,raco_grid_K100_same_lg_picks --split micro [--runs runs]
  python tools/j6_proxy.py --selftest

Needs runs made with kpbench.run --save-picks (per-pick depth, since 13 Nov); runs without picks are refused (gate.require_picks).
Per flat query: fail (0/1) and occ = depth-bin occupancy = number of occupied log2-depth bins of width 0.5 (depths 4, 5, 9, 30 m ->
bins 4, 4, 6, 9 -> occ 3). Linear probability model, first arm = reference:
  M0: fail ~ 1 + arm          M1: fail ~ 1 + arm + occ
Flight-cluster bootstrap 95 % CI (flights resampled, --boot, seed 0) of each arm coefficient in M0 and M1.
Reading (J6 kill, dispatch): the arm coefficient excludes 0 in M0 but covers 0 in M1 -> depth coverage explains the arm effect
-> the depth-quota control (grid + quota, K100) becomes the first SUB3 item after the lift.
"""
import argparse, gzip, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def occupancy(z, w=0.5):
    """number of occupied log2-depth bins of width w among the positive depths z"""
    z = np.array([x for x in z if x is not None and x > 0], float)
    return int(len(np.unique(np.floor(np.log2(z) / w)))) if len(z) else 0


def load(arms, split, runs):
    """-> flight, arm index, fail, occ arrays over the flat queries every arm has"""
    sys.path.insert(0, ROOT)
    from kpbench.gate import require_picks, split_info, labels, RESULTS
    from kpbench.data import slug
    from kpbench.stats import fail_flags
    from kpbench.splits import FLAT
    flights, _, _ = split_info(split); F, A, Y, O = [], [], [], []
    for f in flights:
        per = []
        for rid in arms:
            _, out = require_picks(rid, [f], runs)
            j = json.load(gzip.open(os.path.join(RESULTS, runs, rid, slug(f) + '.json.gz'), 'rt')); op = j.get('onplane') or {}
            per.append({r[1]: (int(b), op.get(str(r[1]))) for r, b in zip(out[f][0], fail_flags(out[f][0]))})
        lab = labels(f); qs = set.intersection(*[set(p) for p in per])
        for q in sorted(qs):
            if (lab.get(str(q), {}).get('plane_frac') or -1) < FLAT:
                continue
            for k, p in enumerate(per):
                e = p[q][1]
                if e is None or len(e) < 5 or e[4] is None:
                    continue
                F.append(f); A.append(k); Y.append(p[q][0]); O.append(occupancy(e[4]))
    return np.array(F), np.array(A), np.array(Y, float), np.array(O, float)


def fit(A, Y, O, n_arms, with_occ):
    X = [np.ones(len(Y))] + [(A == k).astype(float) for k in range(1, n_arms)] + ([O] if with_occ else [])
    return np.linalg.lstsq(np.stack(X, 1), Y, rcond=None)[0]


def proxy(F, A, Y, O, n_arms, boot=2000, seed=0):
    """-> {model: (coef, lo, hi)} for the arm coefficients (and occ in M1)"""
    fl = np.unique(F); idx = {f: np.where(F == f)[0] for f in fl}; rng = np.random.default_rng(seed); res = {}
    for name, w in (('M0', False), ('M1', True)):
        c = fit(A, Y, O, n_arms, w); B = []
        for _ in range(boot):
            ii = np.concatenate([idx[f] for f in rng.choice(fl, len(fl))])
            B.append(fit(A[ii], Y[ii], O[ii], n_arms, w))
        lo, hi = np.percentile(np.array(B), [2.5, 97.5], 0); res[name] = [(float(c[i]), float(lo[i]), float(hi[i])) for i in range(1, len(c))]
    return res


def selftest():
    """synthetic: arm 1 raises occupancy by 1 bin, fail depends on occupancy only -> arm effect in M0, none in M1"""
    rng = np.random.default_rng(1); F, A, Y, O = [], [], [], []
    for f in range(20):
        for q in range(100):
            for k in (0, 1):
                o = rng.integers(2, 6) + k; F.append(f); A.append(k); O.append(o); Y.append(float(rng.random() < 0.35 - 0.06 * o))
    r = proxy(np.array(F), np.array(A), np.array(Y), np.array(O, float), 2, boot=500)
    m0, m1 = r['M0'][0], r['M1'][0]; ok = m0[2] < 0 and m1[1] <= 0 <= m1[2] and r['M1'][1][2] < 0
    print('M0 arm %+.3f [%+.3f, %+.3f]; M1 arm %+.3f [%+.3f, %+.3f], occ %+.3f [%+.3f, %+.3f] -> %s'
          % (*m0, *m1, *r['M1'][1], 'PASS' if ok else 'FAIL'))
    print('occupancy([4, 5, 9, 30]) = %d (expect 3)' % occupancy([4, 5, 9, 30]))
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', default=''); ap.add_argument('--split', default='micro')
    ap.add_argument('--runs', default='runs'); ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    arms = a.arms.split(','); F, A, Y, O = load(arms, a.split, a.runs)
    print('flat query-arm rows %d, flights %d' % (len(Y), len(np.unique(F))))
    for k, rid in enumerate(arms):
        m = A == k; print('%s: n %d fail %.2f %% occ mean %.2f' % (rid, m.sum(), 100 * Y[m].mean(), O[m].mean()))
    r = proxy(F, A, Y, O, len(arms), a.boot)
    for name in ('M0', 'M1'):
        names = arms[1:] + (['occ'] if name == 'M1' else [])
        print(name, '; '.join('%s %+.4f [%+.4f, %+.4f]' % (n, *c) for n, c in zip(names, r[name])))


if __name__ == '__main__':
    main()
