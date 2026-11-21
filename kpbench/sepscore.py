"""SUB1 one-candidate scorer for 'sep' (T33; rule = keypoint-gap-agent results/sub1/sep_rule_prereg.md + addendum 2026-11-21).

  python -m kpbench.sepscore --arms <uniform>,<strong>,<other> --split dev [--runs runs] [--out results/sub1/sep_dev.md]
  python -m kpbench.sepscore ... --check          # format check only: validates the dump, prints NO value

m = off = off-plane share of a query's picks with depth (100 picks with depth, 80 on the plane -> 0.20), higher = better.
All four must hold (uniform = dpp0.5, strong = sd, other = cube0.5; K100, 12 DEV flights, one picks dump):
  1. adds over inlier count: leave-one-flight-out, pooled over the three arms, dAUROC >= 0.01 and dPR-AUC >= 0.01;
  2. uniform raises it: d_q = off(uniform) - off(strong), 20-frame block bootstrap z >= 2.5;
  3. flat frames where exactly one of uniform / strong fails (and uniform / other): the posed arm has the higher off in
     >= 70 % (ties dropped), sign-test p < 0.05, both pairs;
  4. flight check: flights with >= 5 failed and >= 5 posed flat queries in the uniform arm, sep_f = mean off failed - mean off
     posed; zeros dropped, exact one-sided sign test on sep_f < 0, p < 0.05, >= 5 flights left (10 of 12 passes, 9 of 12 does not).
The dump is validated BEFORE any value is computed (arms, has_picks, flights, query sets, <= 5 % missing off): a defect stops
the run with no number printed (crashed run, not a read). A printed table is the one read. The real read also requires the
arms in the order dpp0.5, sd, cube0.5 at K100 lg, the dev split with its registered flat count, one device per flight and no
earlier output file (check_read). Thresholds are applied to the 4-decimal rounded values, as for the 12 closed candidates.
"""
import argparse, gzip, json, os
import numpy as np
from scipy.stats import binomtest
from .data import slug
from .gate import require_picks, split_info, labels, RESULTS, GateError
from .stats import fail_flags
from .splits import FLAT
from .sub1 import boot_se
from . import sub1_ferranti as fe
from .sub1_prereg import within
from .pickarms import cluster_ci

MAX_MISSING = 0.05


def load_arm(rid, flights, runs='runs'):
    """-> {(flight, q): (fail, off or None, n_inl, flat)} over all queries of one run with picks"""
    _, out = require_picks(rid, flights, runs); d = os.path.join(RESULTS, runs, rid); res = {}
    for f in flights:
        lab = labels(f); op = json.load(gzip.open(os.path.join(d, slug(f) + '.json.gz'), 'rt')).get('onplane') or {}
        rows = out[f][0]
        for r, bad in zip(rows, fail_flags(rows)):
            e = op.get(str(r[1]))
            off = None if not e or e[0] is None else 1.0 - float(e[0])
            res[(f, int(r[1]))] = (int(bad), off, float(r[6] or 0), (lab.get(str(r[1]), {}).get('plane_frac') or -1) >= FLAT)
        if len([k for k in res if k[0] == f]) != len(rows):
            raise GateError('%s %s: duplicate query rows' % (rid, f))
    return res


READ_ARMS = ('dpp0.5', 'sd', 'cube0.5')                           # uniform, strong, other: fixed order of the one read


def check_read(arms, split, out, T, flights, runs='runs'):
    """extra checks of the real read (not of --check): arm identity and order, K100 + lg, dev split, flat count of the split,
    same device per flight across arms, no overwrite of an earlier read"""
    bad = []
    for a, p in zip(arms, READ_ARMS):
        if '_%s_K100_' % p not in a or '_lg' not in a:
            bad.append('arm %s is not %s K100 lg (order: uniform dpp0.5, strong sd, other cube0.5)' % (a, p))
    if split != 'dev':
        bad.append('the read is on dev, not %s' % split)
    want = split_info(split)[1] if split == 'dev' else None; got = sum(v[3] for v in T[0].values())
    if want is not None and got != want:
        bad.append('flat queries %d, split says %d (labels missing or stale)' % (got, want))
    dev = [{f: load[1] for f, load in require_picks(a, flights, runs)[1].items()} for a in arms]
    diff = [f for f in flights if len({d[f] for d in dev}) > 1]
    if diff:
        bad.append('arms ran on different devices on %d flights (first %s)' % (len(diff), diff[0]))
    if out and (os.path.exists(out) or os.path.exists(os.path.splitext(out)[0] + '.json')):
        bad.append('%s exists: one read only' % out)
    if bad:
        raise GateError('; '.join(bad))


def validate(arms, flights, runs='runs'):
    """-> (tables, kept keys, n dropped); raises GateError on any dump defect, before any value exists"""
    if len(arms) != 3 or len(set(arms)) != 3:
        raise GateError('need three different arms (uniform, strong, other), got %s' % (arms,))
    T = [load_arm(a, flights, runs) for a in arms]                 # require_picks: missing flight / no picks -> GateError
    if not (T[0].keys() == T[1].keys() == T[2].keys()):
        raise GateError('arms do not share the query set (%s queries)' % ' / '.join(str(len(t)) for t in T))
    keys = sorted(T[0]); keep = [k for k in keys if all(t[k][1] is not None for t in T)]
    miss = len(keys) - len(keep)
    if not keys or miss > MAX_MISSING * len(keys):
        raise GateError('off missing on %d of %d queries (> %d %%): picks dump defect' % (miss, len(keys), round(100 * MAX_MISSING)))
    return T, keep, miss


def sign_p(k, m):
    """exact one-sided P(X >= k | m, 1/2); 9 of 12 -> 0.073, 10 of 12 -> 0.019"""
    return float(binomtest(k, m, 0.5, alternative='greater').pvalue) if m else 1.0


def flight_check(t, keys, min_n=5):
    """rule 4 on the uniform arm: per flight sep_f over flat queries -> dict(flights, neg, pos, zero, p, ok)"""
    S = {}
    for k in keys:
        bad, off, _, flat = t[k]
        if flat:
            S.setdefault(k[0], ([], []))[0 if bad else 1].append(off)
    sep = [float(np.mean(f) - np.mean(p)) for f, p in S.values() if len(f) >= min_n and len(p) >= min_n]
    neg = sum(v < 0 for v in sep); pos = sum(v > 0 for v in sep); m = neg + pos; p = sign_p(neg, m)
    return dict(flights=len(sep), neg=neg, pos=pos, zero=len(sep) - m, p=round(p, 4), ok=bool(m >= min_n and p < 0.05))


def score(T, keys, arms):
    """the four rules on validated tables -> result dict"""
    U, S, O = T; fl = np.array([k[0] for k in keys])
    col = lambda t, i: np.array([t[k][i] for k in keys], float)
    F = [col(t, 0) for t in T]; X = [col(t, 1) for t in T]; flat = col(U, 3).astype(bool)
    r1 = fe.score(np.concatenate([fl] * 3), dict(n_inl=np.log1p(np.concatenate([col(t, 2) for t in T])), hull=np.concatenate(X)),
                  np.concatenate(F)).get('delta', {})
    ok1 = bool(r1) and r1['auroc'] >= 0.01 and r1['prauc'] >= 0.01
    d = X[0] - X[1]; _, inv = np.unique(['%s:%d' % (k[0], k[1] // 20) for k in keys], return_inverse=True)
    se = boot_se(d, inv); z = float(d.mean() / se) if se > 0 else 0.0
    w = {arms[i]: within(F[0].astype(bool), F[i].astype(bool), X[0], X[i], flat) for i in (1, 2)}
    ok3 = all(v['share'] is not None and v['share'] >= 0.70 and v['p'] < 0.05 for v in w.values())
    r4 = flight_check(U, keys)
    sep = {}
    for a, f, x in zip(arms, F, X):
        b = (f == 1) & flat; g = (f == 0) & flat
        sep[a] = None if not b.any() or not g.any() else round(float(x[b].mean() - x[g].mean()), 4)
    ok = ok1 and z >= 2.5 and ok3 and r4['ok']
    return dict(arms=list(arms), n=len(keys), n_flat=int(flat.sum()), fails={a: int(f.sum()) for a, f in zip(arms, F)},
                d_auroc=r1.get('auroc'), d_prauc=r1.get('prauc'), rule1=ok1, d_mean=round(float(d.mean()), 4), z=round(z, 2), rule2=bool(z >= 2.5),
                d_ci=[round(v, 4) for v in cluster_ci(d, list(fl))], within=w, rule3=ok3, flight=r4, rule4=r4['ok'], sep=sep, passed=bool(ok),
                verdict='PASS at K100: sep earns a K400 column (SUB1 not reopened)' if ok else 'FAIL: SUB1 closed at 0/13, sep stays descriptive')


def to_md(r, split, miss):
    y = lambda b: 'yes' if b else 'no'; u, s, o = r['arms']
    f = lambda w: '%d / %s / %s' % (w['n'], w['share'], 'nan' if w['p'] is None else '%.2g' % w['p'])
    q = r['flight']
    return '\n'.join([
        '# SUB1 sep read on %s (%d queries, %d flat, %d dropped for missing off)' % (split, r['n'], r['n_flat'], miss), '',
        'Arms: uniform %s, strong %s, other %s. Fails %s. sep per arm (flat): %s.' % (u, s, o, r['fails'], r['sep']), '',
        '| rule | value | pass |', '|---|---|---|',
        '| 1 adds over inlier count | dAUROC %s, dPR-AUC %s | %s |' % (r['d_auroc'], r['d_prauc'], y(r['rule1'])),
        '| 2 uniform raises off | d %+.4f, z %.2f (flight-cluster CI [%+.4f, %+.4f], descriptive) | %s |' % (r['d_mean'], r['z'], *r['d_ci'], y(r['rule2'])),
        '| 3 within vs strong n / share / p; vs other | %s; %s | %s |' % (f(r['within'][s]), f(r['within'][o]), y(r['rule3'])),
        '| 4 flight check sep_f < 0 | %d of %d flights (%d zero dropped), p %.4f | %s |' % (q['neg'], q['neg'] + q['pos'], q['zero'], q['p'], y(r['rule4'])),
        '', '**%s**' % r['verdict']]) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', required=True, help='uniform,strong,other run ids (with picks)')
    ap.add_argument('--split', default='dev'); ap.add_argument('--runs', default='runs'); ap.add_argument('--out', default=None)
    ap.add_argument('--flights', default=None, help='comma list overriding the split flights (format checks on other data, with --check)')
    ap.add_argument('--check', action='store_true', help='validate the dump only; no value is computed or printed')
    a = ap.parse_args(argv); arms = a.arms.split(',')
    if a.flights and not a.check:
        raise SystemExit('--flights is for --check only: the read runs on the split flights')
    flights = a.flights.split(',') if a.flights else split_info(a.split)[0]
    try:
        T, keys, miss = validate(arms, flights, a.runs)
    except GateError as e:
        raise SystemExit('CRASHED RUN (no value computed, not a read): %s' % e)
    if a.check:
        print('format OK: 3 arms, %d flights, %d queries, %d with off in all arms (%d missing), has_picks True; no value computed'
              % (len(flights), len(T[0]), len(keys), miss)); return None
    try:
        check_read(arms, a.split, a.out, T, flights, a.runs)
    except GateError as e:
        raise SystemExit('CRASHED RUN (no value computed, not a read): %s' % e)
    try:
        r = score(T, keys, arms); txt = to_md(r, a.split, miss)
    except Exception as e:                                      # nothing was printed: still a crashed run (addendum point 8)
        raise SystemExit('CRASHED RUN (scorer error before any value was printed, not a read): %r' % e)
    print(txt)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt); json.dump(r, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1)
    return r


if __name__ == '__main__':
    main()
