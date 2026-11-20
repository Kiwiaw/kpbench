"""Second-method check of results/main/power_fresh2.md (SUB2, dispatch 2026-11-20): power of the registered FRESH-2
endpoint (dpp0.5 vs cube0.5 K100 flat, one-sided flight sign-flip, alpha 0.05) at 2774 and 4849 flat only.

  python tools/power_fresh2_check.py [--reps 10000] [--seed 7]      # prints PASS / FAIL per cell (tolerance 5 points)

Shares only the audited per-flight counts (tools/j7_lofo_check.flight_counts, T44 PASS) with tools/power_fresh2.py.
Own parts: (1) the p-value is the lower tail of the sign-flip sum built as a probability distribution with np.convolve
(each nonzero flight adds +v or -v with chance 1/2; power_fresh2 shifts count arrays), cross-checked on small sets against
a brute-force list of all 2^m patterns; (2) flights are drawn one at a time until the flat count reaches the target
(power_fresh2 draws a batch and cuts it with searchsorted); (3) two-stage draws each flight's (only-dpp, only-cube, neither)
counts in one multinomial (power_fresh2 uses two nested binomials); (4) other seed. Toy: d = (-5, -3, +1): sums of the 8
patterns = -9 -7 -3 -1 +1 +3 +7 +9, observed -7 -> p = 2/8 = 0.25.
"""
import argparse, itertools, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j7_lofo_check import ROOT, SETS, flight_counts


def p_lower(d):
    """one-sided sign-flip p = P(sum of +-|d_f| <= observed sum), zeros dropped, as a convolved distribution"""
    x = [abs(int(v)) for v in d if v]
    if not x:
        return 1.0
    dist = np.ones(1)
    for v in x:
        step = np.zeros(2 * v + 1); step[0] = step[-1] = 0.5
        dist = np.convolve(dist, step)                       # support -S..S after all flights
    S = sum(x)
    return float(dist[:S + int(sum(int(v) for v in d)) + 1].sum())


def p_brute(d):
    """same p by listing all 2^m patterns (m <= 16): the check of p_lower"""
    x = [abs(int(v)) for v in d if v]
    obs = sum(int(v) for v in d)
    return float(np.mean([sum(s * v for s, v in zip(sg, x)) <= obs for sg in itertools.product((-1, 1), repeat=len(x))])) if x else 1.0


def one_set(F, target, rng, two_stage, gap):
    """one simulated future set -> per-flight d = only-dpp fails - only-cube fails"""
    d = []; flat = 0
    while flat < target:
        n, oa, ob = F[rng.integers(len(F))]
        flat += n
        if two_stage:
            pa, pb = oa / n, ob / n
            pb = pa + gap * (pb - pa)                        # gap 0.5 = half the observed gap of this flight
            a, b, _ = rng.multinomial(n, [pa, pb, 1 - pa - pb])
            d.append(a - b)
        else:
            d.append(oa - ob)
    return d


def published():
    """power_fresh2.md table -> {(source, resampling): (power at 2774, power at 4849)}"""
    txt = open(os.path.join(ROOT, 'results', 'main', 'power_fresh2.md'), encoding='utf-8').read()
    T = {}
    for m in re.finditer(r'^\| (board|fresh1|pooled) \| ([^|]+?) \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \|', txt, re.M):
        T[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=10000); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--tol', type=float, default=5.0); a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for _ in range(200):                                     # p_lower against brute force on random small sets
        d = rng.integers(-9, 10, rng.integers(1, 13))
        assert abs(p_lower(d) - p_brute(d)) < 1e-12, d
    assert abs(p_lower([-5, -3, 1]) - 0.25) < 1e-12
    print('p_lower == brute force on 200 random sets and the toy (0.25): OK')
    src = {}
    for name, (ra, rb, exp) in SETS.items():
        C = flight_counts(ra, rb, pairs=True)
        src[name] = [(n, oa, fb - fa + oa) for n, fa, fb, oa in C.values()]
        tot = [sum(c[i] for c in C.values()) for i in range(3)]
        assert tuple(tot) == exp[:3], (name, tot)
        print('%s: %d flights, flat %d, fails %d vs %d, one-sided p %.4f' % (name, len(C), tot[0], tot[1], tot[2], p_lower([o - b for _, o, b in src[name]])))
    src['pooled'] = src['fresh1'] + src['board']
    pub = published(); worst = 0.0; bad = 0
    print('| source | resampling | 2774 flat: check / note / diff | 4849 flat: check / note / diff | verdict |'); print('|---|---|---|---|---|')
    for name, F in src.items():
        for label, ts, gap in (('flights', False, 1.0), ('two-stage', True, 1.0), ('two-stage, gap x 0.5', True, 0.5), ('two-stage, gap x 0.0', True, 0.0)):
            cells = []
            for j, t in enumerate((2774, 4849)):
                pw = 100 * np.mean([p_lower(one_set(F, t, rng, ts, gap)) < 0.05 for _ in range(a.reps)])
                cells.append((pw, pub[(name, label)][j], pw - pub[(name, label)][j]))
            w = max(abs(c[2]) for c in cells); worst = max(worst, w); bad += w > a.tol
            print('| %s | %s | %s | %s |' % (name, label, ' | '.join('%.1f / %.1f / %+.1f' % c for c in cells), 'PASS' if w <= a.tol else 'FAIL'), flush=True)
    print('reps %d per cell, seed %d, Monte Carlo SE <= %.2f pt per number; largest |diff| %.1f pt; tolerance %.1f pt -> %s'
          % (a.reps, a.seed, 50 / np.sqrt(a.reps), worst, a.tol, 'PASS' if not bad else 'FAIL (%d rows)' % bad))


if __name__ == '__main__':
    main()
