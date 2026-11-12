"""SUB2 r30 + MAIN J1: pixel dpp2d120a1 vs 3D dpp0.5, FRESH-1 K100, recomputed from the raw mirrored rows (engineer 12 Nov).

  python tools/r30_dpp2d_recheck.py [--out-audit results/audit/r30_dpp2d_fresh1.md] [--out-j1 results/main/j1_signflip_2026-11-12.md]

r30: reads the run .json.gz files and results/labels/<flight>.json directly (no kpbench.gate / stats / rep5_paired import), re-applies
the rule (flat = plane_frac >= 0.85; fail = no pose or > 25 cm or > 2 deg), re-implements the 20-frame block bootstrap (1000 draws,
seed 0 = gate default, plus seed 1) and the flight-cluster bootstrap (2000 draws, seed 22), and checks every number of the gate json.
J1: exact sign-flip (permutation) test over all 2^13 sign patterns of the per-flight differences, in fail counts and in fail-rate
points; leave-one-flight-out net difference; one table with sign, Wilcoxon, sign-flip and both CIs. Nothing here changes a verdict.
"""
import argparse, gzip, itertools, json, os, sys
from math import comb
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CAND = 'raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh'
INC = 'raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh'
GATE = os.path.join(ROOT, 'results', 'gates', '%s__vs__%s__fresh.json' % (CAND, INC))


def load(rid, flights):
    out = {}
    for f in flights:
        j = json.load(gzip.open(os.path.join(ROOT, 'results', 'runs', rid, f.replace('/', '_') + '.json.gz'), 'rt'))
        out[f] = j['rows']
    return out


def fails(rows):
    return np.array([r[2] is None or r[2] > 25 or r[3] > 2 for r in rows], float)


def block_ci(d, blocks, n_boot, seed):
    """pooled mean of d with a block bootstrap 95 % CI (blocks resampled with replacement)"""
    u, inv = np.unique(blocks, return_inverse=True); m = len(u)
    s = np.bincount(inv, d, m); c = np.bincount(inv, None, m)
    pick = np.random.default_rng(seed).integers(0, m, (n_boot, m))
    bs = s[pick].sum(1) / c[pick].sum(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def signflip(d):
    """exact sign-flip p of sum(d): (one-sided P(S >= obs), two-sided P(|S| >= |obs|), patterns); zeros do not change S"""
    d = np.asarray(d, float); nz = d[d != 0]; obs = nz.sum(); ge = ab = 0
    for s in itertools.product((1, -1), repeat=len(nz)):
        t = float(np.dot(s, nz)); ge += t >= obs - 1e-12; ab += abs(t) >= abs(obs) - 1e-12
    n = 2 ** len(nz)
    return ge / n, ab / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-audit', default=os.path.join(ROOT, 'results', 'audit', 'r30_dpp2d_fresh1.md'))
    ap.add_argument('--out-j1', default=os.path.join(ROOT, 'results', 'main', 'j1_signflip_2026-11-12.md'))
    a = ap.parse_args()
    g = json.load(open(GATE)); flights = json.load(open(os.path.join(ROOT, 'splits_fresh.json')))['fresh']
    rc, ri = load(CAND, flights), load(INC, flights)
    F, Q, FC, FI, FL, RC, RI = [], [], [], [], [], [], []
    per = []
    for f in flights:
        a_, b_ = rc[f], ri[f]
        assert [r[1] for r in a_] == [r[1] for r in b_], f
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))
        fl = np.array([(lab[str(r[1])].get('plane_frac') if lab[str(r[1])].get('plane_frac') is not None else -1) >= 0.85 for r in a_])
        fa, fb = fails(a_), fails(b_)
        F += [f] * len(a_); Q += [r[1] for r in a_]; FC += list(fa); FI += list(fb); FL += list(fl)
        RC += [r[8] for r in a_]; RI += [r[8] for r in b_]
        if fl.sum():
            per.append((f, int(fl.sum()), int(fa[fl].sum()), int(fb[fl].sum())))
    F, Q, FC, FI, FL = np.array(F), np.array(Q), np.array(FC), np.array(FI), np.array(FL, bool)
    blocks = np.array(['%s:%d' % (f, q // 20) for f, q in zip(F, Q)])
    fd = block_ci(100 * (FC[FL] - FI[FL]), blocks[FL], 1000, 0); fd1 = block_ci(100 * (FC[FL] - FI[FL]), blocks[FL], 1000, 1)
    ad = block_ci(100 * (FC - FI), blocks, 1000, 0)
    n = np.array([p[1] for p in per]); dc = np.array([p[2] for p in per]); di = np.array([p[3] for p in per])
    pick = np.random.default_rng(22).integers(0, len(n), (2000, len(n)))
    bs = 100 * (dc[pick] - di[pick]).sum(1) / n[pick].sum(1)
    cl = (100 * (dc - di).sum() / n.sum(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    pc, pi = 100 * dc / n, 100 * di / n
    better, worse = int((pc < pi).sum()), int((pc > pi).sum()); tied = len(n) - better - worse
    sp1 = sum(comb(better + worse, i) for i in range(better, better + worse + 1)) / 2 ** (better + worse)
    sp2 = min(1.0, 2 * sum(comb(better + worse, i) for i in range(0, min(better, worse) + 1)) / 2 ** (better + worse))
    # rep5 flat, pairwise drop of None (as the gate's info block)
    rm = np.array([x is not None and y is not None for x, y in zip(RC, RI)]) & FL
    r5c = np.array([x for x, k in zip(RC, rm) if k], float); r5i = np.array([y for y, k in zip(RI, rm) if k], float)
    r5 = block_ci(r5c - r5i, blocks[rm], 1000, 0)
    g5 = g['rep5']['flat']
    chk = [
        ('flat frames', int(FL.sum()), g['n_flat'], int(FL.sum()) == g['n_flat']),
        ('queries', len(FC), g['n'], len(FC) == g['n']),
        ('flat fails cand (dpp2d120a1)', int(FC[FL].sum()), round(g['flat_cand'] * g['n_flat'] / 100), int(FC[FL].sum()) == round(g['flat_cand'] * g['n_flat'] / 100)),
        ('flat fails inc (dpp0.5)', int(FI[FL].sum()), round(g['flat_inc'] * g['n_flat'] / 100), int(FI[FL].sum()) == round(g['flat_inc'] * g['n_flat'] / 100)),
        ('inc flat fail %', '%.2f' % (100 * FI[FL].mean()), '%.2f' % g['flat_inc'], abs(100 * FI[FL].mean() - g['flat_inc']) < 1e-9),
        ('flat diff, block CI seed 0', '%+.2f [%+.2f, %+.2f]' % fd, '%+.2f [%+.2f, %+.2f]' % tuple(g['flat_diff']), np.allclose(fd, g['flat_diff'], atol=1e-9)),
        ('all diff, block CI seed 0', '%+.2f [%+.2f, %+.2f]' % ad, '%+.2f [%+.2f, %+.2f]' % tuple(g['all_diff']), np.allclose(ad, g['all_diff'], atol=1e-9)),
        ('flights better / worse / tied (cand)', '%d / %d / %d' % (better, worse, tied), '%d / %d / %d' % (g['flights_better'], g['flights_worse'], g['flights_tied']), (better, worse, tied) == (g['flights_better'], g['flights_worse'], g['flights_tied'])),
        ('sign p one-sided (cand lower)', '%.4f' % sp1, '%.4f' % g['sign_p'], abs(sp1 - g['sign_p']) < 1e-12),
        ('flight-cluster CI 2000 x seed 22', '%+.2f [%+.2f, %+.2f]' % cl, '+2.75 [+0.05, +5.58] (10 Nov note)', '%+.2f [%+.2f, %+.2f]' % cl == '+2.75 [+0.05, +5.58]'),
        ('rep5 flat n', len(r5c), g5['n'], len(r5c) == g5['n']),
        ('rep5 flat diff, block CI seed 0', '%+.4f [%+.4f, %+.4f]' % r5, '%+.4f [%+.4f, %+.4f]' % tuple(g5['diff']), np.allclose(r5, g5['diff'], atol=1e-9)),
    ]
    ok = all(c[3] for c in chk)
    L = ['# r30: dpp2d120a1 vs dpp0.5 FRESH-1 K100 gate table recomputed from raw rows (engineer 2026-11-12, laptop, 0 GPU-h)', '',
         'Source: results/runs/%s and results/runs/%s (local mirror, 14 flights, .json.gz read directly), results/labels/<flight>.json, '
         'rule re-implemented here (flat plane_frac >= 0.85; fail = no pose / > 25 cm / > 2 deg; 20-frame blocks). No kpbench.gate / stats '
         'code imported. Compared to %s.' % (CAND, INC, os.path.relpath(GATE, ROOT).replace('\\', '/')), '',
         '| number | recomputed | published | check |', '|---|---|---|---|']
    L += ['| %s | %s | %s | %s |' % (c[0], c[1], c[2], 'PASS' if c[3] else 'FAIL') for c in chk]
    L += ['', 'Seed check: flat diff block CI with seed 1 = %+.2f [%+.2f, %+.2f] (seed 0 is the gate default).' % fd1,
          'Orientation: gate line 3 / 5 / 5 = candidate (pixel dpp2d) better / worse / tied; "5 / 3 / 5 for 3D" in the 10 Nov note is the same split.',
          '', '**Verdict: %s** (%d / %d numbers). %s' % ('PASS' if ok else 'FAIL', sum(c[3] for c in chk), len(chk),
          'The [P] tag on +2.75 [+1.08, +4.90] can be lifted at table+row level.' if ok else 'Freeze +2.75 and the FRESH-2 read until explained.')]
    os.makedirs(os.path.dirname(a.out_audit), exist_ok=True); open(a.out_audit, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    # J1
    d_cnt = dc - di; d_pt = pc - pi
    s1c, s2c, nc = signflip(d_cnt); s1p, s2p, npat = signflip(d_pt)
    from scipy.stats import wilcoxon
    w1 = float(wilcoxon(d_pt[d_pt != 0], alternative='greater').pvalue); w2 = float(wilcoxon(d_pt[d_pt != 0]).pvalue)
    loo = [(per[k][0], int(d_cnt[k]), int(d_cnt.sum() - d_cnt[k]), 100 * (d_cnt.sum() - d_cnt[k]) / (n.sum() - n[k])) for k in range(len(per))]
    J = ['# J1: flight-level tests for pixel dpp2d120a1 vs 3D dpp0.5, FRESH-1 K100 (engineer 2026-11-12, laptop, mirrored rows, 0 GPU-h)', '',
         'd = dpp2d120a1 minus dpp0.5 per flight (positive = 3D dpp0.5 fails less). 13 flights with flat frames, 8 untied, 2075 flat. '
         'Exact sign-flip = all 2^8 sign patterns of the untied flights (tied flights add 0 to the sum, so 2^13 patterns give the same p). '
         'Script: tools/r30_dpp2d_recheck.py.', '',
         '| test (H1: 3D fails less) | statistic | one-sided p | two-sided p |', '|---|---|---|---|',
         '| frame block bootstrap (gate, 20-frame blocks) | +2.75 pt [%+.2f, %+.2f] | lower bound > 0 | - |' % fd[1:],
         '| flight-cluster bootstrap 2000 x seed 22 | %+.2f pt [%+.2f, %+.2f] | lower bound > 0 | - |' % cl,
         '| sign test (flights) | 3D better on %d, worse on %d, %d tied | %.3f | %.3f |' % (worse, better, tied, sum(comb(better + worse, i) for i in range(worse, better + worse + 1)) / 2 ** (better + worse), sp2),
         '| Wilcoxon signed-rank (fail-rate pt) | 8 untied flights | %.3f | %.3f |' % (w1, w2),
         '| exact sign-flip, fail counts | sum d = %+d fails | %.4f (%d/%d) | %.4f (%d/%d) |' % (d_cnt.sum(), s1c, round(s1c * nc), nc, s2c, round(s2c * nc), nc),
         '| exact sign-flip, fail-rate pt | sum d = %+.2f pt | %.4f (%d/%d) | %.4f (%d/%d) |' % (d_pt.sum(), s1p, round(s1p * npat), npat, s2p, round(s2p * npat), npat),
         '', '## Leave one flight out (net fail-count difference, flat)', '', '| flight left out | its d | net d without it | pooled pt without it |', '|---|---|---|---|']
    J += ['| %s | %+d | %+d | %+.2f |' % x for x in loo]
    big = max(loo, key=lambda x: x[1])
    J += ['', 'Largest single flight: %s carries %+d of the net %+d fails (%.0f %%); without it the net is %+d.' % (big[0], big[1], d_cnt.sum(), 100 * big[1] / d_cnt.sum(), big[2]),
          '', '**Reading (dispatch kill):** sign-flip two-sided p = %.3f (counts) and %.3f (rate pt): %s' % (s2c, s2p,
          'both > 0.05 -> kill met: "flight-level borderline" becomes "not significant at flight level"; the flight-level claim rests on the one pre-registered FRESH-2 read.'
          if min(s2c, s2p) > 0.05 else 'at least one <= 0.05 -> flight-level evidence present in that unit.')]
    open(a.out_j1, 'w', encoding='utf-8').write('\n'.join(J) + '\n')
    print('\n'.join(L)); print(); print('\n'.join(J))


if __name__ == '__main__':
    main()
