"""SUB3 T4 power paragraph for a dq3 vs cube0.5 FRESH-2 arm, from the mirrored FRESH-1 pairs (engineer 2026-11-12, 0 GPU-h).

  python tools/dq3_power.py [--append results/sub3/dq3_micro.md]

Per mirrored FRESH-1 pair: flat discordant-frame rate, the gate's 20-frame block bootstrap half-width at 2075 flat (1000 x seed 0),
the iid (McNemar) half-width 1.96 sqrt(disc / n), their ratio (block inflation), and the flat frames needed for 80 % power to see
a 1 pt paired difference at two-sided 5 %: n = 2075 (hw x 2.80 / 1.96)^2 (half-width scales as 1 / sqrt(n)). Kill (dispatch 16:25):
needed > 4849 (FRESH-1 + FRESH-2) -> drop the dq3 arm.
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT)
from kpbench.gate import load_run, labels, split_info
from kpbench.stats import fail_flags, block_ids, paired_diff
from kpbench.splits import FLAT

PAIRS = [('raco_cube0.5_K100_same_lg_g-cube0.5-fresh', 'raco_sd_K100_same_lg_g-cube0.5-fresh', 'cube0.5 vs sd K100'),
         ('raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh', 'raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh', 'dpp2d120a1 vs dpp0.5 K100'),
         ('raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', 'raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400', 'cube0.5 vs dpp0.5 K400'),
         ('raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', 'raco_sd_K400_same_lg_g-cube0.5-fresh-K400', 'cube0.5 vs sd K400'),
         ('raco_dpp0.317a1.6_K100_same_lg_g-dpp0.317a1.6-fresh', 'raco_dpp0.5_K100_same_lg_g-dpp0.317a1.6-fresh', 'dpp0.317a1.6 vs dpp0.5 K100'),
         ('raco_dppr0.317a1.6b2.01t0.9_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', 'raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', 'dppr0.317 vs dpp0.5 K100')]


def pair(c, i, flights):
    _, rc = load_run(c, flights); _, ri = load_run(i, flights); A, B, F, Q, FL = [], [], [], [], []
    for f in flights:
        a, b = rc[f][0], ri[f][0]; assert [r[1] for r in a] == [r[1] for r in b]; lab = labels(f)
        A += list(fail_flags(a)); B += list(fail_flags(b)); F += [f] * len(a); Q += [r[1] for r in a]
        FL += [(lab[str(r[1])].get('plane_frac') or -1) >= FLAT for r in a]
    A, B, FL = np.array(A), np.array(B), np.array(FL, bool); bl = block_ids(F, Q)
    d = paired_diff(A[FL], B[FL], bl[FL]); n = FL.sum(); disc = float((A[FL] != B[FL]).mean())
    hw = (d[2] - d[1]) / 2; iid = 100 * 1.96 * np.sqrt(disc / n)
    return dict(n=int(n), fc=100 * A[FL].mean(), fi=100 * B[FL].mean(), d=d, disc=disc, hw=hw, iid=iid, need=int(round(n * (hw * 2.80 / 1.96) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--append', default=os.path.join(ROOT, 'results', 'sub3', 'dq3_micro.md')); a = ap.parse_args()
    flights = split_info('fresh')[0]; R = []
    for c, i, name in PAIRS:
        if os.path.isdir(os.path.join(ROOT, 'results', 'runs', c)) and os.path.isdir(os.path.join(ROOT, 'results', 'runs', i)):
            R.append((name, pair(c, i, flights)))
    L = ['', '## Power paragraph for a dq3 vs cube0.5 FRESH-2 arm (T4 step 2, 0 GPU-h, mirrored FRESH-1 rows, tools/dq3_power.py)', '',
         '| FRESH-1 pair | flat n | fail % cand / inc | diff [block 95 %] | discordant frames | block half-width | iid half-width | inflation | flat frames for 80 % power at 1 pt |',
         '|---|---|---|---|---|---|---|---|---|']
    for name, r in R:
        L.append('| %s | %d | %.2f / %.2f | %+.2f [%+.2f, %+.2f] | %.2f %% | %.2f pt | %.2f pt | %.2f | %d |' % (name, r['n'], r['fc'], r['fi'], *r['d'], 100 * r['disc'], r['hw'], r['iid'], r['hw'] / r['iid'], r['need']))
    need = sorted(r['need'] for _, r in R); lo, hi = need[0], need[-1]; med = int(np.median(need))
    L += ['', 'Needed flat frames for a 1 pt dq3 vs cube0.5 difference at 80 %% power: %d to %d over the %d mirrored pairs (median %d). '
          'Pool: FRESH-2 ~2774, FRESH-1 + FRESH-2 4849. Pairs at or below 4849: %d of %d.' % (lo, hi, len(R), med, sum(n <= 4849 for n in need), len(R)),
          'Which row fits dq3 vs cube0.5 is not known (no picks or fail rows for dq3 yet; MICRO pick Jaccard 0.291 says the pick sets differ more than '
          'cube vs dpp, so the pairs with a score-ranked arm (vs sd) are the closer analogues).',
          '**Reading:** kill rule "needed > 4849 -> drop the dq3 arm" applied to the median pair: %s' % ('MET -> drop the dq3 arm from FRESH-2.' if med > 4849 else 'not met -> the dq3 arm can stay in the FRESH-2 list, powered for ~1 pt only on the pooled 4849 if its discordance is like the median pair.')]
    open(a.append, 'a', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
