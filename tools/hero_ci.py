"""SUB4 E1 pre-check (0 GPU): paired block-bootstrap CI of the very-flat and flat fail gain vs sd over the 31 leaderboard flights.

  python tools/hero_ci.py [--out results/sub4/e1_power.md]

Same-map K100 LightGlue leaderboard rows, 20-frame blocks, 1000 boots seed 0 (kpbench.stats.paired_diff), sign test over flights.
These runs are not gate runs (flights ran on different nodes), so this is a CI for the page numbers, not a gate verdict.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.audit import run_rows, labels
from kpbench.stats import fail_flags, block_ids, paired_diff, per_flight, sign_test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); a = ap.parse_args()
    lab = labels(); R = {p: run_rows('raco_%s_K100_same_lg' % p) for p in ('sd', 'cube0.5', 'dpp0.5')}
    fl = np.array([r[0] for r in R['sd']]); q = [r[1] for r in R['sd']]
    assert all([r[1] for r in R[p]] == q and [r[0] for r in R[p]] == list(fl) for p in R)
    pf = np.array([lab[r[0]][str(r[1])].get('plane_frac') or -1 for r in R['sd']]); F = {p: fail_flags(R[p]) for p in R}; bl = block_ids(fl, q)
    out = ['| pair | band | n | cand % | sd % | diff pt [95 %] | flights better/worse/tied | sign p |', '|---|---|---|---|---|---|---|---|']; res = {}
    for p in ('cube0.5', 'dpp0.5'):
        for band, thr in (('flat', 0.85), ('very flat', 0.95)):
            m = pf >= thr; d = paired_diff(F[p][m], F['sd'][m], bl[m]); fa = F[p][m]; fb = F['sd'][m]; b = w = t = 0
            for f in sorted(set(fl[m])):
                x, y = fa[fl[m] == f].mean(), fb[fl[m] == f].mean(); b += x < y; w += x > y; t += x == y
            res[p, band] = d
            out.append('| %s vs sd | %s | %d | %.2f (%d) | %.2f (%d) | %+.2f [%+.2f, %+.2f] | %d/%d/%d | %.4f |' % (p, band, m.sum(), 100 * fa.mean(), fa.sum(), 100 * fb.mean(), fb.sum(), *d, b, w, t, sign_test(int(b), int(w))))
    print('\n'.join(out))
    if a.out:
        open(a.out, 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
