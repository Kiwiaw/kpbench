"""Flight-cluster CI (R22) for one mirrored FRESH-1 pair, with the per-flight fail table (MAIN step, dispatch 2026-11-09 13:12).

  python tools/k400_cluster_ci.py [--cand raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400] [--inc raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400]
                                  [--ref raco_cube0.5_K100_same_lg_g-cube0.5-fresh,raco_sd_K100_same_lg_g-cube0.5-fresh] [--boot 2000] [--seed 22]
                                  [--out results/main/k400_cluster_ci_<date>.md]

Uses tools/class_bootstrap.flight_table (rows + repo labels, flat = plane_frac >= 0.85, fail = no pose / > 25 cm / > 2 deg) and its
cluster_ci (flights resampled with replacement, frame-weighted pooled difference in points). The frame-level 20-frame block CI is
quoted from the gate json. Nothing here changes a verdict.
"""
import argparse, glob, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from class_bootstrap import flight_table, cluster_ci
from kpbench.gate import split_info, GATES


def gate_json(cand, inc):
    for f in glob.glob(os.path.join(GATES, '*__fresh.json')):
        g = json.load(open(f))
        if g.get('cand') == cand and g.get('inc') == inc:
            return g
    return None


def block(cand, inc, split, boot, seed):
    t = flight_table(cand, inc, split); g = gate_json(cand, inc)
    if t is None:
        return None, g, None
    dc = np.array(t['pc']) * np.array(t['n']) / 100; di = np.array(t['pi']) * np.array(t['n']) / 100
    return t, g, cluster_ci(dc, di, t['n'], n_boot=boot, seed=seed)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cand', default='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400')
    ap.add_argument('--inc', default='raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400')
    ap.add_argument('--ref', default='raco_cube0.5_K100_same_lg_g-cube0.5-fresh,raco_sd_K100_same_lg_g-cube0.5-fresh')
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=22)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'main', 'k400_cluster_ci_%s.md' % time.strftime('%Y-%m-%d'))); a = ap.parse_args()
    flights = split_info('fresh')[0]
    t, g, ci = block(a.cand, a.inc, 'fresh', a.boot, a.seed)
    if t is None:
        sys.exit('rows missing for %s / %s' % (a.cand, a.inc))
    rc, ri = a.ref.split(','); t2, g2, ci2 = block(rc, ri, 'fresh', a.boot, a.seed)
    fl = [f for f in flights]       # flight_table keeps flights with flat > 0 in split order
    kept = []
    from kpbench.gate import labels
    from kpbench.splits import FLAT
    for f in flights:
        lab = labels(f); n = sum(1 for v in lab.values() if (v.get('plane_frac') or -1) >= FLAT)
        if n: kept.append(f)
    fmt = lambda c: '%+.2f [%+.2f, %+.2f]' % tuple(c)
    L = ['# K400 near miss cube0.5 vs dpp0.5 on FRESH-1: flight-cluster CI next to the frame CI (%s)' % time.strftime('%Y-%m-%d %H:%M'), '',
         'Rows: local mirror of %s and %s (14 FRESH-1 flights, 13440 queries, 2075 flat). Cluster bootstrap: %d resamples of the %d flights with flat frames, seed %d (tools/class_bootstrap.cluster_ci); the default 1000 / seed 0 is given too.' % (a.cand, a.inc, a.boot, t['flights'], a.seed), '',
         '| pair | flat fails cand / inc | frame 20-frame block CI (gate json) | flights better / worse / tied | cluster CI %d x seed %d | cluster CI 1000 x seed 0 | sign p | Wilcoxon p |' % (a.boot, a.seed), '|---|---|---|---|---|---|---|---|']
    for name, tt, gg, cc in (('cube0.5 vs dpp0.5, K400', t, g, ci), ('E1 cube0.5 vs sd, K100 (reference)', t2, g2, ci2)):
        if tt is None: L.append('| %s | rows missing | | | | | | |' % name); continue
        fa = int(round(sum(p * n / 100 for p, n in zip(tt['pc'], tt['n'])))); fb = int(round(sum(p * n / 100 for p, n in zip(tt['pi'], tt['n']))))
        L.append('| %s | %d / %d | %s | %d / %d / %d | %s | %s | %s | %s |' % (name, fa, fb, fmt(gg['flat_diff']) if gg else '?', tt['better'], tt['worse'], tt['tied'], fmt(cc), fmt(tt['cluster']),
                                                                '%.3f' % tt['sign_p'] if tt['sign_p'] is not None else '-', '%.3f' % tt['wilcoxon_p'] if tt['wilcoxon_p'] is not None else '-'))
    hw = (ci[2] - ci[1]) / 2
    L += ['', 'Cluster CI half-width for the K400 pair: %.2f pt (kill line 2 pt: %s).' % (hw, 'MET -> not decidable at flight level on FRESH-1' if hw > 2 else 'not met'), '',
          '## Per-flight flat fails, K400 pair', '', '| flight | n flat | cube0.5 fails | dpp0.5 fails | diff | cube0.5 % | dpp0.5 % |', '|---|---|---|---|---|---|---|']
    for f, pc, pi, n in zip(kept, t['pc'], t['pi'], t['n']):
        a_, b_ = int(round(pc * n / 100)), int(round(pi * n / 100))
        L.append('| %s | %d | %d | %d | %+d | %.2f | %.2f |' % (f, n, a_, b_, a_ - b_, pc, pi))
    L += ['| **total** | **%d** | **%d** | **%d** | **%+d** | | |' % (sum(t['n']), int(round(sum(p * n / 100 for p, n in zip(t['pc'], t['n'])))), int(round(sum(p * n / 100 for p, n in zip(t['pi'], t['n'])))),
                                                                      int(round(sum((p - q) * n / 100 for p, q, n in zip(t['pc'], t['pi'], t['n']))))),
          '', 'Flights with 0 flat frames are not rows (oldtown_easy P004). Descriptive: FRESH-1 has been read 10x, CIs are nominal and unadjusted; the K400 cells stay [T] until the 148 = 148 audit (results/audit/r29_148_prep_2026-11-09.md).']
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
