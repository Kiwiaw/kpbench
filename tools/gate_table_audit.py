"""SUB2 laptop table audit of gate jsons (no rows): whole counts and the verdict, recomputed from the json alone.

  python tools/gate_table_audit.py <gate.json> [...] [--out results/audit/x.md]

Checks per file: flat_cand / flat_inc x n_flat and all_diff x n are whole frame counts; flat diff = cand - inc; sign_p = binomial
of better / worse; rep5 means x n x K are whole; route A, B, relative guard, all-frames guard
and win follow from the stored interval by the rule of kpbench.stats. Row level (does the table match the rows) stays with
kpbench.audit C25 / C37 on Euler: TABLE-PASS is not a row audit.
"""
import argparse, json, os, sys
from math import comb


def sign(b, w):
    n = b + w
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n


def whole(x, eps=1e-6):
    return abs(x - round(x)) < eps


def audit(g):
    bad = []; n, nf = g['n'], g['n_flat']; c, i = g['flat_cand'] * nf / 100, g['flat_inc'] * nf / 100
    if not whole(c) or not whole(i):
        bad.append('flat counts %.4f / %.4f not whole' % (c, i))
    if abs(g['flat_cand'] - g['flat_inc'] - g['flat_diff'][0]) > 1e-9:
        bad.append('flat diff != cand - inc')
    if not whole(g['all_diff'][0] * n / 100):
        bad.append('all-frames diff %.4f frames not whole' % (g['all_diff'][0] * n / 100))
    b, w = g['flights_better'], g['flights_worse']
    if abs(sign(b, w) - g['sign_p']) > 1e-9:
        bad.append('sign_p')
    A, B, R, G = g['flat_diff'][2] < 0, sign(b, w) <= 0.025, g['rel_drop'] >= 0.10, g['all_diff'][2] <= 0.2
    for k, v in (('route_A', A), ('route_B', B), ('guard_rel', R), ('guard_all', G), ('win', (A or B) and R and G)):
        if g[k] != v:
            bad.append('%s stored %s, rule %s' % (k, g[k], v))
    r = {}
    for name, e in (g.get('rep5') or {}).items():
        r[name] = (e['n'], e['diff'], e['better'], e['worse'])
        q = g.get('K', 100)   # rep5 is a share of the K picks
        if not whole(e['cand'] * e['n'] * q, 1e-4) or not whole(e['inc'] * e['n'] * q, 1e-4):
            bad.append('rep5 %s means not whole counts' % name)
        if abs(e['cand'] - e['inc'] - e['diff'][0]) > 1e-9:
            bad.append('rep5 %s diff' % name)
    return dict(picker=g.get('picker'), K=g.get('K', 100), split=g['split'], cand=round(c), inc=round(i), nf=nf, n=n, all_frames=round(g['all_diff'][0] * n / 100),
                diff=g['flat_diff'], all_diff=g['all_diff'], fbw=(b, w, g['flights_tied']), win=g['win'], rep5=r, bad=bad)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--out', default=None); a = ap.parse_args()
    out = ['| gate | K | split | flat frames cand/inc of n | flat diff | all-frames frames of n | flights b/w/t | rep5 flat (n) | table |', '|---|---|---|---|---|---|---|---|---|']
    rc = 0
    for fn in a.files:
        r = audit(json.load(open(fn))); rc |= bool(r['bad']); e = r['rep5'].get('flat')
        out.append('| %s | %d | %s | %d/%d of %d | %+.2f [%+.2f,%+.2f] | %+d of %d, %+.2f [%+.2f,%+.2f] | %d/%d/%d | %s | %s |' % (
            r['picker'], r['K'], r['split'], r['cand'], r['inc'], r['nf'], *r['diff'], r['all_frames'], r['n'], *r['all_diff'], *r['fbw'],
            '%+.4f [%+.4f,%+.4f] %d/%d (%d)' % (*e[1], e[2], e[3], e[0]) if e else '-', 'FROZEN: ' + '; '.join(r['bad']) if r['bad'] else 'TABLE-PASS'))
    print('\n'.join(out))
    if a.out:
        open(a.out, 'w').write('\n'.join(out) + '\n')
    sys.exit(rc)


if __name__ == '__main__':
    main()
