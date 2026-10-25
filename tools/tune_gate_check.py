"""Laptop audit: does every DEV-gated picker's tune trial fail % equal its gate fail %? (SUB2, dispatch 2026-10-25 16:20)

  python tools/tune_gate_check.py --tune <dir of tune jsons> --gates <dir of gate jsons> [--out results/audit/2026-10-25_laptop.md]

Per DEV gate json: cand flat fail (gate rerun, cand + inc on one node) vs every tune trial with the same picker name, the
incumbent flat fail vs 14.42 (dpp0.5 on 936 DEV flat), and integer counts (fail % x n_flat / 100 must be a whole number).
A gap > 0.01 pt is FLAG unless the tune trial has the same whole fail count over 936 or 937 flat (pre-sync tune trials used
937, audit C25): then it is DENOM (same failures, other denominator). Tune trials run on other nodes than the gate, so device
noise (audit C4, ~1 pt) could explain a count FLAG; a non-integer count cannot.
"""
import argparse, glob, json, os

INC = 14.423076923076922    # dpp0.5, 135 of 936 DEV flat


def tcount(pct):
    """whole fail count of a tune trial over 936 or 937 flat queries (the denominator that gives an integer), else None"""
    for n in (936, 937):
        c, ok = count(pct, n)
        if ok or abs(c - round(c)) < 0.01:
            return round(c), n
    return None


def count(pct, n):
    c = pct * n / 100.0
    return c, abs(c - round(c)) < 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', required=True); ap.add_argument('--gates', required=True)
    ap.add_argument('--out', default='results/audit/tune_gate_check.md'); ap.add_argument('--tol', type=float, default=0.01)
    a = ap.parse_args(argv)
    trials = {}
    for f in sorted(glob.glob(os.path.join(a.tune, '*.json'))):
        d = json.load(open(f))
        for t in d.get('trials', []):
            trials.setdefault(t['picker'], []).append((d['study'], t['fail_flat']))
    rows, flags, denoms = [], 0, 0
    for f in sorted(glob.glob(os.path.join(a.gates, '*__dev.json'))):
        g = json.load(open(f)); p, n = g.get('picker'), g.get('n_flat')
        if not p or n is None:
            continue
        cc, ci = count(g['flat_cand'], n); ic, ii = count(g['flat_inc'], n)
        tt = trials.get(p, [])
        gap = min((abs(v - g['flat_cand']) for _, v in tt), default=None)
        inc_ok = abs(g['flat_inc'] - INC) <= a.tol if n == 936 else None
        same = tt and all(tcount(v) and tcount(v)[0] == round(cc) for _, v in tt)
        flag = (gap is not None and gap > a.tol and not same) or not (ci and ii) or inc_ok is False
        denom = not flag and gap is not None and gap > a.tol
        flags += flag; denoms += denom
        rows.append('| %s | %d | %.3f (%d) | %.3f (%d) | %s | %s | %s | %s |' % (
            p, n, g['flat_cand'], round(cc), g['flat_inc'], round(ic),
            ', '.join('%s %.3f (%s)' % (st, v, '%d/%d' % tcount(v) if tcount(v) else '?') for st, v in tt) or '-', '-' if gap is None else '%.3f' % gap,
            {True: 'yes', False: 'NO', None: 'n/a (937)'}[inc_ok], 'FLAG' if flag else 'DENOM' if denom else 'PASS'))
    md = ['# Tune vs DEV gate check (laptop)', '',
          'Gate = cand and inc rerun on one node; tune = trial value on its own node. Counts = fail % x n_flat / 100 (whole number).',
          'inc = dpp0.5 gate flat fail; expected %.2f %% (135 of 936) when n_flat = 936.' % INC, '',
          '| picker | n_flat | gate cand % (count) | gate inc % (count) | tune trials (study fail %, count/n) | min gap pt | inc = 14.42 | verdict |',
          '|---|---|---|---|---|---|---|---|'] + rows + ['', '%d gates, %d FLAG, %d DENOM (same count, 937 vs 936 flat), tol %.2f pt' % (len(rows), flags, denoms, a.tol)]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
