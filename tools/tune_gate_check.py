"""Laptop audit: does every DEV-gated picker's tune trial fail % equal its gate fail %? (SUB2, dispatch 2026-10-25 16:20)

  python tools/tune_gate_check.py --tune <dir of tune jsons> --gates <dir of gate jsons> [--out results/audit/2026-10-25_laptop.md]

Per DEV gate json: cand flat fail (gate rerun, cand + inc on one node) vs every tune trial with the same picker name, the
incumbent flat fail vs 14.42 (dpp0.5 on 936 DEV flat), and integer counts (fail % x n_flat / 100 must be a whole number).
A gap > 0.01 pt is FLAG unless the tune trial has the same whole fail count over 936 or 937 flat (pre-sync tune trials used
937, audit C25): then it is DENOM (same failures, other denominator). Tune trials run on other nodes than the gate, so device
noise (audit C4, ~1 pt) could explain a count FLAG; a non-integer count cannot.

K-aware (dispatch 2026-10-26 13:15, K200 gate acceptance): a gate is matched only to tune trials at its own keypoint budget
(gate json K, else the _K<k> of its run id, else 100; tune json K, else the _K<k> of the study name, else 100), and dpp<s>a1 =
dpp<s>. The inc reference is 14.42 for dpp0.5 at K100, otherwise the inc picker's own tune trials at that K (e.g. the K200
acceptance gate dpp0.5 vs cube0.5 must read 137 and 132 of 936, the dpp_K200_dev1 / cube_K200_dev1 seed arms).
"""
import argparse, glob, json, os, re

INC = 14.423076923076922    # dpp0.5, 135 of 936 DEV flat, K100


def canon(p):
    """picker name with the default alpha dropped (dpp0.5a1 = dpp0.5)"""
    return re.sub(r'^(dpp[\d.]+)a1$', r'\1', p or '')


def kof(name, default=100):
    m = re.search(r'_K(\d+)', name or '')
    return int(m.group(1)) if m else default


def run_picker(rid):
    """picker of a run id raco_<picker>_K<k>_same_lg_<tag>"""
    m = re.match(r'[^_]+_(.+?)_K\d+_', rid or '')
    return m.group(1) if m else None


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
        d = json.load(open(f)); K = d.get('K') or kof(d['study'])
        for t in d.get('trials', []):
            trials.setdefault((canon(t['picker']), K), []).append((d['study'], t['fail_flat']))
    rows, flags, denoms = [], 0, 0
    for f in sorted(glob.glob(os.path.join(a.gates, '*__dev.json'))):
        g = json.load(open(f)); p, n = g.get('picker'), g.get('n_flat')
        if not p or n is None:
            continue
        K = g.get('K') or kof(g.get('cand')); ip = canon(run_picker(g.get('inc')))
        cc, ci = count(g['flat_cand'], n); ic, ii = count(g['flat_inc'], n)
        tt = trials.get((canon(p), K), [])
        gap = min((abs(v - g['flat_cand']) for _, v in tt), default=None)
        it = trials.get((ip, K), [])
        if ip == 'dpp0.5' and K == 100:
            inc_ok = abs(g['flat_inc'] - INC) <= a.tol if n == 936 else None
        else:
            inc_ok = all(tcount(v) and tcount(v)[0] == round(ic) for _, v in it) if it and n == 936 else None
        same = tt and all(tcount(v) and tcount(v)[0] == round(cc) for _, v in tt)
        flag = (gap is not None and gap > a.tol and not same) or not (ci and ii) or inc_ok is False
        denom = not flag and gap is not None and gap > a.tol
        flags += flag; denoms += denom
        rows.append('| %s | %d | %s | %d | %.3f (%d) | %.3f (%d) | %s | %s | %s | %s |' % (
            p, K, ip, n, g['flat_cand'], round(cc), g['flat_inc'], round(ic),
            ', '.join('%s %.3f (%s)' % (st, v, '%d/%d' % tcount(v) if tcount(v) else '?') for st, v in tt) or '-', '-' if gap is None else '%.3f' % gap,
            {True: 'yes', False: 'NO', None: 'n/a'}[inc_ok], 'FLAG' if flag else 'DENOM' if denom else 'PASS'))
    md = ['# Tune vs DEV gate check (laptop)', '',
          'Gate = cand and inc rerun on one node; tune = trial value on its own node, same K only. Counts = fail % x n_flat / 100 (whole number).',
          'inc reference: dpp0.5 at K100 = %.2f %% (135 of 936); other inc pickers / K: the inc picker\'s own tune trials at that K (936 flat only).' % INC, '',
          '| picker | K | inc | n_flat | gate cand % (count) | gate inc % (count) | tune trials (study fail %, count/n) | min gap pt | inc = reference | verdict |',
          '|---|---|---|---|---|---|---|---|---|---|'] + rows + ['', '%d gates, %d FLAG, %d DENOM (same count, 937 vs 936 flat), tol %.2f pt' % (len(rows), flags, denoms, a.tol)]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
    return flags


if __name__ == '__main__':
    main()
