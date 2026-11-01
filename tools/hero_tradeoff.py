"""SUB4 hero trade-off table: flat-fail gain and rep5 cost side by side, from audited gate jsons + the leaderboard (no rows).

  python tools/hero_tradeoff.py <gate.json> [...] [--leaderboard results/leaderboard.json] [--out results/sub4/hero_tradeoff_<date>.md] [--stamp PROVISIONAL]

One row per gate json: candidate vs incumbent, split, K, flat fail % (frames of n), paired flat diff [95 %], flights better/worse/tied,
sign p, relative drop, verdict, paired rep5 on flat frames (diff [95 %], flights better/worse) and on all queries. Then the 31-flight
leaderboard rep5 column, so the repeatability cost of the hero pair stands next to its fail-% gain. Numbers come straight from the
jsons (table level, kpbench.audit C25 / C37 on Euler is the row audit); --stamp marks every table until that audit passes.
"""
import argparse, json, os, sys, time

ROOT = os.path.join(os.path.dirname(__file__), '..')


def row(g):
    nf, n = g['n_flat'], g['n']; r = g.get('rep5') or {}
    def rp(e):
        return '%+.4f [%+.4f,%+.4f] %d/%d (n %d)' % (e['diff'][0], e['diff'][1], e['diff'][2], e['better'], e['worse'], e['n']) if e and 'diff' in e else '-'
    return '| %s vs %s | %s | %d | %.2f (%d) vs %.2f (%d) of %d | %+.2f [%+.2f,%+.2f] | %d/%d/%d p %.2f | %+.1f %% | %s | %s | %s |' % (
        g.get('picker', g['cand']), g['inc'].split('_')[1], g['split'], g.get('K', 100), g['flat_cand'], round(g['flat_cand'] * nf / 100), g['flat_inc'],
        round(g['flat_inc'] * nf / 100), nf, *g['flat_diff'], g['flights_better'], g['flights_worse'], g['flights_tied'], g['sign_p'],
        -100 * g['rel_drop'], 'KEEP' if g['win'] else 'REVERT', rp(r.get('flat')), rp(r.get('all')))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--leaderboard', default=os.path.join(ROOT, 'results', 'leaderboard.json'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub4', 'hero_tradeoff_%s.md' % time.strftime('%Y-%m-%d'))); ap.add_argument('--stamp', default='PROVISIONAL until T26 (row audit C25 / C37 on Euler)')
    a = ap.parse_args()
    G = [json.load(open(f)) for f in a.files]
    L = ['# Hero trade-off: flat-fail gain next to the rep5 cost (%s)' % time.strftime('%Y-%m-%d'), '', 'Status: %s. Table level from the gate jsons (tools/hero_tradeoff.py); rep5 = share of the K picks re-detected 5 frames on, paired on the same queries.' % a.stamp, '',
         '| gate | split | K | flat fail % (frames) cand vs inc | flat diff pt [95 %] | flights b/w/t | rel | verdict | rep5 flat diff [95 %] b/w | rep5 all diff [95 %] b/w |',
         '|---|---|---|---|---|---|---|---|---|---|'] + [row(g) for g in G]
    lb = json.load(open(a.leaderboard))
    L += ['', '## Leaderboard (%s, 31 flights, K %d, %s): the repeatability column' % (lb['generated'][:10], lb['K'], lb['matcher']), '', '| picker | fail % all | fail % flat | fail % very flat | rep5 |', '|---|---|---|---|---|']
    val = lambda x: x[0] if isinstance(x, (list, tuple)) else x      # leaderboard stores [value, lo, hi]
    for r in sorted(lb['runs'], key=lambda r: val(r['flat'])):
        if r['id'].endswith('_same_lg'):
            L.append('| %s | %.2f | %.1f | %.1f | %.4f |' % (r['picker'], val(r['all']), val(r['flat']), val(r['vflat']), r.get('rep5', float('nan'))))
    L += ['', 'Read: every flat-fail winner in the table pays in rep5 and every rep5 winner pays in flat fail; the paper states the trade-off, not a fix.']
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
