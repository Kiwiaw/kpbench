"""Leaderboard summary for the morning board: every complete 31-flight LightGlue run at one (det, K, map).

  python -m kpbench.leaderboard [--det raco --K 100 --map same]   -> results/leaderboard.json

Per run: fail % on all frames, flat (plane_frac >= 0.85) and very flat (>= 0.95) with 20-frame block-bootstrap 95 %
intervals, mean repeatability at lag 5, devices seen (per-flight device field; 'unknown' for flights stored before it).
"""
import os, json, gzip, datetime
import numpy as np
from .data import FLIGHTS, slug
from .cache import REPO
from .stats import fail_flags, block_ids, boot_ci
from .bench import COLS

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='raco'); ap.add_argument('--K', type=int, default=100); ap.add_argument('--map', default='same')
    a = ap.parse_args(argv)
    lab = {f: json.load(open(os.path.join(REPO, 'results', 'labels', slug(f) + '.json'))) for f in FLIGHTS}
    suffix = '_K%d_%s_lg' % (a.K, a.map); runs = []
    for rid in sorted(os.listdir(os.path.join(RESULTS, 'runs'))):
        if not (rid.startswith(a.det + '_') and rid.endswith(suffix)):
            continue
        rows, pf, devs = [], [], set()
        for f in FLIGHTS:
            fn = os.path.join(RESULTS, 'runs', rid, slug(f) + '.json.gz')
            if not os.path.isfile(fn):
                break
            j = json.load(gzip.open(fn, 'rt')); rows += j['rows']; devs.add(j.get('device') or 'unknown')
            pf += [lab[f][str(r[1])].get('plane_frac') or 0 for r in j['rows']]
        else:
            F = fail_flags(rows); pf = np.array(pf); b = block_ids([r[0] for r in rows], [r[1] for r in rows])
            rep = [r[COLS.index('rep5')] for r in rows if r[COLS.index('rep5')] is not None]
            e = dict(id=rid, picker=rid[len(a.det) + 1:-len(suffix)], n=len(rows), devices=sorted(devs),
                     rep5=round(float(np.mean(rep)), 4) if rep else None)
            for k, m in (('all', np.ones(len(F), bool)), ('flat', pf >= 0.85), ('vflat', pf >= 0.95)):
                p, lo, hi = boot_ci(F[m], b[m]); e[k] = [round(p, 3), round(lo, 3), round(hi, 3)]; e['n_' + k] = int(m.sum())
            runs.append(e)
    runs.sort(key=lambda e: e['all'][0])
    out = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'), det=a.det, K=a.K, map=a.map, matcher='lg', runs=runs)
    json.dump(out, open(os.path.join(RESULTS, 'leaderboard.json'), 'w'), indent=1)
    for e in runs:
        print('%-14s all %.2f  flat %.2f  vflat %.2f  rep5 %s' % (e['picker'], e['all'][0], e['flat'][0], e['vflat'][0], e['rep5']))


if __name__ == '__main__':
    main()
