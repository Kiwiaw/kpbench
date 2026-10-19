"""Cross-dataset picker table: fail % (25 cm / 2 deg) on flat and all queries, per group, and the paired block-bootstrap
difference of each run against its own reference run (same queries, same device per flight).

  python tools/xscale_table.py --results <dir> --flights f1,f2,.. --ref <ref_run> RUN[=REF] RUN ... [--group env|flight|all]
  e.g. --results results --flights dev --ref raco_dpp0.5_K100_same_lg_xs raco_dppm0.4a1_K100_same_lg_xs \
       raco_dppm0.256a0.5_K100_same_lg_g-dppm0.256a0.5=raco_dpp0.5_K100_same_lg_g-dppm0.256a0.5

RUN=REF pairs a run with another reference (e.g. a gate pair run on its own nodes); without =REF the --ref run is used.
Labels (plane_frac, flat >= 0.85) come from <results>/labels. Output: markdown rows
picker | group | n flat | flat fail % | flat diff [95 % CI] | n all | all fail % | all diff [95 % CI].
"""
import argparse, gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.stats import fail_flags, paired_diff, block_ids
from kpbench.data import slug

FLAT = 0.85


def load(res, run, flights):
    out = {}
    for f in flights:
        d = json.load(gzip.open(os.path.join(res, 'runs', run, slug(f) + '.json.gz')))
        out[f] = (d['rows'], d.get('device'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True); ap.add_argument('--flights', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--group', default='env', choices=['env', 'flight', 'all']); ap.add_argument('runs', nargs='+')
    a = ap.parse_args(argv)
    if a.flights == 'dev':
        flights = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'splits.json')))['dev']
    else:
        flights = a.flights.split(',')
    lab = {f: json.load(open(os.path.join(a.results, 'labels', slug(f) + '.json'))) for f in flights}
    gkey = {'env': lambda f: f.split('/')[0], 'flight': lambda f: f, 'all': lambda f: 'all'}[a.group]
    print('| picker | group | n flat | flat fail % | flat diff vs ref [95 % CI] | n all | all fail % | all diff vs ref [95 % CI] | device |')
    print('|---|---|---|---|---|---|---|---|---|')
    for spec in [a.ref] + a.runs:
        run, ref = spec.split('=') if '=' in spec else (spec, a.ref)
        R, B = load(a.results, run, flights), load(a.results, ref, flights)
        fa, fb, fl, qs, flat, same = [], [], [], [], [], True
        for f in flights:
            (ra, da), (rb, db) = R[f], B[f]
            assert [r[1] for r in ra] == [r[1] for r in rb], f + ': query lists differ'
            same &= bool(da) and da == db
            fa.append(fail_flags(ra)); fb.append(fail_flags(rb))
            for r in ra:
                pf = lab[f][str(r[1])].get('plane_frac'); fl.append(f); qs.append(r[1]); flat.append(pf is not None and pf >= FLAT)
        fa, fb, fl, qs, flat = np.concatenate(fa), np.concatenate(fb), np.array(fl), np.array(qs), np.array(flat)
        blocks = block_ids(fl, qs); grp = np.array([gkey(f) for f in fl])
        for g in sorted(set(grp)) + (['all'] if a.group != 'all' else []):
            m = np.ones(len(fl), bool) if g == 'all' else grp == g
            cells = []
            for sel in (m & flat, m):
                if not sel.any():
                    cells += ['0', '-', '-']; continue
                d = paired_diff(fa[sel], fb[sel], blocks[sel]) if run != ref else None
                cells += ['%d' % sel.sum(), '%.2f' % (100 * fa[sel].mean()),
                          '-' if d is None else '%+.2f [%+.2f, %+.2f]' % d]
            print('| %s | %s | %s | %s |' % (run.split('_')[1], g, ' | '.join(cells),
                                             'same' if same else 'differs'))


if __name__ == '__main__':
    main()
