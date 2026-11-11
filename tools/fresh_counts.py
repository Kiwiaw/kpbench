"""Independent recount of the FRESH-1 flat-fail counts from the mirrored run rows and the repo labels (T11 step, 8 Nov).

  python tools/fresh_counts.py [--out results/main/fresh1_counts.md]

Rows: results/runs/<run>/<flight>.json.gz (euler/pull.sh mirror of the Euler gate runs, 14 FRESH-1 flights). Labels:
results/labels/<flight>.json (plane_frac >= 0.85 = flat). Fail = no pose, > 25 cm or > 2 deg, applied here by hand (not
kpbench.stats). Prints every count next to the r28 audit value (keypoint-gap-agent results/audit/r28_fresh1_flat_labels.md,
4 Nov: K400 cube0.5 57 / dpp0.5 64, K100 dpp0.5 62, E1 cube0.5 95 / sd 148, all of 2075 flat / 13440 queries) and exits 1
on any difference.
"""
import argparse, gzip, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R28 = [('raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', 57, 'K400 cube0.5'), ('raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400', 64, 'K400 dpp0.5'),
       ('raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', 62, 'K100 dpp0.5'), ('raco_dppr0.317a1.6b2.01t0.9_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', 67, 'K100 dppr0.317'),
       ('raco_cube0.5_K100_same_lg_g-cube0.5-fresh', 95, 'E1 cube0.5'), ('raco_sd_K100_same_lg_g-cube0.5-fresh', 148, 'E1 sd'),
       ('raco_dpp0.5_K100_same_lg_g-dpp2d120a1-fresh', 62, 'dpp2d gate inc dpp0.5'), ('raco_dpp2d120a1_K100_same_lg_g-dpp2d120a1-fresh', None, 'dpp2d120a1 (5.73 %)'),
       ('raco_sd_K400_same_lg_g-cube0.5-fresh-K400', 148, 'K400 sd (gate 8 Nov, mirrored 11 Nov)')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'main', 'fresh1_counts.md')); a = ap.parse_args()
    flights = json.load(open(os.path.join(ROOT, 'splits_fresh.json')))['fresh']
    flat = {}
    for f in flights:
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))
        flat[f] = {int(q) for q, v in lab.items() if v.get('plane_frac') is not None and v['plane_frac'] >= 0.85}
    nq = sum(len(json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))) for f in flights)
    nflat = sum(len(v) for v in flat.values())
    L = ['# FRESH-1 flat-fail recount from mirrored rows (tools/fresh_counts.py, laptop)', '',
         'Labels: %d flights, %d queries, %d flat (plane_frac >= 0.85). Fail = no pose or > 25 cm or > 2 deg, counted by hand.' % (len(flights), nq, nflat), '',
         '| run | rows | flat rows | flat fails | flat fail % | r28 | match |', '|---|---|---|---|---|---|---|']
    ok = (nq, nflat) == (13440, 2075)
    for rid, exp, what in R28:
        d = os.path.join(ROOT, 'results', 'runs', rid); n = nf = fails = 0
        for f in flights:
            fn = os.path.join(d, f.replace('/', '_') + '.json.gz')
            if not os.path.isfile(fn):
                continue
            for r in json.load(gzip.open(fn, 'rt'))['rows']:
                n += 1
                if r[1] in flat[f]:
                    nf += 1; fails += r[2] is None or r[2] > 25 or r[3] > 2
        same = n == 13440 and nf == nflat and (exp is None or fails == exp); ok &= same
        L.append('| %s | %d | %d | %d | %.2f | %s | %s |' % (what, n, nf, fails, 100 * fails / nf if nf else float('nan'), exp if exp is not None else '-', 'yes' if same else 'NO'))
    L += ['', '**%s** (labels %d/%d vs 13440/2075)' % ('all counts reproduce r28' if ok else 'MISMATCH: mirror flagged, numbers frozen until reconciled', nq, nflat)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
