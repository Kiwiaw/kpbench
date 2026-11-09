"""148 = 148 audit, laptop half (SUB2 r29 prep, dispatch 2026-11-09 13:12): everything the Euler half needs from the local rows.

  python tools/failset_prep.py [--run raco_sd_K100_same_lg_g-cube0.5-fresh]
                               [--k400 raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400,raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400]
                               [--out results/audit/sd_K100_failset_<date>.json] [--md results/audit/r29_148_prep_<date>.md]

From the mirrored sd K100 rows (E1 run) and the repo FRESH-1 labels: the flat fail frame ids (expected 148), sha256 + bytes of
every flight row file, and the realised K per row (column n_sel) on the flat frames; the same n_sel histogram for the mirrored
K400 runs (they should sit at ~400 if the K400 jobs honoured the budget). Once the sd K400 rows are mirrored, the Euler half is
one Jaccard of the two fail sets plus the same n_sel histogram, scored by the pre-registered rules written into the md.
"""
import argparse, gzip, hashlib, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT)
from kpbench.stats import fail_flags
from kpbench.gate import split_info, labels, slug, RESULTS
from kpbench.splits import FLAT


def run_rows(rid, flights):
    d = os.path.join(RESULTS, 'runs', rid); meta = json.load(open(os.path.join(d, 'meta.json'))); out = {}; files = {}
    for f in flights:
        fn = os.path.join(d, slug(f) + '.json.gz'); b = open(fn, 'rb').read()
        files[slug(f)] = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        out[f] = json.loads(gzip.decompress(b))['rows']
    return meta, out, files


def flat_stats(rid, flights):
    meta, rows, files = run_rows(rid, flights); cols = meta['cols']; isel = cols.index('n_sel') if 'n_sel' in cols else None
    fails, nsel, nflat = [], [], 0
    for f in flights:
        lab = labels(f); r = rows[f]; ff = fail_flags(r)
        flat = np.array([(lab[str(x[1])].get('plane_frac') or -1) >= FLAT for x in r], bool); nflat += int(flat.sum())
        fails += ['%s:%d' % (f, x[1]) for x, fl, fa in zip(r, flat, ff) if fl and fa]
        if isel is not None:
            nsel += [x[isel] for x, fl in zip(r, flat) if fl]
    ns = np.array(nsel, float) if nsel else None
    hist = {str(int(k)): int(v) for k, v in zip(*np.unique(ns, return_counts=True))} if ns is not None else None
    nstat = dict(min=float(ns.min()), max=float(ns.max()), mean=float(ns.mean()), le100=int((ns <= 100).sum()), hist=hist) if ns is not None else None
    return dict(run=rid, K=meta.get('K'), picker=meta.get('picker'), date=meta.get('date'), commit=meta.get('commit'), device=meta.get('device'),
                has_picks=meta.get('has_picks'), rows=sum(len(v) for v in rows.values()), flat_rows=nflat, flat_fails=len(fails), fail_ids=fails,
                n_sel_col=isel is not None, n_sel_flat=nstat, files=files)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='raco_sd_K100_same_lg_g-cube0.5-fresh')
    ap.add_argument('--k400', default='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400,raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400')
    d = time.strftime('%Y-%m-%d')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'audit', 'sd_K100_failset_%s.json' % d))
    ap.add_argument('--md', default=os.path.join(ROOT, 'results', 'audit', 'r29_148_prep_%s.md' % d)); a = ap.parse_args()
    flights = split_info('fresh')[0]
    sd = flat_stats(a.run, flights); others = [flat_stats(r, flights) for r in a.k400.split(',') if r]
    json.dump(dict(generated=time.strftime('%Y-%m-%d %H:%M'), split='fresh', flights=flights, sd_K100=sd, k400_runs=others), open(a.out, 'w'), indent=1)
    rel = os.path.relpath(a.out, ROOT).replace(os.sep, '/')
    ns = sd['n_sel_flat']
    L = ['# 148 = 148 audit, laptop half (%s)' % time.strftime('%Y-%m-%d %H:%M'), '',
         'Question: sd flat fails on FRESH-1 are 148 at K100 (E1 run, 1 Nov) and 148 at K400 (Qg-cube0.5-vs-sd-K400-fresh, jobs 20220547/48, 8 Nov) while rep5 differs (0.583 vs 0.384). Saturation (sd never has more than 100 depth candidates) or row reuse? The sd K400 rows exist only on Euler (ssh timeout 9 Nov 13:00); this file holds the local half.', '',
         '## sd K100 (E1 run) from the mirrored rows + repo labels', '',
         '- run %s: %d rows, %d flat, **%d flat fails** (fail = no pose, > 25 cm or > 2 deg); fail ids in %s' % (sd['run'], sd['rows'], sd['flat_rows'], sd['flat_fails'], rel),
         '- meta: K %s, date %s, device %s, has_picks %s, commit %r' % (sd['K'], sd['date'], sd['device'], sd['has_picks'], sd['commit'])]
    if ns:
        L.append('- realised K on the flat rows (column n_sel): min %g, max %g, mean %.2f, rows with n_sel <= 100: %d of %d' % (ns['min'], ns['max'], ns['mean'], ns['le100'], sd['flat_rows']))
    else:
        L.append('- no n_sel column')
    L += ['', '| flight file | bytes | sha256 (first 16) |', '|---|---|---|'] + ['| %s | %d | %s |' % (k, v['bytes'], v['sha256'][:16]) for k, v in sd['files'].items()]
    L += ['', '## Mirrored K400 runs (same job family as the sd K400 arm)', '', '| run | K | rows | flat | flat fails | n_sel on flat: min / max / mean | n_sel <= 100 |', '|---|---|---|---|---|---|---|']
    for o in others:
        n2 = o['n_sel_flat']
        if n2:
            L.append('| %s | %s | %d | %d | %d | %g / %g / %.1f | %d |' % (o['run'], o['K'], o['rows'], o['flat_rows'], o['flat_fails'], n2['min'], n2['max'], n2['mean'], n2['le100']))
        else:
            L.append('| %s | %s | %d | %d | %d | no n_sel | - |' % (o['run'], o['K'], o['rows'], o['flat_rows'], o['flat_fails']))
    L += ['', 'Rows DO carry a per-row realised-K column (n_sel = picks handed to the matcher), so realised K needs no picks rerun (T33 is not required for this audit).', '',
          '## Pre-registered verdict rules for the Euler half (one Jaccard + one n_sel histogram once the sd K400 rows are mirrored)', '',
          '1. Jaccard(sd K100 fail set, sd K400 fail set) = 1.0 AND sd K400 n_sel <= 100 on every flat row -> saturation: the sd K400 cell stands, tag [V].',
          '2. Jaccard = 1.0 AND (identical sha256 of any flight row file OR sd K400 n_sel == sd K100 n_sel row by row while cube0.5 K400 n_sel ~ 400) -> row reuse: sd K400 void (57 vs 148, -4.39 [-7.01,-2.01] struck), and the cube0.5 K400 57 is re-checked the same way.',
          '3. Jaccard < 1.0 -> the two 148 are different frame sets: a recount error somewhere, both sd cells frozen until the recount agrees.',
          '', 'Commands on Euler (login node, CPU): `python tools/failset_prep.py --run raco_sd_K400_same_lg_g-cube0.5-fresh-K400 --k400 "" --out /tmp/sd_K400.json --md /tmp/sd_K400.md`, then Jaccard of fail_ids against %s; or mirror the two sd K400 run dirs with euler/pull_runs.sh and run the same here.' % rel]
    open(a.md, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L[:9])); print('...'); print('\n'.join(L[-10:-6])); print('->', a.md)


if __name__ == '__main__':
    main()
