"""Stage 4: results/ -> docs/data/ (index.json, labels.json, runs/<id>.json, picks/<id>/<slug>.json). See docs/data_spec.md.
Also results_hypersim_w3/ -> docs/data/hypersim.json, a separate Hypersim table (never mixed into the TartanAir data).

  python -m kpbench.export              # everything
  python -m kpbench.export --hypersim   # only docs/data/hypersim.json
Thumbnails are produced separately by kpbench.thumbs (they do not change between runs).
"""
import os, json, gzip, glob, datetime
import numpy as np
from .data import FLIGHTS, slug, KF
from .cache import REPO
from .run import RESULTS

SITE = os.path.join(REPO, 'docs', 'data')
BINS = [dict(id='all', label='all frames'),
        dict(id='ordinary', label='ordinary (<0.55)', lo=-1, hi=0.55),
        dict(id='mid', label='mid (0.55-0.85)', lo=0.55, hi=0.85),
        dict(id='flat', label='flat (>=0.85)', lo=0.85, hi=2),
        dict(id='very_flat', label='very flat (>=0.95)', lo=0.95, hi=2)]
THR = [5, 25, 50, 500]                      # position thresholds in cm
ROT_FAIL = {5: 2, 25: 2, 50: 5, 500: 10}   # rotation threshold (deg) that goes with each: the visuallocalization.net pairs


def load_labels():
    lab = {}
    for fn in glob.glob(os.path.join(RESULTS, 'labels', '*.json')):
        s = os.path.basename(fn)[:-5]
        d = json.load(open(fn)); qs = sorted(int(q) for q in d)
        lab[s] = dict(q=qs, **{k: [d[str(q)].get(k) for q in qs] for k in ('plane_frac', 'med_depth', 'missing_depth', 'coverage')})
    return lab


def main():
    os.makedirs(os.path.join(SITE, 'runs'), exist_ok=True)
    labels = load_labels()
    # flights = every flight that any run or label file covers, in the canonical order
    present = set()
    run_dirs = sorted(glob.glob(os.path.join(RESULTS, 'runs', '*', 'meta.json')))
    per_run = {}
    for mfn in run_dirs:
        meta = json.load(open(mfn)); rd = os.path.dirname(mfn); parts = {}
        for fn in glob.glob(os.path.join(rd, '*.json.gz')):
            d = json.load(gzip.open(fn, 'rt')); parts[d['flight']] = d; present.add(d['flight'])
        per_run[meta['id']] = (meta, parts)
    present |= {f for f in FLIGHTS if slug(f) in labels}
    flights = [f for f in FLIGHTS if f in present]
    fidx = {f: i for i, f in enumerate(flights)}
    finfo = []
    for f in flights:
        s = slug(f); nq = len(labels[s]['q']) if s in labels else None
        if nq is None:
            for meta, parts in per_run.values():
                if f in parts:
                    nq = len(parts[f]['rows']); break
        finfo.append(dict(id=f, slug=s, env=f.split('/')[0], n_queries=nq, n_frames=(nq + (nq // (KF - 1)) + 1) if nq else None))
    runs = []
    for rid, (meta, parts) in per_run.items():
        rows = []; has_picks = False
        for f in flights:
            if f not in parts:
                continue
            for r in parts[f]['rows']:
                rows.append([fidx[f]] + r[1:])
            if parts[f].get('picks'):
                has_picks = True
                pd = os.path.join(SITE, 'picks', rid); os.makedirs(pd, exist_ok=True)
                json.dump(parts[f]['picks'], open(os.path.join(pd, slug(f) + '.json'), 'w'), separators=(',', ':'))
        if not rows:
            continue
        fail25 = float(np.mean([r[2] is None or r[2] > 25 or r[3] > ROT_FAIL[25] for r in rows]))
        cfg = {k: v for k, v in meta.items() if k not in ('cols',)}
        json.dump(dict(id=rid, config=cfg, cols=meta['cols'], rows=rows), open(os.path.join(SITE, 'runs', rid + '.json'), 'w'), separators=(',', ':'))
        runs.append(dict(id=rid, det=meta['det'], picker=meta['picker'], K=meta['K'], map=meta['map'], matcher=meta.get('matcher', 'nn'),
                         date=meta.get('date', ''), commit=meta.get('commit', ''), flights=[f for f in flights if f in parts],
                         n_queries=len(rows), fail25=round(100 * fail25, 2), has_picks=has_picks, notes=meta.get('notes', '')))
    # order: RaCo before SuperPoint, same map before dense, K=100 first, the strongest-with-depth baseline first within a group
    runs.sort(key=lambda r: (r['det'] != 'raco', r['map'] != 'same', r['K'] != 100, r['K'], r['picker'] != 'sd', r['picker']))
    index = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'),
                 dataset='TartanAir (Hard), PnP relocalisation against a local map of ground-truth 3D points',
                 flights=finfo, bins=BINS, thresholds_cm=THR, rot_fail=ROT_FAIL, rot_fail_deg=2, runs=runs)
    json.dump(index, open(os.path.join(SITE, 'index.json'), 'w'), indent=1)
    json.dump({f: labels[slug(f)] for f in flights if slug(f) in labels}, open(os.path.join(SITE, 'labels.json'), 'w'), separators=(',', ':'))
    print('exported', len(runs), 'runs,', len(flights), 'flights ->', SITE)
    export_hypersim()


# Second dataset, kept apart from the TartanAir aggregates: results_hypersim_w3/ -> docs/data/hypersim.json (one precomputed table)
HS_RESULTS = os.path.join(REPO, 'results_hypersim_w3')
HS_SCENES = ['hypersim_ai_001_001', 'hypersim_ai_001_002']
HS_RUNS = ['raco_dpp0.5_K100_same_lg', 'raco_sd_K100_same_lg', 'raco_cube0.5_K100_same_lg']
HS_BASE = 'raco_sd_K100_same_lg'
HS_PROTOCOL = [
    'Apple Hypersim, 2 scenes ai_001_001 and ai_001_002, camera cam_00 (ai_003_001 dropped: its renders are black in the released data)',
    'images 1024x768 resized to 640x480, fx = fy = 554.256 px, cx = 320, cy = 240; planar depth from depth_meters',
    'frames are about 1.1 m apart, so a keyframe every 2nd frame and a query uses the keyframes within +-3 frames (4 keyframes: q-3, q-1, q+1, q+3)',
    'RaCo candidates, K = 100, same map, LightGlue, AP3P RANSAC + LM refinement (as TartanAir)',
    'fail = no pose, position error > 25 cm or rotation error > 2 deg; 98 queries; laptop GPU (RTX 5060 Laptop)',
]


def export_hypersim():
    from .stats import fail_flags, paired_diff, block_ids, boot_ci
    if not os.path.isdir(os.path.join(HS_RESULTS, 'runs')):
        return
    lab = {}
    for s in HS_SCENES:
        d = json.load(open(os.path.join(HS_RESULTS, 'labels', s + '.json')))
        lab.update({(s, int(q)): v.get('plane_frac') for q, v in d.items()})

    def load(rid):
        rows = []
        for s in HS_SCENES:
            rows += [[s] + x[1:] for x in json.load(gzip.open(os.path.join(HS_RESULTS, 'runs', rid, s + '.json.gz'), 'rt'))['rows']]
        return rows
    base = load(HS_BASE); fb = fail_flags(base)
    keys = [(r[0], r[1]) for r in base]; blocks = block_ids([k[0] for k in keys], [k[1] for k in keys])
    pf = np.array([np.nan if lab.get(k) is None else lab[k] for k in keys])
    bins = []
    for b in BINS:
        m = np.ones(len(keys), bool) if b['id'] == 'all' else (pf >= b['lo']) & (pf < b['hi'])
        bins.append(dict(b, n=int(m.sum()), mask=m))
    out_runs = []
    for rid in HS_RUNS:
        R = load(rid); assert [(r[0], r[1]) for r in R] == keys, rid
        meta = json.load(open(os.path.join(HS_RESULTS, 'runs', rid, 'meta.json')))
        fa = fail_flags(R); _, lo, hi = boot_ci(fa, blocks)
        avg = lambda c: round(float(np.nanmean([np.nan if x[c] is None else x[c] for x in R])), 2)
        ci = lambda c: meta['cols'].index(c)
        d = None if rid == HS_BASE else [round(v, 1) for v in paired_diff(fa, fb, blocks)]
        out_runs.append(dict(id=rid, picker=meta['picker'], n=len(R), fail=round(100 * fa.mean(), 1), ci=[round(lo, 1), round(hi, 1)],
                             scenes={s: round(100 * fa[np.array([k[0] == s for k in keys])].mean(), 1) for s in HS_SCENES},
                             diff=d, n_match=avg(ci('n_match')), n_inl=avg(ci('n_inl')), rep1=avg(ci('rep1')), rep5=avg(ci('rep5')),
                             bins={b['id']: (round(100 * fa[b['mask']].mean(), 1) if b['n'] else None) for b in bins},
                             device=meta.get('device', ''), date=meta.get('date', ''), commit=meta.get('commit', '')))
    out = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'),
               dataset='Hypersim (preliminary): PnP relocalisation against a local map, same pipeline as TartanAir',
               source='Apple Hypersim (Roberts et al., ICCV 2021), https://github.com/apple/ml-hypersim, CC BY-SA 3.0',
               protocol=HS_PROTOCOL, scenes=[dict(id=s, n_queries=int(sum(k[0] == s for k in keys))) for s in HS_SCENES],
               base=HS_BASE, n_boot_block=20, bins=[{k: v for k, v in b.items() if k != 'mask'} for b in bins], runs=out_runs)
    json.dump(out, open(os.path.join(SITE, 'hypersim.json'), 'w'), indent=1)
    print('exported hypersim:', len(out_runs), 'runs,', len(keys), 'queries ->', os.path.join(SITE, 'hypersim.json'))


if __name__ == '__main__':
    import sys
    if '--hypersim' in sys.argv:   # only the Hypersim table, TartanAir files untouched
        export_hypersim()
    else:
        main()
