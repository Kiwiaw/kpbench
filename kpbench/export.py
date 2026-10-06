"""Stage 4: results/ -> docs/data/ (index.json, labels.json, runs/<id>.json, picks/<id>/<slug>.json). See docs/data_spec.md.

  python -m kpbench.export
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
THR = [5, 25, 50, 100]


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
        fail25 = float(np.mean([r[2] is None or r[2] > 25 or r[3] > 5 for r in rows]))
        cfg = {k: v for k, v in meta.items() if k not in ('cols',)}
        json.dump(dict(id=rid, config=cfg, cols=meta['cols'], rows=rows), open(os.path.join(SITE, 'runs', rid + '.json'), 'w'), separators=(',', ':'))
        runs.append(dict(id=rid, det=meta['det'], picker=meta['picker'], K=meta['K'], map=meta['map'], matcher=meta.get('matcher', 'nn'),
                         date=meta.get('date', ''), commit=meta.get('commit', ''), flights=[f for f in flights if f in parts],
                         n_queries=len(rows), fail25=round(100 * fail25, 2), has_picks=has_picks, notes=meta.get('notes', '')))
    index = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'),
                 dataset='TartanAir (Hard), PnP relocalisation against a local map of ground-truth 3D points',
                 flights=finfo, bins=BINS, thresholds_cm=THR, rot_fail_deg=5, runs=runs)
    json.dump(index, open(os.path.join(SITE, 'index.json'), 'w'), indent=1)
    json.dump({f: labels[slug(f)] for f in flights if slug(f) in labels}, open(os.path.join(SITE, 'labels.json'), 'w'), separators=(',', ':'))
    print('exported', len(runs), 'runs,', len(flights), 'flights ->', SITE)


if __name__ == '__main__':
    main()
