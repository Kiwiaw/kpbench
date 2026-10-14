"""Stage 4: results/ -> docs/data/ (index.json, labels.json, runs/<id>.json, picks/<id>/<slug>.json). See docs/data_spec.md.
Extra datasets (DATASETS below, e.g. Hypersim) are written in the same format to docs/data/<name>/; the page chooses one
with its dataset dropdown. They never enter the TartanAir files.

  python -m kpbench.export                      # TartanAir and every extra dataset
  python -m kpbench.export --only hypersim      # only the named extra dataset(s), comma separated; TartanAir files untouched
Thumbnails are produced separately by kpbench.thumbs (they do not change between runs; one folder per flight slug, shared by
all datasets).
"""
import os, json, gzip, glob, datetime
import numpy as np
from .data import FLIGHTS, slug, KF, flight_dir
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

# Extra datasets: results folder, flights (in this order; only flights that every run covers are exported), keyframe step,
# keyframe window and the protocol note shown under the table. Hypersim needs KPBENCH_INTR="554.256258 554.256258 320 240".
HS_FLIGHTS = ['hypersim/ai_001_001', 'hypersim/ai_001_002']   # ai_003_001 dropped: its renders are black in the released data
HS_SOURCE = 'Apple Hypersim (Roberts et al., ICCV 2021), https://github.com/apple/ml-hypersim, CC BY-SA 3.0'


def _hs_protocol(win, n_kf):
    return ['Apple Hypersim, 2 scenes ai_001_001 and ai_001_002, camera cam_00 (ai_003_001 dropped: its renders are black in the released data)',
            'images 1024x768 resized to 640x480, fx = fy = 554.256 px, cx = 320, cy = 240; planar depth from depth_meters',
            'frames are about 1.1 m apart, so a keyframe every 2nd frame and a query uses the keyframes within +-%d frames (%d keyframes)' % (win, n_kf),
            'RaCo candidates, K = 100, same map, LightGlue, AP3P RANSAC + LM refinement; fail = no pose, > 25 cm or > 2 deg (as TartanAir)',
            'planarity bins, thresholds and block bootstrap (20-frame blocks) as TartanAir; preliminary: few queries, wide intervals']


DATASETS = {
    'hypersim': dict(results='results_hypersim_w3', flights=HS_FLIGHTS, kf=2, win=3, source=HS_SOURCE,
                     label='Hypersim (2 scenes, preliminary)', protocol=_hs_protocol(3, 4),
                     dataset='Hypersim (preliminary), PnP relocalisation against a local map of ground-truth 3D points, same pipeline as TartanAir'),
    'hypersim_w2': dict(results='results_hypersim_w2', flights=HS_FLIGHTS, kf=2, win=2, source=HS_SOURCE,
                        label='Hypersim, 2 keyframes per query (variant)', protocol=_hs_protocol(2, 2),
                        dataset='Hypersim (preliminary, variant: 2 keyframes per query), PnP relocalisation against a local map of ground-truth 3D points'),
}


def load_labels(results=RESULTS):
    lab = {}
    for fn in glob.glob(os.path.join(results, 'labels', '*.json')):
        s = os.path.basename(fn)[:-5]
        d = json.load(open(fn)); qs = sorted(int(q) for q in d)
        lab[s] = dict(q=qs, **{k: [d[str(q)].get(k) for q in qs] for k in ('plane_frac', 'med_depth', 'missing_depth', 'coverage')})
    return lab


def _n_frames(f, nq, kf, local=False):
    """Frame count. TartanAir: from the query count (a keyframe every kf frames). Extra datasets (local=True): from the local
    data when it is there (as data.Flight counts it), else unknown."""
    if not local:
        return (nq + (nq // (kf - 1)) + 1) if nq else None
    d = flight_dir(f)
    if d:
        n = [len(np.loadtxt(os.path.join(d, 'pose_left.txt')))]
        n += [len([x for x in os.listdir(os.path.join(d, sub)) if x.endswith(ext)]) for sub, ext in (('image_left', '.png'), ('depth_left', '.npy'))]
        return min(n)
    return None


def export_set(results=RESULTS, site=SITE, flights=None, kf=KF, dataset=None, extra=None):
    """One dataset: results/ -> site/. flights=None: every TartanAir flight that any run or label file covers, in the canonical
    order. A list (extra datasets, whose flights are not in data.FLIGHTS): those of its flights that every run and the labels cover."""
    os.makedirs(os.path.join(site, 'runs'), exist_ok=True)
    labels = load_labels(results); extra_set = flights is not None
    present = set()
    run_dirs = sorted(glob.glob(os.path.join(results, 'runs', '*', 'meta.json')))
    per_run = {}
    for mfn in run_dirs:
        meta = json.load(open(mfn)); rd = os.path.dirname(mfn); parts = {}
        for fn in glob.glob(os.path.join(rd, '*.json.gz')):
            d = json.load(gzip.open(fn, 'rt'))
            if flights is not None and d['flight'] not in flights:
                continue
            parts[d['flight']] = d; present.add(d['flight'])
        per_run[meta['id']] = (meta, parts)
    if flights is None:
        present |= {f for f in FLIGHTS if slug(f) in labels}
        flights = [f for f in FLIGHTS if f in present]
    else:
        keep = []
        for f in flights:
            lack = [rid for rid, (_, p) in per_run.items() if f not in p]
            if lack or slug(f) not in labels:
                print('skip', f, ': missing in runs', lack, '' if slug(f) in labels else '(and no labels)')
            else:
                keep.append(f)
        flights = keep
    fidx = {f: i for i, f in enumerate(flights)}
    finfo = []
    for f in flights:
        s = slug(f); nq = len(labels[s]['q']) if s in labels else None
        if nq is None:
            for meta, parts in per_run.values():
                if f in parts:
                    nq = len(parts[f]['rows']); break
        finfo.append(dict(id=f, slug=s, env=f.split('/')[0], n_queries=nq, n_frames=_n_frames(f, nq, kf, extra_set)))
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
                pd = os.path.join(site, 'picks', rid); os.makedirs(pd, exist_ok=True)
                json.dump(parts[f]['picks'], open(os.path.join(pd, slug(f) + '.json'), 'w'), separators=(',', ':'))
        if not rows:
            continue
        fail25 = float(np.mean([r[2] is None or r[2] > 25 or r[3] > ROT_FAIL[25] for r in rows]))
        cfg = {k: v for k, v in meta.items() if k not in ('cols',)}
        json.dump(dict(id=rid, config=cfg, cols=meta['cols'], rows=rows), open(os.path.join(site, 'runs', rid + '.json'), 'w'), separators=(',', ':'))
        runs.append(dict(id=rid, det=meta['det'], picker=meta['picker'], K=meta['K'], map=meta['map'], matcher=meta.get('matcher', 'nn'),
                         date=meta.get('date', ''), commit=meta.get('commit', ''), flights=[f for f in flights if f in parts],
                         n_queries=len(rows), fail25=round(100 * fail25, 2), has_picks=has_picks, notes=meta.get('notes', '')))
    # order: RaCo before SuperPoint, same map before dense, K=100 first, the strongest-with-depth baseline first within a group
    runs.sort(key=lambda r: (r['det'] != 'raco', r['map'] != 'same', r['K'] != 100, r['K'], r['picker'] != 'sd', r['picker']))
    index = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'),
                 dataset=dataset or 'TartanAir (Hard), PnP relocalisation against a local map of ground-truth 3D points',
                 flights=finfo, bins=BINS, thresholds_cm=THR, rot_fail=ROT_FAIL, rot_fail_deg=2, **(extra or {}), runs=runs)
    json.dump(index, open(os.path.join(site, 'index.json'), 'w'), indent=1)
    json.dump({f: labels[slug(f)] for f in flights if slug(f) in labels}, open(os.path.join(site, 'labels.json'), 'w'), separators=(',', ':'))
    print('exported', len(runs), 'runs,', len(flights), 'flights ->', site)


def export_extra(name):
    """Extra dataset `name` -> docs/data/<name>/ (same files as TartanAir, plus kf / win / protocol / source in index.json)."""
    c = DATASETS[name]
    res = os.path.join(REPO, c['results'])
    if not os.path.isdir(os.path.join(res, 'runs')):
        print('skip dataset', name, ': no', res)
        return
    export_set(res, os.path.join(SITE, name), flights=c['flights'], kf=c['kf'], dataset=c['dataset'],
               extra=dict(name=name, label=c['label'], kf=c['kf'], win=c['win'], protocol=c['protocol'], source=c['source'],
                          results=c['results']))


def main():
    export_set()
    for name in DATASETS:
        export_extra(name)


if __name__ == '__main__':
    import sys
    if '--only' in sys.argv:   # only the named extra datasets, TartanAir files untouched
        for name in sys.argv[sys.argv.index('--only') + 1].split(','):
            export_extra(name)
    else:
        main()
