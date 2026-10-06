"""Stage 3: run one configuration on flights and store rows.

  python -m kpbench.run --det raco --picker sd --K 100 --map same --flights all [--save-picks] [--tag x] [--force]

Run id = <det>_<picker>_K<K>_<map>_<matcher>[_<tag>]  (matcher nn | lg). Output per flight: results/runs/<run_id>/<flight_slug>.json.gz
with {"rows": [...], "picks": {...}} and results/runs/<run_id>/meta.json (config, date, git commit).
Flights already done are skipped unless --force. Export (kpbench.export) merges flights for the page.
"""
import os, sys, json, gzip, time, subprocess, datetime
from .data import parse_flights, slug
from .cache import REPO
from .bench import run_flight, COLS

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))


def run_id(det, picker, K, mapmode, tag='', matcher='nn'):
    return '%s_%s_K%d_%s_%s%s' % (det, picker, K, mapmode, matcher, ('_' + tag) if tag else '')


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ''


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True); ap.add_argument('--picker', required=True); ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--map', default='same', choices=['same', 'dense']); ap.add_argument('--flights', default='all')
    ap.add_argument('--tag', default=''); ap.add_argument('--save-picks', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=0); ap.add_argument('--notes', default='')
    ap.add_argument('--matcher', default='nn', choices=['nn', 'lg'], help='nn = mutual nearest neighbour, lg = LightGlue')
    a = ap.parse_args(argv)
    rid = run_id(a.det, a.picker, a.K, a.map, a.tag, a.matcher); out = os.path.join(RESULTS, 'runs', rid); os.makedirs(out, exist_ok=True)
    meta_fn = os.path.join(out, 'meta.json')
    meta = json.load(open(meta_fn)) if os.path.isfile(meta_fn) else {}
    meta.update(id=rid, det=a.det, picker=a.picker, K=a.K, map=a.map, matcher=a.matcher, tag=a.tag, cols=COLS,
                protocol='C23: keyframes every 10 within +-20, %s, PnP RANSAC 3 px 2000 it >= 6 inliers' % ('LightGlue per keyframe' if a.matcher == 'lg' else 'mutual NN'),
                has_picks=bool(a.save_picks) or meta.get('has_picks', False))
    if a.notes:
        meta['notes'] = a.notes
    for fid in parse_flights(a.flights):
        fn = os.path.join(out, slug(fid) + '.json.gz')
        if os.path.isfile(fn) and not a.force:
            print('exists', fn); continue
        try:
            rows, picks = run_flight(fid, a.det, a.picker, a.K, a.map, workers=a.workers or None, save_picks=a.save_picks, matcher=a.matcher)
        except FileNotFoundError as e:
            print('skip', fid, e); continue
        json.dump(dict(flight=fid, rows=rows, picks=picks), gzip.open(fn, 'wt'))
        meta['date'] = datetime.datetime.now().isoformat(timespec='seconds'); meta['commit'] = git_commit()
        json.dump(meta, open(meta_fn, 'w'), indent=1)
    json.dump(meta, open(meta_fn, 'w'), indent=1)
    print('run', rid, 'stored in', out)


if __name__ == '__main__':
    main()
