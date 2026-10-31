"""Restore check: per flight of a split, candidate cache frames vs flight frames (the C.N == F.N test of kpbench.run), and images.

  python tools/dev_frame_check.py [--split dev|held_out|fresh]    (tools/dev_frame_check.sh on Euler); python tools/dev_frame_check.py --self-test

Exit 1 on any flight that kpbench.run would refuse: missing flight, missing cache, or cache frames != flight frames.
Missing images only WARN: cached runs never read them (kpbench.data.Flight).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def verdict(cache, flight, images):
    """frame counts, None = missing -> (status, blocks a run)"""
    if flight is None or cache is None:
        return 'MISSING %s' % ('flight' if flight is None else 'cache'), True
    if cache != flight:
        return 'SHORT cache %d, flight %d, %d missing' % (cache, flight, cache - flight), True
    return ('OK' if images >= flight else 'WARN images %d of %d' % (images, flight)), False


def flights_of(split):
    from kpbench.cache import REPO
    if split == 'fresh':
        return json.load(open(os.path.join(REPO, 'results', 'gates', 'FRESH_set.json')))['fresh']
    return json.load(open(os.path.join(REPO, 'splits.json')))[split]


def counts(f, det):
    from kpbench.cache import Cands
    from kpbench.data import Flight, slug
    try:
        F = Flight(f); fl, im = F.N, F.n_img
    except (FileNotFoundError, OSError):
        fl, im = None, 0
    try:
        ca = Cands(slug(f), det).N
    except FileNotFoundError:
        ca = None
    return ca, fl, im


def self_test():
    assert verdict(577, 384, 384) == ('SHORT cache 577, flight 384, 193 missing', True)
    assert verdict(533, 401, 401) == ('SHORT cache 533, flight 401, 132 missing', True)
    assert verdict(577, 577, 0) == ('WARN images 0 of 577', False)
    assert verdict(577, 577, 577) == ('OK', False)
    assert verdict(None, 577, 577)[1] and verdict(577, None, 0)[1]
    print('self-test PASS: 384/577 and 401/533 flagged')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev', choices=['dev', 'held_out', 'fresh']); ap.add_argument('--det', default='raco')
    ap.add_argument('--self-test', action='store_true'); a = ap.parse_args()
    if a.self_test:
        return self_test()
    bad = 0; fl = flights_of(a.split)
    for f in fl:
        s, b = verdict(*counts(f, a.det)); bad += b; print('%-22s %s' % (f, s), flush=True)
    print('%s: %d of %d flights block a run' % (a.split, bad, len(fl)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
