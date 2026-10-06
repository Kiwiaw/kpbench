"""Candidate cache: per flight and detector, the strongest 2000 candidates of every frame in score order.

File: <CACHE_DIR>/<flight_slug>_<det>.npz with
  n    (N,)            number of candidates in frame f
  u, v (N, C) float32  pixel coordinates (NaN beyond n[f])
  s    (N, C) float32  detector score (RaCo: ranker score; SuperPoint: keypoint score)
  z    (N, C) float32  ground-truth depth at the pixel (>= 1000 = no depth)
  desc (N, C, 256) float16  SuperPoint descriptor sampled at the point (unit length)
CACHE_DIR = env KPBENCH_CACHE (default <repo>/cache).
"""
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get('KPBENCH_CACHE', os.path.join(REPO, 'cache'))
NCAND = 2000
DETS = ('raco', 'sp')


def path(flight_slug, det):
    return os.path.join(CACHE_DIR, '%s_%s.npz' % (flight_slug, det))


def save(flight_slug, det, frames):
    """frames: list over f of (u, v, s, z, desc) arrays in score order"""
    N = len(frames)
    C = max(len(t[0]) for t in frames)
    n = np.zeros(N, np.int32)
    u = np.full((N, C), np.nan, np.float32); v = u.copy(); s = u.copy(); z = np.full((N, C), 1e9, np.float32)
    desc = np.zeros((N, C, 256), np.float16)
    for f, (uu, vv, ss, zz, dd) in enumerate(frames):
        m = len(uu); n[f] = m
        u[f, :m] = uu; v[f, :m] = vv; s[f, :m] = ss; z[f, :m] = zz; desc[f, :m] = dd
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path(flight_slug, det), n=n, u=u, v=v, s=s, z=z, desc=desc)


class Cands:
    """one flight + detector; frame(f) -> (u, v, s, z, desc) float64 views in score order"""
    def __init__(self, flight_slug, det):
        p = path(flight_slug, det)
        if not os.path.isfile(p):
            raise FileNotFoundError('no candidate cache %s (run kpbench.extract first)' % p)
        d = np.load(p)
        self.n = d['n']; self.u = d['u']; self.v = d['v']; self.s = d['s']; self.z = d['z']; self.desc = d['desc']
        self.N = len(self.n)

    def frame(self, f):
        m = int(self.n[f])
        return (self.u[f, :m].astype(np.float64), self.v[f, :m].astype(np.float64), self.s[f, :m].astype(np.float64),
                self.z[f, :m].astype(np.float64), self.desc[f, :m])
