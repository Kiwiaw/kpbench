"""T33: per-pick depth in the --save-picks onplane entry and the pickstats reader, on a synthetic frame (no data)."""
import numpy as np
from kpbench import bench, pickstats
from kpbench.data import DEPTH_MAX


def plane_frame(n=600, seed=0):
    """ground plane 3 m below the camera (5 .. 30 m) plus a 'tree' cluster of 150 points 2 m above it"""
    rng = np.random.default_rng(seed); z = rng.uniform(5, 30, n); x = rng.uniform(-0.8, 0.8, n) * z; y = np.full(n, 3.0)
    y[:150] = 1.0; z[-30:] = 1e9                                     # 150 off-plane points, 30 sky
    u = 320 * x / np.where(z < DEPTH_MAX, z, 1) + 320; v = 320 * y / np.where(z < DEPTH_MAX, z, 1) + 240
    return u, v, z


def test_query_stats_known_values():
    off, dstd, dspan = pickstats.query_stats([0.8, 5, 0.9, [1, 1, 1, 1, 0], [10.0, 10.0, 10.0, 10.0, 10.0]])
    assert off == 0.2 and dstd == 0.0 and dspan == 0.0
    off, dstd, dspan = pickstats.query_stats([0.5, 3, 0.9, [1, 0, None]])       # pre-13-Nov entry: no depths
    assert off == 0.5 and dstd is None and dspan is None
    assert pickstats.query_stats([None, 0, None, [], []]) == (None, None, None)


def test_onplane_share_matches_planted_split():
    u, v, z = plane_frame()
    idx = np.arange(200, 600)                                         # 0 off-plane picks with depth -> off 0
    (share, n_depth, _), lab = bench.onplane(u, v, z, idx)
    assert n_depth == 370 and share == 1.0
    idx = np.r_[np.arange(0, 100), np.arange(300, 400)]               # 100 off-plane + 100 on-plane
    (share, n_depth, _), lab = bench.onplane(u, v, z, idx)
    assert n_depth == 200 and abs(share - 0.5) < 1e-9
    zz = [round(float(x), 2) if x < DEPTH_MAX else None for x in z[idx]]
    off, dstd, dspan = pickstats.query_stats([share, n_depth, None, lab, zz])
    assert off == 0.5 and abs(dstd - z[idx].std()) < 0.01 and dspan > 0


def test_paired_counts():
    A = {('f', 1): (0, 0.3, 2.0, 1.0), ('f', 2): (0, 0.2, 1.0, 0.5)}
    B = {('f', 1): (1, 0.1, 1.0, 0.5), ('f', 2): (1, 0.2, None, 0.5)}
    p = pickstats.paired(A, B, sorted(A))
    assert p['off'] == dict(n=2, median=0.1, mean=0.1, up=1, down=0)
    assert p['dstd']['n'] == 1 and p['dspan']['up'] == 1
