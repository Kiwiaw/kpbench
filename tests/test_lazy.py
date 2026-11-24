"""T55: the opt-in dpplazy picker returns the benchmark dpp picks (synthetic frames, no data)."""
import numpy as np
from kpbench.pickers import parse, to3d, dpp_greedy, dpp_greedy_lazy
from kpbench.data import DEPTH_MAX


def frame(n, seed):
    """n candidates in score order, depths 2 .. 60 m, 10 % sky"""
    rng = np.random.default_rng(seed); u = rng.uniform(0, 640, n); v = rng.uniform(0, 480, n)
    z = rng.uniform(2, 60, n); z[rng.random(n) < 0.1] = 1e9
    return u, v, np.linspace(1, 0, n), z


def test_lazy_greedy_equals_full_greedy():
    for seed in range(5):
        u, v, s, z = frame(1500, seed); ok = np.flatnonzero(z < DEPTH_MAX); P = to3d(u[ok], v[ok], z[ok])
        q = np.exp(-np.arange(len(ok)) / 100.0)
        for K in (1, 50, 100, 400):
            assert np.array_equal(dpp_greedy(P, q, K, 0.5), dpp_greedy_lazy(P, q, K, 0.5))


def test_dpplazy_picker_equals_dpp_picker():
    ref, lz = parse('dpp0.5'), parse('dpplazy0.5')
    for seed, n in ((0, 30), (1, 129), (2, 2000)):                    # fewer candidates than K, prefix edge, full frame
        u, v, s, z = frame(n, seed)
        for K in (50, 100):
            assert np.array_equal(ref(u, v, s, z, K), lz(u, v, s, z, K))
