"""Statistics for comparing pickers: fail flags, block bootstrap, paired differences, flight sign test, the gate rule.

Gate rule (agreed 11 Oct 2026, fixed before any held-out run). A candidate beats the incumbent on a flight set if
route A or route B passes, each at one-sided 2.5 % (total false-win chance <= 5 %):
  A (size)        paired block-bootstrap difference candidate - incumbent on the flat bin (same queries),
                  one-sided 97.5 % upper bound < 0
  B (consistency) flight sign test on flat-bin fail %, tied flights dropped, one-sided binomial p <= 0.025
and both guards hold:
  relative        flat fail % at least 10 % lower than the incumbent's
  all frames      upper bound of the all-frames paired difference <= +0.2 points
Never judge by overlap of two separate intervals: a paired difference removes the flight-to-flight spread.
"""
import numpy as np
from math import comb

BLOCK = 20
N_BOOT = 1000


def fail_flags(rows, thr_cm=25, rot_deg=2):
    """rows: [flight, q, err_cm, rot_deg, ...] -> 0/1 per query (no pose, position > thr_cm or rotation > rot_deg)"""
    return np.array([r[2] is None or r[2] > thr_cm or r[3] > rot_deg for r in rows], float)


def block_ids(flights, qs, block=BLOCK):
    return np.array(['%s:%d' % (f, q // block) for f, q in zip(flights, qs)])


def _groups(blocks):
    _, inv = np.unique(blocks, return_inverse=True)
    return inv, inv.max() + 1


def boot_ci(fails, blocks, n_boot=N_BOOT, seed=0, alpha=0.05):
    """block bootstrap interval of a fail % -> (pct, lo, hi)"""
    inv, m = _groups(blocks); s = np.bincount(inv, fails, m); c = np.bincount(inv, None, m)
    rng = np.random.default_rng(seed); pick = rng.integers(0, m, (n_boot, m))
    p = 100 * s[pick].sum(1) / c[pick].sum(1)
    return 100 * fails.mean(), float(np.quantile(p, alpha / 2)), float(np.quantile(p, 1 - alpha / 2))


def paired_diff(fa, fb, blocks, n_boot=N_BOOT, seed=0):
    """candidate fa - incumbent fb (same queries, same order) in points -> (diff, lo95, hi95). hi95 = one-sided 97.5 % bound"""
    assert len(fa) == len(fb) == len(blocks)
    inv, m = _groups(blocks); d = np.bincount(inv, fa - fb, m); c = np.bincount(inv, None, m)
    rng = np.random.default_rng(seed); pick = rng.integers(0, m, (n_boot, m))
    p = 100 * d[pick].sum(1) / c[pick].sum(1)
    return 100 * (fa - fb).mean(), float(np.quantile(p, 0.025)), float(np.quantile(p, 0.975))


def sign_test(better, worse):
    """one-sided binomial p of >= better successes out of better + worse at p = 1/2"""
    n = better + worse
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(better, n + 1)) / 2 ** n


def per_flight(fa, fb, flights):
    """-> (better, worse, tied) flights by fail % of candidate vs incumbent"""
    b = w = t = 0
    for f in sorted(set(flights)):
        m = flights == f; a, c = fa[m].mean(), fb[m].mean()
        if a < c: b += 1
        elif a > c: w += 1
        else: t += 1
    return b, w, t


def gate(cand, inc, flights, qs, flat, envs=None, n_boot=N_BOOT, block=BLOCK):
    """cand, inc: fail flags (same queries); flights, qs: per query; flat: bool per query. -> dict with verdict"""
    flights = np.asarray(flights); blocks = block_ids(flights, qs, block); flat = np.asarray(flat, bool)
    dA = paired_diff(cand[flat], inc[flat], blocks[flat], n_boot)
    dall = paired_diff(cand, inc, blocks, n_boot)
    b, w, t = per_flight(cand[flat], inc[flat], flights[flat]); pB = sign_test(b, w)
    rel = (inc[flat].mean() - cand[flat].mean()) / max(inc[flat].mean(), 1e-9)
    out = dict(flat_cand=100 * cand[flat].mean(), flat_inc=100 * inc[flat].mean(), flat_diff=dA, all_diff=dall,
               flights_better=b, flights_worse=w, flights_tied=t, sign_p=pB, rel_drop=rel,
               route_A=dA[2] < 0, route_B=pB <= 0.025, guard_rel=rel >= 0.10, guard_all=dall[2] <= 0.2)
    if envs is not None:
        envs = np.asarray(envs)
        out['per_env'] = {e: (100 * cand[flat & (envs == e)].mean(), 100 * inc[flat & (envs == e)].mean())
                          for e in sorted(set(envs[flat]))}
    out['win'] = bool((out['route_A'] or out['route_B']) and out['guard_rel'] and out['guard_all'])
    return out
