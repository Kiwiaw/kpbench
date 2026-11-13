"""T33 / SUB3 R20: per-query PnP conditioning of the picks, read from runs made with kpbench.run --save-picks.

  python -m kpbench.pickstats --cand raco_dpp0.5_K100_same_lg --inc raco_sd_K100_same_lg --split fresh [--runs runs]

A --save-picks flight file holds onplane[q] = [on-plane share, picks with depth, plane share of the fit set, per-pick labels,
per-pick depth (since 13 Nov)]. Per query this module reads
  off   = off-plane share of the picks with depth (1 - on-plane share), e.g. 100 picks, 80 on the ground plane -> 0.20
  dstd  = std of the pick depths [m]          (5th entry; None for runs saved before 13 Nov)
  dspan = (q90 - q10) / median of pick depth  (scale-free spread: depths 5 .. 25 m around 10 m -> about 2)
and compares two runs on the same queries: all flat queries, and the 'rescued' ones (cand posed, inc failed), the T33 kill:
rescued frames whose cand picks show no off-plane gain over the inc picks (median paired diff <= 0) kill the mechanism.
Runs without picks are refused (gate.require_picks).
"""
import argparse, gzip, json, os
import numpy as np
from .data import slug
from .gate import require_picks, split_info, labels, RESULTS
from .stats import fail_flags
from .splits import FLAT


def query_stats(entry):
    """one onplane entry -> (off, dstd, dspan); None where the entry lacks the field"""
    share = entry[0] if entry else None
    off = None if share is None else round(1.0 - float(share), 4)
    z = None
    if entry is not None and len(entry) >= 5 and entry[4] is not None:
        z = np.array([x for x in entry[4] if x is not None], float)
    if z is None or len(z) < 2:
        return off, None, None
    q10, q50, q90 = np.percentile(z, [10, 50, 90])
    return off, round(float(z.std()), 3), round(float((q90 - q10) / q50), 3) if q50 > 0 else None


def run_stats(rid, flights, runs='runs'):
    """-> {(flight, q): (fail, off, dstd, dspan)} for a run with picks"""
    _, out = require_picks(rid, flights, runs); d = os.path.join(RESULTS, runs, rid); res = {}
    for f in flights:
        j = json.load(gzip.open(os.path.join(d, slug(f) + '.json.gz'), 'rt')); rows = out[f][0]; fl = fail_flags(rows)
        op = j.get('onplane') or {}
        for r, bad in zip(rows, fl):
            res[(f, r[1])] = (int(bad),) + query_stats(op.get(str(r[1])))
    return res


def paired(A, B, keys):
    """median and mean of A - B per field (off, dstd, dspan) over keys where both have it, plus counts above / below 0"""
    out = {}
    for k, name in ((1, 'off'), (2, 'dstd'), (3, 'dspan')):
        d = np.array([A[q][k] - B[q][k] for q in keys if A[q][k] is not None and B[q][k] is not None], float)
        out[name] = None if not len(d) else dict(n=len(d), median=round(float(np.median(d)), 4), mean=round(float(d.mean()), 4),
                                                 up=int((d > 0).sum()), down=int((d < 0).sum()))
    return out


def compare(cand, inc, split, runs='runs'):
    flights, _, _ = split_info(split); A, B = run_stats(cand, flights, runs), run_stats(inc, flights, runs)
    keys = sorted(set(A) & set(B))
    flat = [q for q in keys if (labels(q[0])[str(q[1])].get('plane_frac') or -1) >= FLAT]
    rescued = [q for q in flat if A[q][0] == 0 and B[q][0] == 1]
    res = dict(cand=cand, inc=inc, split=split, n=len(keys), n_flat=len(flat), n_rescued=len(rescued),
               flat=paired(A, B, flat), rescued=paired(A, B, rescued))
    r = res['rescued']['off']; res['kill'] = None if r is None else bool(r['median'] <= 0)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--cand', required=True); ap.add_argument('--inc', required=True)
    ap.add_argument('--split', default='fresh'); ap.add_argument('--runs', default='runs'); a = ap.parse_args(argv)
    print(json.dumps(compare(a.cand, a.inc, a.split, a.runs), indent=1))


if __name__ == '__main__':
    main()
