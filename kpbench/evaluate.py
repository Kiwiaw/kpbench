"""Score one picker on a frozen split (splits.json) under the kpbench protocol, with the search rules enforced.

  python -m kpbench.evaluate --picker cube0.5 [--split micro|dev|held_out] [--matcher lg] [--save]

evaluate(picker) is the scoring function for code search (ShinkaEvolve / OpenEvolve) and tuning (Optuna).
picker = a name for kpbench.pickers.parse or a function f(u, v, s, z, K) -> indices into the candidates.
Rules (a violation raises PickerError, the candidate scores nothing):
  - indices are integers, unique, inside the candidate list
  - exactly K of them whenever the frame has >= K candidates
  - no oracle: pickers marked needs_ctx (they see true poses / other frames) are refused unless allow_oracle=True
Protocol = kpbench.bench (same map: keyframes keep the same picker's points; LightGlue by default; AP3P-RANSAC + LM;
fail = no pose, > 25 cm or > 2 deg). Only the frames a query set needs are picked (its queries and their keyframes), so
MICRO runs in about a minute on the laptop GPU. Output: fail % on the flat bin and all queries of the split, mean
repeatability at lag 5, rows per query (for paired comparisons with kpbench.stats).
"""
import os, json, time, gzip
import numpy as np
from .data import Flight, to3d, DEPTH_MAX, slug
from .cache import Cands, REPO
from .pickers import parse as parse_picker
from .bench import match_lightglue, match_mutual, pnp, repeat, COLS
from .splits import load as load_splits, FLAT
from .stats import fail_flags

EVAL_DIR = os.path.join(REPO, 'results', 'eval')


class PickerError(Exception):
    pass


def checked_pick(pk, u, v, s, z, de, K, ctx, allow_oracle):
    kw = {}
    if getattr(pk, 'needs_ctx', False):
        if not allow_oracle:
            raise PickerError('oracle picker (needs_ctx) not allowed')
        kw['ctx'] = ctx
    if getattr(pk, 'needs_desc', False):
        kw['de'] = de
    idx = np.asarray(pk(u, v, s, z, K, **kw))
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise PickerError('indices must be integers, got %s' % idx.dtype)
    idx = idx.astype(int).ravel()
    if len(np.unique(idx)) != len(idx):
        raise PickerError('duplicate indices')
    if len(idx) and (idx.min() < 0 or idx.max() >= len(u)):
        raise PickerError('index out of range')
    if len(idx) > K or (len(u) >= K and len(idx) != K):
        raise PickerError('picked %d, need exactly K=%d' % (len(idx), K))
    return idx


def query_sets(split):
    sp = load_splits()
    if split == 'micro':
        return {f: qs for f, qs in sp['micro'].items()}
    if split in ('dev', 'held_out'):
        return {f: None for f in sp[split]}          # None = every query of the flight
    raise ValueError(split)


def eval_flight(fid, qs, pk, K, det, matcher, allow_oracle):
    F = Flight(fid); C = Cands(F.slug, det)
    if C.N != F.N:
        raise RuntimeError('%s: cache has %d frames, flight has %d' % (fid, C.N, F.N))
    lab = json.load(open(os.path.join(REPO, 'results', 'labels', F.slug + '.json')))
    qs = F.queries if qs is None else qs
    need = sorted(set(qs) | {k for q in qs for k in F.near(q)})
    sel, MAP = {}, {}
    for f in need:
        u, v, s, z, de = C.frame(f)
        sel[f] = checked_pick(pk, u, v, s, z, de, K, (F, C, f), allow_oracle)
    for k in need:
        if k % 10 == 0:
            u, v, s, z, de = C.frame(k); i = sel[k]; i = i[z[i] < DEPTH_MAX]; Rk, tk = F.cam_pose(k)
            MAP[k] = (to3d(u[i], v[i], z[i]) @ Rk.T + tk, de[i], np.stack([u[i], v[i]], 1))
    rows, flat, trees = [], [], {}
    for q in qs:
        near = F.near(q); Rq, tq = F.cam_pose(q); u, v, s, z, de = C.frame(q); i = sel[q]
        if matcher == 'lg':
            mq, pairs = match_lightglue(np.stack([u[i], v[i]], 1), de[i], [(MAP[k][2], MAP[k][1]) for k in near])
            obj = np.array([MAP[near[ki]][0][pi] for ki, pi in pairs]).reshape(-1, 3)
        else:
            mq, mm = match_mutual(de[i], np.concatenate([MAP[k][1] for k in near]))
            obj = np.concatenate([MAP[k][0] for k in near])[mm]
        err, rot, ninl = pnp(obj, np.stack([u[i][mq], v[i][mq]], 1), Rq, tq)
        rep = repeat(F, C, q, i, trees)
        for f in [f for f in trees if f < q]:
            del trees[f]
        rows.append([fid, q, err, rot, int(len(i)), int(len(mq)), ninl] + rep)
        pf = lab[str(q)].get('plane_frac'); flat.append(pf is not None and pf >= FLAT)
    return rows, flat


def evaluate(picker, split='micro', K=100, det='raco', matcher='lg', allow_oracle=False, save=False, name=None):
    pk = parse_picker(picker) if isinstance(picker, str) else picker
    t0 = time.time(); rows, flat = [], []
    for fid, qs in query_sets(split).items():
        r, fl = eval_flight(fid, qs, pk, K, det, matcher, allow_oracle); rows += r; flat += fl
    fails = fail_flags(rows); flat = np.array(flat, bool)
    rep5 = [r[COLS.index('rep5')] for r in rows if r[COLS.index('rep5')] is not None]
    out = dict(picker=name or (picker if isinstance(picker, str) else getattr(picker, '__name__', 'fn')), split=split, K=K,
               det=det, matcher=matcher, n=len(rows), n_flat=int(flat.sum()),
               fail_flat=round(100 * fails[flat].mean(), 3) if flat.any() else None, fail_all=round(100 * fails.mean(), 3),
               rep5=round(float(np.mean(rep5)), 4) if rep5 else None, secs=round(time.time() - t0, 1))
    if save:
        d = os.path.join(EVAL_DIR, split); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, '%s_%s_K%d_%s.json.gz' % (det, out['picker'], K, matcher))
        json.dump(dict(summary=out, cols=COLS, rows=rows, flat=flat.tolist()), gzip.open(fn, 'wt'))
        out['saved'] = os.path.relpath(fn, REPO)
    return out, rows, flat


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--picker', required=True); ap.add_argument('--split', default='micro')
    ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--matcher', default='lg', choices=['lg', 'nn']); ap.add_argument('--save', action='store_true')
    a = ap.parse_args(argv)
    out, _, _ = evaluate(a.picker, a.split, a.K, a.det, a.matcher, save=a.save)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
