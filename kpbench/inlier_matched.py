"""MAIN control: does a spread picker fail less than sd at the SAME inlier count? (scout objection 'spreading only raises inliers')

  python -m kpbench.inlier_matched --split dev --cand dpp0.5 [--ref sd]   -> results/main/inlier_matched_<split>.md + .json

Rows: results/eval/<split>/raco_<picker>_K100_lg_m.json.gz (same queries, one node), flat queries only (plane_frac >= 0.85).
(1) stratified: both pickers' flat queries pooled, binned by their own inlier count (deciles of the pooled count);
    per bin fail % of cand and ref with n; matched difference = sum_b w_b (fail_cand_b - fail_ref_b), w_b = share of
    the bin among ref queries; 95 % CI by block bootstrap (20-frame blocks, the same blocks for both pickers).
(2) logistic fail ~ log(1+n_inl) + [picker = cand] on the stacked rows; coefficient of the picker dummy with block CI.
Kill (dispatch 2026-10-19 13:10): picker-dummy CI covers 0 -> the uniformity gain is 'more inliers'.
"""
import os, json, argparse
import numpy as np
from .cache import REPO
from .stats import fail_flags
from .sub1 import table, blocks
from .sub1_ferranti import logit_fit

OUT = os.path.join(REPO, 'results', 'main')


def strat_diff(nc, fc, nr, fr, edges):
    """weighted mean over inlier bins of fail(cand) - fail(ref); weights = ref share per bin, renormalised over bins both fill"""
    bc, br = np.digitize(nc, edges), np.digitize(nr, edges); d = w = 0.0; tab = []
    for b in range(len(edges) + 1):
        mc, mr = bc == b, br == b
        tab.append((int(mc.sum()), float(fc[mc].mean()) if mc.any() else None, int(mr.sum()), float(fr[mr].mean()) if mr.any() else None))
        if mc.any() and mr.any():
            d += mr.mean() * (fc[mc].mean() - fr[mr].mean()); w += mr.mean()
    return d / w if w else 0.0, tab


def analyse(split='dev', cand='dpp0.5', ref='sd', n_boot=1000, seed=0):
    R, M, Fl = table(split, (ref, cand)); flat = Fl[ref]
    fc, fr = fail_flags(R[cand])[flat].astype(float), fail_flags(R[ref])[flat].astype(float)
    nc, nr = np.nan_to_num(M[cand]['n_inl'])[flat], np.nan_to_num(M[ref]['n_inl'])[flat]
    inv = blocks([r for r, k in zip(R[ref], flat) if k]); m = inv.max() + 1
    edges = np.unique(np.percentile(np.concatenate([nc, nr]), np.arange(10, 100, 10)))
    d0, tab = strat_diff(nc, fc, nr, fr, edges)
    def dummy(idx):
        x = np.log1p(np.concatenate([nc[idx], nr[idx]])); y = np.concatenate([fc[idx], fr[idx]])
        A = np.stack([x, np.r_[np.ones(len(idx)), np.zeros(len(idx))]], 1); mu, sd = A.mean(0), A.std(0) + 1e-12
        return logit_fit((A - mu) / sd, y)[2] / sd[1]               # log-odds of cand vs ref at equal log inliers
    allidx = np.arange(len(fc)); w0 = dummy(allidx)
    groups = [np.where(inv == g)[0] for g in range(m)]; rng = np.random.default_rng(seed); D, W = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, m, m)])
        D.append(strat_diff(nc[idx], fc[idx], nr[idx], fr[idx], edges)[0]); W.append(dummy(idx))
    raw = fc.mean() - fr.mean()
    out = dict(split=split, cand=cand, ref=ref, n_flat=int(flat.sum()), fail_cand=round(100 * fc.mean(), 2), fail_ref=round(100 * fr.mean(), 2),
               raw_diff=round(100 * raw, 2), inl_median=dict(cand=float(np.median(nc)), ref=float(np.median(nr))),
               matched_diff=[round(100 * d0, 2)] + [round(100 * v, 2) for v in np.percentile(D, [2.5, 97.5])],
               matched_share=round(d0 / raw, 2) if raw else None,
               dummy_logodds=[round(w0, 3)] + [round(float(v), 3) for v in np.percentile(W, [2.5, 97.5])],
               edges=[float(e) for e in edges], bins=tab)
    lo, hi = out['dummy_logodds'][1:]
    out['verdict'] = ('KILL: picker effect covers 0 at equal inlier count (gain = more inliers)' if lo <= 0 <= hi else
                      '%s %s than %s at equal inlier count' % (cand, 'fails less' if hi < 0 else 'fails more', ref))
    return out


def write(out):
    os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, 'inlier_matched_%s' % out['split'])
    json.dump(out, open(base + '.json', 'w'), indent=1)
    L = ['# Inlier-matched control on %s flat (%d queries): %s vs %s' % (out['split'], out['n_flat'], out['cand'], out['ref']), '',
         'raw flat fail %.2f vs %.2f %% (diff %+.2f pt); median inliers %.0f vs %.0f' % (out['fail_cand'], out['fail_ref'], out['raw_diff'],
                                                                                    out['inl_median']['cand'], out['inl_median']['ref']),
         'inlier-matched diff (deciles of own inlier count, weighted by %s): %+.2f pt [%+.2f, %+.2f], %s of the raw gap' % (
             out['ref'], *out['matched_diff'], out['matched_share']),
         'logistic fail ~ log(1+inliers) + picker: log-odds %s vs %s %+.3f [%+.3f, %+.3f]' % (out['cand'], out['ref'], *out['dummy_logodds']), '',
         '| inlier bin | upper edge | n %s | fail %% %s | n %s | fail %% %s |' % (out['cand'], out['cand'], out['ref'], out['ref']), '|---|---|---|---|---|---|']
    for b, (n1, f1, n2, f2) in enumerate(out['bins']):
        e = '%.0f' % out['edges'][b] if b < len(out['edges']) else 'max'
        L.append('| %d | %s | %d | %s | %d | %s |' % (b + 1, e, n1, '-' if f1 is None else '%.1f' % (100 * f1), n2, '-' if f2 is None else '%.1f' % (100 * f2)))
    L += ['', '**%s**' % out['verdict']]
    open(base + '.md', 'w').write('\n'.join(L) + '\n'); return base + '.md'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); ap.add_argument('--cand', default='dpp0.5'); ap.add_argument('--ref', default='sd')
    a = ap.parse_args(argv); out = analyse(a.split, a.cand, a.ref); print(write(out)); print(out['verdict'])


if __name__ == '__main__':
    main()
