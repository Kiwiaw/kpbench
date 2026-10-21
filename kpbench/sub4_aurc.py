"""SUB4: failure-prediction table for the flat queries of one picker: AUROC + AURC / risk-coverage (Darabi-style protocol).

  python -m kpbench.sub4_aurc [--split dev] [--picker dpp0.5]   -> results/sub4/aurc_<split>.md + .json
  python -m kpbench.sub4_aurc --split fresh --run <gate run id>   -> results/sub4/aurc_fresh.md: rows of a gate run (no metric
      columns, e.g. FRESH-1), so only rep5, inlier count and the LOFO count+rep5 combination (T16 repeat, 2026-10-21)

Rows: results/eval/<split>/raco_<picker>_K100_lg_m.json.gz (metric columns), flat queries (plane_frac >= 0.85).
Confidence scores (higher = trusted): rep5 (needs the true pose: a measurement, upper reference), inlier count (test time),
geom2d = leave-one-flight-out logistic on the 2D inlier geometry hull / cells / ent (Chan-style, test time), and the
LOFO logistic combinations count+geom2d and count+geom2d+rep5.
Risk-coverage: sort by confidence, risk(c) = fail % of the most confident c share; AURC = mean risk over all coverages,
E-AURC = AURC - AURC of the perfect ranking. Block-bootstrap 95 % CI (20-frame blocks) for AURC(model) - AURC(count).
Kill (dispatch 2026-10-19 13:10): the best combination's AURC is no better than inlier count alone.
"""
import os, json, gzip, argparse
import numpy as np
from .cache import REPO
from .stats import fail_flags
from .sub1 import auroc, blocks
from .sub1_ferranti import lofo, FILL

OUT = os.path.join(REPO, 'results', 'sub4')
GEOM = ('hull', 'cells', 'ent')
COVER = (0.5, 0.8, 0.9)


def aurc(conf, y):
    o = np.argsort(-conf, kind='stable'); r = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    return float(r.mean())


def risk_at(conf, y, c):
    o = np.argsort(-conf, kind='stable'); return float(y[o[:max(1, int(round(c * len(y))))]].mean())


def load(split, picker):
    d = json.load(gzip.open(os.path.join(REPO, 'results', 'eval', split, 'raco_%s_K100_lg_m.json.gz' % picker), 'rt'))
    c = d['cols']; fl = np.array(d['flat'], bool); rows = [r for r, k in zip(d['rows'], fl) if k]
    col = lambda m: np.array([FILL.get(m, 0.0) if r[c.index(m)] is None else r[c.index(m)] for r in rows], float)
    X = {m: col(m) for m in GEOM + ('rep5',)}; X['count'] = np.log1p(col('n_inl'))
    return rows, X, fail_flags(rows).astype(float)


def load_gate_run(split, rid):
    """flat rows of a gate run (results/runs/<rid>, all flights of the split), flat by the repo labels; no metric columns"""
    from .gate import load_run, labels, split_info
    from .bench import COLS
    from .splits import FLAT
    flights = split_info(split)[0]; _, R = load_run(rid, flights); rows = []
    for f in flights:
        lab = labels(f)
        rows += [[f] + list(r[1:]) for r in R[f][0] if (lab[str(r[1])].get('plane_frac') or -1) >= FLAT]   # run rows carry no flight name
    col = lambda m: np.array([FILL.get(m, 0.0) if r[COLS.index(m)] is None else r[COLS.index(m)] for r in rows], float)
    return rows, dict(rep5=col('rep5'), count=np.log1p(col('n_inl'))), fail_flags(rows).astype(float)


def analyse(split='dev', picker='dpp0.5', n_boot=1000, seed=0, run=None):
    rows, X, y = load_gate_run(split, run) if run else load(split, picker); fid = np.array([str(r[0]) for r in rows]); inv = blocks(rows)
    S = dict(rep5=X['rep5'], count=X['count'])
    models = (('count+rep5', ('count', 'rep5')),) if run else (('geom2d', GEOM), ('count+geom2d', ('count',) + GEOM), ('count+geom2d+rep5', ('count',) + GEOM + ('rep5',)))
    for name, cols in models:
        S[name] = -lofo(fid, np.stack([X[k] for k in cols], 1), y)          # confidence = -predicted fail probability
    opt = aurc(-y, y); m = inv.max() + 1; groups = [np.where(inv == g)[0] for g in range(m)]
    rng = np.random.default_rng(seed); B = [np.concatenate([groups[g] for g in rng.integers(0, m, m)]) for _ in range(n_boot)]
    res = {}
    for k, s in S.items():
        a = aurc(s, y)
        diff = [aurc(s[i], y[i]) - aurc(S['count'][i], y[i]) for i in B] if k != 'count' else None
        res[k] = dict(auroc=round(auroc(s, y), 4), aurc=round(100 * a, 2), e_aurc=round(100 * (a - opt), 2),
                      risk={str(c): round(100 * risk_at(s, y, c), 2) for c in COVER},
                      vs_count=None if diff is None else [round(100 * (a - aurc(S['count'], y)), 2)] + [round(100 * v, 2) for v in np.percentile(diff, [2.5, 97.5])])
    combos = [k for k in res if k.startswith('count+')]
    best = min(combos, key=lambda k: res[k]['aurc']); lo, hi = res[best]['vs_count'][1:]
    verdict = ('KILL: best combination %s AURC %+.2f [%+.2f, %+.2f] pt vs inlier count, no better' % (best, *res[best]['vs_count'])
               if hi >= 0 else 'best combination %s beats inlier count: AURC %+.2f [%+.2f, %+.2f] pt' % (best, *res[best]['vs_count']))
    return dict(split=split, picker=run or picker, n_flat=len(y), fails=int(y.sum()), fail_pct=round(100 * y.mean(), 2), aurc_opt=round(100 * opt, 2),
                scores=res, verdict=verdict)


def write(out):
    os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, 'aurc_%s' % out['split'])
    json.dump(out, open(base + '.json', 'w'), indent=1)
    L = ['# Failure prediction on %s flat queries, %s (%d queries, %d fails = %.2f %%)' % (out['split'], out['picker'], out['n_flat'], out['fails'], out['fail_pct']), '',
         'AURC and risk in fail %% (lower = better); perfect ranking AURC %.2f. Fitted scores are leave-one-flight-out logistic. '
         'rep5 needs the true pose (measurement, not deployable).' % out['aurc_opt'], '',
         '| score | AUROC | AURC | E-AURC | risk @50 % | risk @80 % | risk @90 % | AURC - count [95 %] |', '|---|---|---|---|---|---|---|---|']
    for k, r in out['scores'].items():
        v = r['vs_count']
        L.append('| %s | %.3f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |' % (k, r['auroc'], r['aurc'], r['e_aurc'], r['risk']['0.5'], r['risk']['0.8'], r['risk']['0.9'],
                                                                         '-' if v is None else '%+.2f [%+.2f, %+.2f]' % tuple(v)))
    L += ['', '**%s**' % out['verdict']]
    open(base + '.md', 'w').write('\n'.join(L) + '\n'); return base + '.md'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); ap.add_argument('--picker', default='dpp0.5')
    ap.add_argument('--run', default=None, help='gate run id under results/runs (rows without metric columns)')
    a = ap.parse_args(argv); out = analyse(a.split, a.picker, run=a.run); print(write(out)); print(out['verdict'])


if __name__ == '__main__':
    main()
