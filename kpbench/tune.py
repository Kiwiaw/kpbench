"""Optuna tuning of a picker family on DEV flat queries (week-1 step 4). No LLM; runs as an Euler array (euler/tune.sh).

  python -m kpbench.tune --study dpp_dev1 --trials 5          # one worker: 5 trials appended to the shared study
  python -m kpbench.tune --study dpp_dev1 --summary           # results/tune/<study>.json + .md (best trials)

Objective = fail % on the flat queries of the 12 DEV flights (936 queries, kpbench protocol, LightGlue). Repeatability at
lag 5 and the picker name are stored per trial. Held-out flights are never touched here: the best trial goes to the gate
(euler/gate.sh <picker> <incumbent> dev, then held_out). Trials run on different nodes, so LightGlue device noise
(~1 point on the flat bin, audit C4) stays in the objective; the gate reruns candidate and incumbent on one node.
Families: dpp (sigma 0.1-2 m log, alpha 0.2-5 log; dpp<sigma>a<alpha>), cube (size 0.15-1.5 m; cube<size>), pd (r 0.1-1.5 m,
theta 0-0.1; pd<r>a<theta>), up (T8: border b px, NMS radius r px, pool n before dpp0.5; upb<b>r<r>+dpp0.5p<n>,
docs/t8_upstream_design.md), win (T4 step 2: strongest per N x N px window, N 8-32, then dpp sigma 0.3-0.7 m, alpha 0.5-2;
win<N>+dpp<sigma>a<alpha>), dppr (T4 step 3: sigma 0.25-1 m log, alpha 0-2, beta 0-3, tau 0.7/0.8/0.9;
dppr<sigma>a<alpha>b<beta>t<tau>, docs/t4_dppr_design.md; ablation via env KPBENCH_DPPR_QMODE), ssc (MAIN spread-class baseline, Bailo 2018 ANMS: oversample o 1-2, pool M 200-2000
log; ssc<o>p<M>), dppm (sigma = c x frame median depth, c 0.05-1 log, alpha 0.5-2 log; dppm<c>a<alpha>) and dppl
(per-point scale c x depth, c 0.08-1.5 log, alpha 0.5-2 log; dppl<c>a<alpha>); their first seeds are the dpp0.5 equivalents
(dppm0.256 = 0.5 m at the ~1.95 m DEV flat median candidate depth; dppl0.43 by pick overlap). Study name = <family>_<anything>.
Every trial also stores hull (SUB1 metric: mean inlier image-coverage share on the flat queries, no pose = 0) as the
uniformity guard; None if the metrics path fails, so the trial still counts.
Objective rep5c (SUB3, T4): a study name containing '_rep5c' maximises rep5 subject to fail_flat <= REP5_CAP (incumbent dpp0.5
DEV flat fail 14.41 %, env KPBENCH_REP5_CAP); value = -rep5, plus 1 + (fail_flat - cap) when the cap is broken, so every
feasible trial ranks above every infeasible one. Device noise (~1 point) makes the cap soft: the gate decides.
Objective fcap (SUB3, T4, dispatch 2026-10-20): a study name containing '_fcap' minimises fail_flat subject to rep5 >= REP5_FLOOR
(dpp0.5 DEV flat rep5 0.4527, env KPBENCH_REP5_FLOOR); value = fail_flat, plus 100 + 100 x (floor - rep5) when the floor is
broken (fail_flat <= 100, so every feasible trial ranks first). Family dpp2d (T11 (b): pixel-kernel DPP, sigma 20-200 px log,
alpha 0.2-5 log; dpp2d<sigma>a<alpha>; seed dpp2d74 = 0.5 m at the 2.15 m median pick depth, its DEV gate was +3.74 REVERT).
Summaries mark the Pareto front (lower fail_flat, higher rep5) of all complete trials.
"""
import os, json
import numpy as np
import optuna
from .evaluate import evaluate
from .bench import COLS
from .metrics import MCOLS
from .cache import REPO

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))
TUNE = os.path.join(RESULTS, 'tune')
SPACES = {'dpp': lambda t: 'dpp%.3ga%.3g' % (t.suggest_float('sigma', 0.1, 2.0, log=True), t.suggest_float('alpha', 0.2, 5.0, log=True)),
          'cube': lambda t: 'cube%.3g' % t.suggest_float('size', 0.15, 1.5, log=True),
          'pd': lambda t: 'pd%.3ga%.3g' % (t.suggest_float('r', 0.1, 1.5, log=True), t.suggest_float('theta', 0.0, 0.1)),
          'up': lambda t: 'upb%dr%d+dpp0.5p%d' % (t.suggest_categorical('b', [0, 4, 8, 12, 16, 24, 32]), t.suggest_int('r', 0, 12),
                                                  t.suggest_int('n', 150, 2000, log=True)),
          'win': lambda t: 'win%d+dpp%.3ga%.3g' % (t.suggest_int('N', 8, 32), t.suggest_float('sigma', 0.3, 0.7),
                                                  t.suggest_float('alpha', 0.5, 2.0, log=True)),
          'dppr': lambda t: 'dppr%.3ga%.3gb%.3gt%.2g' % (t.suggest_float('sigma', 0.25, 1.0, log=True), t.suggest_float('alpha', 0.0, 2.0),
                                                     t.suggest_float('beta', 0.0, 3.0), t.suggest_categorical('tau', [0.7, 0.8, 0.9])),
          'ssc': lambda t: 'ssc%.3gp%d' % (t.suggest_float('o', 1.0, 2.0), t.suggest_int('M', 200, 2000, log=True)),
          'dppm': lambda t: 'dppm%.3ga%.3g' % (t.suggest_float('c', 0.05, 1.0, log=True), t.suggest_float('alpha', 0.5, 2.0, log=True)),
          'dppl': lambda t: 'dppl%.3ga%.3g' % (t.suggest_float('c', 0.08, 1.5, log=True), t.suggest_float('alpha', 0.5, 2.0, log=True)),
          'nms': lambda t: 'nms%d' % t.suggest_int('r', 0, 16),
          'dpp2d': lambda t: 'dpp2d%.3ga%.3g' % (t.suggest_float('sigma', 20.0, 200.0, log=True), t.suggest_float('alpha', 0.2, 5.0, log=True))}
REP5_CAP = float(os.environ.get('KPBENCH_REP5_CAP', 14.41))
REP5_FLOOR = float(os.environ.get('KPBENCH_REP5_FLOOR', 0.4527))
SEEDS = {'dpp': [dict(sigma=0.5, alpha=1.0), dict(sigma=0.5, alpha=0.5), dict(sigma=0.3, alpha=1.0), dict(sigma=1.0, alpha=2.0)],
         'cube': [dict(size=0.5), dict(size=0.3)], 'pd': [dict(r=0.5, theta=0.0)],
         'up': [dict(b=0, r=0, n=2000), dict(b=0, r=6, n=2000), dict(b=16, r=0, n=2000), dict(b=0, r=0, n=500),
                dict(b=8, r=6, n=800), dict(b=0, r=4, n=2000)],
         'win': [dict(N=8, sigma=0.5, alpha=1.0), dict(N=16, sigma=0.5, alpha=1.0), dict(N=32, sigma=0.5, alpha=1.0)],
         'dppr': [dict(sigma=0.5, alpha=1.0, beta=0.0, tau=0.8), dict(sigma=0.5, alpha=1.0, beta=1.0, tau=0.8),
                  dict(sigma=0.5, alpha=1.0, beta=2.0, tau=0.8), dict(sigma=0.5, alpha=0.0, beta=1.0, tau=0.8)],
         'ssc': [dict(o=1.0, M=2000), dict(o=1.0, M=400), dict(o=1.5, M=2000)],
         'dppm': [dict(c=0.256, alpha=1.0), dict(c=0.15, alpha=1.0), dict(c=0.4, alpha=1.0), dict(c=0.6, alpha=1.0), dict(c=0.256, alpha=0.5)],
         'dppl': [dict(c=0.43, alpha=1.0), dict(c=0.256, alpha=1.0), dict(c=0.7, alpha=1.0), dict(c=1.0, alpha=1.0), dict(c=0.43, alpha=0.5)],
         'nms': [dict(r=0), dict(r=2), dict(r=4), dict(r=8), dict(r=16)],
         'dpp2d': [dict(sigma=74.0, alpha=1.0), dict(sigma=40.0, alpha=1.0), dict(sigma=120.0, alpha=1.0), dict(sigma=74.0, alpha=0.5)]}   # known points enqueued first


def study(name):
    os.makedirs(TUNE, exist_ok=True)
    fn = os.path.join(TUNE, name + '.journal'); J = optuna.storages.journal
    st = optuna.storages.JournalStorage(J.JournalFileBackend(fn, lock_obj=J.JournalFileOpenLock(fn)))   # open-lock: Windows + cluster FS
    return optuna.create_study(study_name=name, storage=st, direction='minimize', load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(n_startup_trials=8))


def family(name):
    return name.split('_')[0]


def rep5c(name):
    return '_rep5c' in name


def fcap(name):
    return '_fcap' in name


def score(fail_flat, rep5):
    """rep5c objective (minimised): -rep5 if fail_flat <= cap, else 1 - rep5 + (fail_flat - cap)"""
    return -rep5 + (1.0 + fail_flat - REP5_CAP if fail_flat > REP5_CAP else 0.0)


def score_fcap(fail_flat, rep5):
    """fcap objective (minimised): fail_flat if rep5 >= floor, else fail_flat + 100 + 100 (floor - rep5)"""
    return fail_flat + (100.0 + 100.0 * (REP5_FLOOR - rep5) if rep5 < REP5_FLOOR else 0.0)


def pareto(rows):
    """flag rows not dominated in (lower fail_flat, higher rep5); rows without rep5 are never on the front"""
    for r in rows:
        f, q = r['fail_flat'], r['rep5']
        r['pareto'] = q is not None and not any(o['rep5'] is not None and o['fail_flat'] <= f and o['rep5'] >= q and
                                                (o['fail_flat'] < f or o['rep5'] > q) for o in rows)
    return rows


def objective(trial, fam, rc=False, fc=False):
    pk = SPACES[fam](trial)
    try:
        out, rows, flat = evaluate(pk, split='dev_flat', metrics=True)
        h = np.array([np.nan if r[len(COLS) + MCOLS.index('hull')] is None else r[len(COLS) + MCOLS.index('hull')] for r in rows], float)
        hull = round(float(np.nan_to_num(h)[flat].mean()), 4) if flat.any() else None
    except Exception:
        (out, _, _), hull = evaluate(pk, split='dev_flat'), None
    for k in ('fail_flat', 'rep5', 'n_flat', 'secs'):
        trial.set_user_attr(k, out[k])
    trial.set_user_attr('hull', hull)
    trial.set_user_attr('picker', pk); trial.set_user_attr('host', os.uname().nodename.split('.')[0] if hasattr(os, 'uname') else '')
    if fc:
        return score_fcap(out['fail_flat'], out['rep5'])
    return score(out['fail_flat'], out['rep5']) if rc else out['fail_flat']


def enqueue_seeds(s, fam):
    if s.trials:
        return
    for params in SEEDS.get(fam, []):
        s.enqueue_trial(params)


def summary(name):
    s = study(name); done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value)
    rows = [dict(n=t.number, picker=t.user_attrs.get('picker'), fail_flat=t.user_attrs.get('fail_flat', t.value), rep5=t.user_attrs.get('rep5'),
                 hull=t.user_attrs.get('hull'), params=t.params, host=t.user_attrs.get('host')) for t in done]
    pareto(rows)
    out = dict(study=name, split='dev_flat', n_complete=len(done), n_total=len(s.trials), best=rows[:1], trials=rows,
               objective=('rep5 s.t. fail_flat <= %.2f' % REP5_CAP) if rep5c(name) else
               ('fail_flat s.t. rep5 >= %.4g' % REP5_FLOOR) if fcap(name) else 'fail_flat')
    json.dump(out, open(os.path.join(TUNE, name + '.json'), 'w'), indent=1)
    md = ['# Tuning %s (DEV flat queries, %d of %d trials done, objective %s)' % (name, len(done), len(s.trials), out['objective']), '',
          '| rank | picker | flat fail % | rep5 | hull | Pareto | trial |', '|---|---|---|---|---|---|---|']
    md += ['| %d | %s | %.2f | %s | %s | %s | %d |' % (i + 1, r['picker'], r['fail_flat'], r['rep5'], r['hull'], 'yes' if r['pareto'] else '', r['n'])
           for i, r in enumerate(rows[:15])]
    front = [r for r in rows[15:] if r['pareto']]
    md += ['| - | %s | %.2f | %s | %s | yes | %d |' % (r['picker'], r['fail_flat'], r['rep5'], r['hull'], r['n']) for r in front]
    open(os.path.join(TUNE, name + '.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True); ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--summary', action='store_true'); ap.add_argument('--seed-only', action='store_true')
    a = ap.parse_args(argv)
    fam = family(a.study); assert fam in SPACES, 'study name must start with a family: %s' % list(SPACES)
    if a.summary:
        return summary(a.study)
    s = study(a.study)
    if a.seed_only:
        return enqueue_seeds(s, fam)
    assert not (rep5c(a.study) and fcap(a.study)), 'one objective per study: _rep5c or _fcap'
    s.optimize(lambda t: objective(t, fam, rep5c(a.study), fcap(a.study)), n_trials=a.trials, catch=(Exception,))
    summary(a.study)


if __name__ == '__main__':
    main()
