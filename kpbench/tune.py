"""Optuna tuning of a picker family on DEV flat queries (week-1 step 4). No LLM; runs as an Euler array (euler/tune.sh).

  python -m kpbench.tune --study dpp_dev1 --trials 5          # one worker: 5 trials appended to the shared study
  python -m kpbench.tune --study dpp_dev1 --summary           # results/tune/<study>.json + .md (best trials)

Objective = fail % on the flat queries of the 12 DEV flights (936 queries, kpbench protocol, LightGlue). Repeatability at
lag 5 and the picker name are stored per trial. Held-out flights are never touched here: the best trial goes to the gate
(euler/gate.sh <picker> <incumbent> dev, then held_out). Trials run on different nodes, so LightGlue device noise
(~1 point on the flat bin, audit C4) stays in the objective; the gate reruns candidate and incumbent on one node.
Families: dpp (sigma 0.1-2 m log, alpha 0.2-5 log; dpp<sigma>a<alpha>), cube (size 0.15-1.5 m; cube<size>), pd (r 0.1-1.5 m,
theta 0-0.1; pd<r>a<theta>). Study name = <family>_<anything>.
"""
import os, json
import optuna
from .evaluate import evaluate
from .cache import REPO

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))
TUNE = os.path.join(RESULTS, 'tune')
SPACES = {'dpp': lambda t: 'dpp%.3ga%.3g' % (t.suggest_float('sigma', 0.1, 2.0, log=True), t.suggest_float('alpha', 0.2, 5.0, log=True)),
          'cube': lambda t: 'cube%.3g' % t.suggest_float('size', 0.15, 1.5, log=True),
          'pd': lambda t: 'pd%.3ga%.3g' % (t.suggest_float('r', 0.1, 1.5, log=True), t.suggest_float('theta', 0.0, 0.1))}
SEEDS = {'dpp': [dict(sigma=0.5, alpha=1.0), dict(sigma=0.5, alpha=0.5), dict(sigma=0.3, alpha=1.0), dict(sigma=1.0, alpha=2.0)],
         'cube': [dict(size=0.5), dict(size=0.3)], 'pd': [dict(r=0.5, theta=0.0)]}   # known points enqueued first


def study(name):
    os.makedirs(TUNE, exist_ok=True)
    fn = os.path.join(TUNE, name + '.journal'); J = optuna.storages.journal
    st = optuna.storages.JournalStorage(J.JournalFileBackend(fn, lock_obj=J.JournalFileOpenLock(fn)))   # open-lock: Windows + cluster FS
    return optuna.create_study(study_name=name, storage=st, direction='minimize', load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(n_startup_trials=8))


def family(name):
    return name.split('_')[0]


def objective(trial, fam):
    pk = SPACES[fam](trial)
    out, _, _ = evaluate(pk, split='dev_flat')
    for k in ('fail_flat', 'rep5', 'n_flat', 'secs'):
        trial.set_user_attr(k, out[k])
    trial.set_user_attr('picker', pk); trial.set_user_attr('host', os.uname().nodename.split('.')[0] if hasattr(os, 'uname') else '')
    return out['fail_flat']


def enqueue_seeds(s, fam):
    if s.trials:
        return
    for params in SEEDS.get(fam, []):
        s.enqueue_trial(params)


def summary(name):
    s = study(name); done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value)
    rows = [dict(n=t.number, picker=t.user_attrs.get('picker'), fail_flat=t.value, rep5=t.user_attrs.get('rep5'),
                 params=t.params, host=t.user_attrs.get('host')) for t in done]
    out = dict(study=name, split='dev_flat', n_complete=len(done), n_total=len(s.trials), best=rows[:1], trials=rows)
    json.dump(out, open(os.path.join(TUNE, name + '.json'), 'w'), indent=1)
    md = ['# Tuning %s (DEV flat queries, %d of %d trials done)' % (name, len(done), len(s.trials)), '',
          '| rank | picker | flat fail % | rep5 | trial |', '|---|---|---|---|---|']
    md += ['| %d | %s | %.2f | %s | %d |' % (i + 1, r['picker'], r['fail_flat'], r['rep5'], r['n']) for i, r in enumerate(rows[:15])]
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
    s.optimize(lambda t: objective(t, fam), n_trials=a.trials, catch=(Exception,))
    summary(a.study)


if __name__ == '__main__':
    main()
