"""Frozen flight split for picker search (11 Oct 2026). Writes splits.json once; never regenerated.

  python -m kpbench.splits          # creates splits.json if missing, else prints it

DEV       12 flights: where pickers are designed and tuned (Optuna, gate stage 1).
HELD_OUT  19 flights: the exam. A candidate is run on them once, by the gate, after it beat the incumbent on DEV.
MICRO     the flat queries (plane_frac >= 0.85) of oldtown/P001 and hospital/P041: the fast laptop check (code search).

Rule for DEV: every flight already used to design or tune a picker before the split is DEV (contaminated):
oldtown P000 P002 P006 P007 (learned re-ranker training), oldtown P001 + hospital P041 (all two-flight picker work,
DPP sigma chosen on them). The remaining DEV slots (3 hospital, 3 factory) are drawn with seed 0 from the flights
with < 50 flat queries, so HELD_OUT keeps the flat-rich flights and the flat-bin gate keeps its power.
"""
import os, json
import numpy as np
from .data import FLIGHTS, slug
from .cache import REPO

SPLITS = os.path.join(REPO, 'splits.json')
CONTAMINATED = ['oldtown/P000', 'oldtown/P001', 'oldtown/P002', 'oldtown/P006', 'oldtown/P007', 'hospital/P041']
MICRO_FLIGHTS = ['oldtown/P001', 'hospital/P041']
FLAT = 0.85


def flat_queries(fid):
    d = json.load(open(os.path.join(REPO, 'results', 'labels', slug(fid) + '.json')))
    return sorted(int(q) for q, v in d.items() if v.get('plane_frac') is not None and v['plane_frac'] >= FLAT)


def make():
    rng = np.random.default_rng(0)
    nflat = {f: len(flat_queries(f)) for f in FLIGHTS}
    dev = list(CONTAMINATED)
    for env, n in (('hospital', 3), ('factory', 3)):
        pool = [f for f in FLIGHTS if f.startswith(env + '/') and f not in dev and nflat[f] < 50]
        dev += [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
    dev = [f for f in FLIGHTS if f in dev]
    held = [f for f in FLIGHTS if f not in dev]
    micro = {f: flat_queries(f) for f in MICRO_FLIGHTS}
    return dict(created='2026-10-11', rule=__doc__.split('Rule for DEV:')[1].strip(), dev=dev, held_out=held, micro=micro,
                flat_counts=dict(dev=sum(nflat[f] for f in dev), held_out=sum(nflat[f] for f in held),
                                 micro=sum(len(v) for v in micro.values())))


def load():
    return json.load(open(SPLITS))


def main():
    if os.path.isfile(SPLITS):
        s = load(); print('frozen', SPLITS)
    else:
        s = make(); json.dump(s, open(SPLITS, 'w'), indent=1); print('created', SPLITS)
    print('dev', len(s['dev']), s['dev']); print('held_out', len(s['held_out'])); print('flat', s['flat_counts'])


if __name__ == '__main__':
    main()
