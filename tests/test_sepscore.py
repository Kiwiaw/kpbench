"""kpbench.sepscore self-test on synthetic picks dumps (no flight data): 12 fake flights x 100 queries, q < 80 flat.

planted: off drives failure beyond the inlier count (fail logit = 0.5 + 1.2 h - 25 (off - 0.2), n_inl = 80 - 15 h + noise),
         and the uniform arm has more off-plane picks (0.30 vs other 0.20 vs strong 0.10)  -> all four rules pass.
null   : off is noise unrelated to failure and equal across arms (fail logit = -1 + 1.2 h) -> the read fails.
Also: sign-test counts of the addendum (9 of 12 fails, 10 of 12 passes, zeros dropped), and dump defects stop the run
before any value (missing picks, missing flight, different query sets, off missing on > 5 %).
"""
import gzip, json, os
import numpy as np
import pytest
from kpbench import sepscore, gate

FLIGHTS = ['env%d/P%03d' % (i % 3, i) for i in range(12)]
ARMS = ['uni', 'strong', 'other']


def write_arm(root, rid, base, planted, seed, picks=True, flights=FLIGHTS, nq=100, drop_off=0):
    d = os.path.join(root, 'runs', rid); os.makedirs(d, exist_ok=True)
    json.dump(dict(has_picks=picks), open(os.path.join(d, 'meta.json'), 'w'))
    for i, f in enumerate(flights):
        h = np.random.default_rng(1000 + i).normal(size=nq)                  # query difficulty, shared by the arms
        g = np.random.default_rng(seed * 100 + i)
        off = np.clip(base + g.normal(0, 0.06, nq), 0.0, 1.0) if planted else g.uniform(0.1, 0.4, nq)
        logit = 0.5 + 1.2 * h - 25 * (off - 0.2) if planted else -1 + 1.2 * h
        bad = g.random(nq) < 1 / (1 + np.exp(-logit)); ninl = np.clip(80 - 15 * h + g.normal(0, 10, nq), 0, None)
        rows = [[i, q, 300.0 if bad[q] else 5.0, 0.5, 100, int(100 * (1 - off[q])), int(ninl[q])] for q in range(nq)]   # n_match carries off: reading r[5] as n_inl kills rule 1
        op = {str(q): [None if q < drop_off else 1 - float(off[q]), 10, 0.9, [1] * 10, [10.0] * 10] for q in range(nq)}
        j = dict(rows=rows, picks={str(q): list(range(10)) for q in range(nq)} if picks else {}, onplane=op)
        with gzip.open(os.path.join(d, sepscore.slug(f) + '.json.gz'), 'wt') as fh:
            json.dump(j, fh)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'RESULTS', str(tmp_path)); monkeypatch.setattr(sepscore, 'RESULTS', str(tmp_path))
    lab = {str(q): dict(plane_frac=0.9 if q < 80 else 0.5) for q in range(100)}
    monkeypatch.setattr(sepscore, 'labels', lambda f: lab)
    return str(tmp_path)


def run(root, planted):
    for k, (a, base) in enumerate(zip(ARMS, (0.30, 0.10, 0.20))):
        write_arm(root, a, base, planted, seed=k + 1)
    T, keys, miss = sepscore.validate(ARMS, FLIGHTS)
    return sepscore.score(T, keys, ARMS), miss


def test_planted_effect_passes(root):
    r, miss = run(root, True)
    assert miss == 0 and r['n'] == 1200 and r['n_flat'] == 960
    assert r['rule1'] and r['rule2'] and r['rule3'] and r['rule4'] and r['passed'], r
    assert r['d_mean'] > 0.15 and r['sep']['uni'] < 0 and r['flight']['neg'] >= 10
    assert 'PASS' in sepscore.to_md(r, 'dev', miss)


def test_null_fails(root):
    r, _ = run(root, False)
    assert not r['passed'] and not r['rule1'] and not r['rule2'] and not r['rule3'] and not r['rule4'], r
    assert abs(r['d_mean']) < 0.02 and 'FAIL' in r['verdict']


def test_sign_counts():
    assert sepscore.sign_p(9, 12) > 0.05 and sepscore.sign_p(10, 12) < 0.05          # 0.073, 0.019
    assert sepscore.sign_p(8, 10) > 0.05 and sepscore.sign_p(9, 10) < 0.05 and sepscore.sign_p(5, 5) < 0.05
    mk = lambda seps: ({('f%d' % i, q): (int(q < 5), 0.2 + (s if q < 5 else 0.0), 50.0, True) for i, s in enumerate(seps) for q in range(10)})
    t = mk([-0.1] * 9 + [0.1] * 3); c = sepscore.flight_check(t, sorted(t))
    assert (c['neg'], c['pos'], c['zero'], c['ok']) == (9, 3, 0, False)
    t = mk([-0.1] * 10 + [0.1] * 2); assert sepscore.flight_check(t, sorted(t))['ok']
    t.update({(k[0], q + 100): (int(q < 5), 0.2 + (0.5 if q < 5 else 0.0), 50.0, False) for k in list(t) for q in range(10)})   # non-flat rows with the opposite sign are ignored
    assert sepscore.flight_check(t, sorted(t))['neg'] == 10
    t = mk([-0.1] * 5 + [0.0] * 7); c = sepscore.flight_check(t, sorted(t))            # zeros dropped: 5 of 5
    assert (c['neg'], c['zero'], c['ok']) == (5, 7, True)
    t = mk([-0.1] * 4 + [0.0] * 8); assert not sepscore.flight_check(t, sorted(t))['ok']   # 4 flights left: not evaluable


def test_dump_defects_stop_before_any_value(root):
    for k, a in enumerate(ARMS):
        write_arm(root, a, 0.2, True, seed=k + 1)
    write_arm(root, 'nopicks', 0.2, True, seed=9, picks=False)
    write_arm(root, 'short', 0.2, True, seed=9, flights=FLIGHTS[:11])
    write_arm(root, 'fewq', 0.2, True, seed=9, nq=90)
    write_arm(root, 'nooff', 0.2, True, seed=9, drop_off=10)                           # 10 % of queries without off
    for bad in ('nopicks', 'short', 'fewq', 'nooff'):
        with pytest.raises(gate.GateError):
            sepscore.validate(['uni', 'strong', bad], FLIGHTS)
    with pytest.raises(gate.GateError):
        sepscore.validate(['uni', 'strong'], FLIGHTS)
    with pytest.raises(SystemExit):
        sepscore.main(['--arms', 'uni,strong,nopicks', '--flights', ','.join(FLIGHTS), '--check'])
    assert sepscore.main(['--arms', ','.join(ARMS), '--flights', ','.join(FLIGHTS), '--check']) is None


def test_read_guards(root, monkeypatch):
    names = ['raco_dpp0.5_K100_same_lg_picks', 'raco_sd_K100_same_lg_picks', 'raco_cube0.5_K100_same_lg_picks']
    for k, (a, base) in enumerate(zip(names, (0.30, 0.10, 0.20))):
        write_arm(root, a, base, True, seed=k + 1)
    monkeypatch.setattr(sepscore, 'split_info', lambda split: (FLIGHTS, 960, None))
    T, keys, _ = sepscore.validate(names, FLIGHTS)
    sepscore.check_read(names, 'dev', None, T, FLIGHTS)                                 # the registered call passes
    for arms, split in ((names[::-1], 'dev'), ([names[1], names[0], names[2]], 'dev'), (names, 'fresh')):
        with pytest.raises(gate.GateError):
            sepscore.check_read(arms, split, None, T, FLIGHTS)
    monkeypatch.setattr(sepscore, 'split_info', lambda split: (FLIGHTS, 961, None))    # a label went missing
    with pytest.raises(gate.GateError):
        sepscore.check_read(names, 'dev', None, T, FLIGHTS)
