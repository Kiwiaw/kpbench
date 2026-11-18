"""kpbench.pickarms on synthetic --save-picks runs with a known answer (no flight data).

2 flights x 10 queries, q 0-5 flat (plane_frac 0.9), 6-9 not flat. Every pick list has 10 picks with depth.
  ref : fails q 0, 1, 2; off 0.10 everywhere            -> offn 1.0, depths 10 x [1]*10 -> dspan 0
  cand: fails q 0;       off 0.30 on posed, 0.50 on failed -> flat median off 0.30, sep = 0.50 - 0.30 = +0.20
        depths 10 .. 100 m (10 values)                     -> dspan (91 - 19) / 55 = 1.309
Known answer: n_flat 12 per arm, cand - ref off mean (0.5 x 2 + 0.3 x 10)/12 - 0.1 = +0.2333, offn = 10 x that, up 2/2 flights,
CI of a constant per-flight diff = that value; ref sep 0 (same off on failed and posed). Third arm without picks is refused.
"""
import gzip, json, os
import pytest
from kpbench import pickarms, gate

FLIGHTS = ['envA/P000', 'envB/P001']


def write_run(root, rid, fail_q, off_of, depths, picks=True):
    d = os.path.join(root, 'runs', rid); os.makedirs(d, exist_ok=True)
    json.dump(dict(has_picks=picks), open(os.path.join(d, 'meta.json'), 'w'))
    for f in FLIGHTS:
        rows = [[f, q, 300.0 if q in fail_q else 5.0, 0.5] for q in range(10)]
        j = dict(rows=rows, picks={str(q): list(range(10)) for q in range(10)} if picks else {},
                 onplane={str(q): [1 - off_of(q), 10, 0.9, [1] * 10, depths] for q in range(10)})
        with gzip.open(os.path.join(d, pickarms.slug(f) + '.json.gz'), 'wt') as fh:
            json.dump(j, fh)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'RESULTS', str(tmp_path)); monkeypatch.setattr(pickarms, 'RESULTS', str(tmp_path))
    monkeypatch.setattr(pickarms, 'split_info', lambda split: (FLIGHTS, None, None))
    lab = {str(q): dict(plane_frac=0.9 if q < 6 else 0.5) for q in range(10)}
    monkeypatch.setattr(pickarms, 'labels', lambda f: lab)
    write_run(str(tmp_path), 'ref', {0, 1, 2}, lambda q: 0.10, [10.0] * 10)
    write_run(str(tmp_path), 'cand', {0}, lambda q: 0.50 if q == 0 else 0.30, [10.0 * i for i in range(1, 11)])
    return tmp_path


def test_known_answer(fake):
    r = pickarms.compare_arms(['ref', 'cand'], 'dev', n_boot=500)
    ref, cand = r['arms']
    assert (ref['n_flat'], ref['fail'], cand['n_flat'], cand['fail']) == (12, 6, 12, 2)
    assert abs(cand['median']['off'] - 0.30) < 0.01 and abs(ref['median']['offn'] - 1.0) < 1e-9
    assert ref['median']['dspan'] == 0.0 and abs(cand['median']['dspan'] - 1.3091) < 1e-3
    assert ref['sep'] == 0.0 and abs(cand['sep'] - 0.20) < 1e-9
    v = cand['vs_ref']
    assert abs(v['off']['mean'] - 0.2333) < 1e-4 and abs(v['offn']['mean'] - 2.3333) < 1e-4
    assert (v['off']['up'], v['off']['down'], v['off']['flights']) == (2, 0, 2)
    assert abs(v['off']['ci'][0] - 0.2333) < 1e-4 and abs(v['off']['ci'][1] - 0.2333) < 1e-4
    assert 'cand' in pickarms.to_md(r)


def test_arm_without_picks_refused(fake):
    write_run(str(fake), 'nopicks', {0}, lambda q: 0.3, [10.0] * 10, picks=False)
    with pytest.raises(gate.GateError):
        pickarms.compare_arms(['ref', 'nopicks'], 'dev', n_boot=100)
