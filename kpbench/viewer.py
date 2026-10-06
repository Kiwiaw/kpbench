"""Frame viewer: a standalone HTML page that shows one frame in 3D with the keypoints each picker keeps.

  python -m kpbench.viewer --flight oldtown/P001 --frame 334 [--det raco] [--pickers sd,cube0.5,far2d,far3d]
                           [--kmax 2000] [--step 4]
-> docs/frames/<slug>_<frame>.html (plotly.js from the CDN, all data inline as JSON).

Plot axes follow the usual convention: x = right (camera x), y = forward (camera z), z = up (-camera y).
"""
import argparse
import base64
import io
import json
import os

import numpy as np
from PIL import Image

from .data import Flight, to3d, DEPTH_MAX, W, H, FX, FY
from .cache import Cands
from .pickers import parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'docs', 'frames')
PLOTLY = 'https://cdn.plot.ly/plotly-2.35.2.min.js'
# Pickers whose selection for K is NOT the first K entries of the selection for kmax (stored per K).
NOT_PREFIX = ('grid',)


def plot_axes(P):
    """camera frame (x right, y down, z forward) -> plot axes (x right, y forward, z up)"""
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def r3(a):
    return [round(float(x), 3) for x in a]


def build(fid, f, det='raco', pickers=('sd', 'cube0.5', 'far2d', 'far3d'), kmax=2000, step=4):
    F = Flight(fid)
    img = F.img(f)
    dep = F.depth(f)

    # 1. point cloud: every step-th pixel with valid depth
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    uu = uu.ravel().astype(np.float64); vv = vv.ravel().astype(np.float64)
    zz = dep[vv.astype(int), uu.astype(int)]
    ok = (zz > 0) & (zz < DEPTH_MAX)
    P = plot_axes(to3d(uu[ok], vv[ok], zz[ok]))
    rgb = img[vv[ok].astype(int), uu[ok].astype(int)]
    cloud = {'x': r3(P[:, 0]), 'y': r3(P[:, 1]), 'z': r3(P[:, 2]),
             'c': ['#%02x%02x%02x' % tuple(int(c) for c in p) for p in rgb]}

    # 2. candidates with valid depth (rank = index in score order)
    u, v, s, z, _ = Cands(F.slug, det).frame(f)
    valid = np.flatnonzero((z > 0) & (z < DEPTH_MAX))
    pos = {int(j): i for i, j in enumerate(valid)}       # candidate index -> position in the embedded list
    Pc = to3d(u[valid], v[valid], z[valid])
    Q = plot_axes(Pc)
    cands = {'u': [round(float(x), 2) for x in u[valid]], 'v': [round(float(x), 2) for x in v[valid]],
             'x': r3(Q[:, 0]), 'y': r3(Q[:, 1]), 'z': r3(Q[:, 2]),
             'cx': r3(Pc[:, 0]), 'cy': r3(Pc[:, 1]), 'cz': r3(Pc[:, 2]),   # camera frame, for the 0.5 m cubes
             'rank': [int(j) for j in valid], 'd': [round(float(x), 2) for x in z[valid]],
             'col': ['#%02x%02x%02x' % tuple(int(c) for c in img[int(min(H - 1, max(0, round(b)))), int(min(W - 1, max(0, round(a))))]) for a, b in zip(u[valid], v[valid])]}

    # 3. selections. sd, cube<s>, far2d/far3d are prefix-consistent: the first K entries of the selection for
    # kmax are exactly the selection for K, so one ordered list per picker covers every K. grid is not, so its
    # selection is stored separately for every K (perK[name][K-1]). Indices point into the embedded candidate list;
    # candidates without depth (possible only through a picker's top-up fill) are dropped.
    sel, perK, dropped = {}, {}, {}
    for name in pickers:
        fn = parse(name)
        idx = fn(u, v, s, z, kmax)
        sel[name] = [pos[int(j)] for j in idx if int(j) in pos]
        dropped[name] = len(idx) - len(sel[name])
        if name in NOT_PREFIX:
            perK[name] = [[pos[int(j)] for j in fn(u, v, s, z, K) if int(j) in pos] for K in range(1, kmax + 1)]

    # 4. image as base64 JPEG
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=80)
    jpg = base64.b64encode(buf.getvalue()).decode('ascii')

    data = {'flight': fid, 'frame': int(f), 'det': det, 'kmax': int(kmax), 'step': int(step),
            'W': W, 'H': H, 'fx': FX, 'fy': FY, 'pickers': list(pickers), 'cloud': cloud, 'cands': cands,
            'sel': sel, 'perK': perK, 'dropped': dropped}
    return data, jpg


def render(data, jpg):
    title = '%s frame %d (%s)' % (data['flight'], data['frame'], data['det'])
    blob = json.dumps(data, separators=(',', ':'))
    return (HTML.replace('__TITLE__', title).replace('__PLOTLY__', PLOTLY)
            .replace('__JPG__', jpg).replace('__DATA__', blob))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY__"></script>
<style>
  body { margin: 0; padding: 10px 14px; background: #fafafa; color: #222;
         font: 13px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  h1 { font-size: 15px; margin: 0 0 6px; font-weight: 600; }
  #controls { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; margin-bottom: 6px; }
  #controls label { white-space: nowrap; cursor: pointer; }
  #controls fieldset { border: 1px solid #ddd; border-radius: 4px; padding: 2px 8px; margin: 0; }
  #controls legend { font-size: 11px; color: #666; padding: 0 3px; }
  #kslider { width: 260px; vertical-align: middle; }
  #kbox { width: 56px; }
  #readout { font-family: ui-monospace, Consolas, monospace; font-size: 12px; color: #333; margin-bottom: 6px; }
  #panels { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
  #scene { flex: 1 1 560px; min-width: 300px; height: 600px; background: #fff; border: 1px solid #ddd; }
  #imgwrap { flex: 0 1 640px; min-width: 280px; }
  #img { width: 100%; max-width: 640px; height: auto; border: 1px solid #ddd; background: #fff; display: block; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div id="controls">
  <fieldset><legend>picker</legend><span id="pickers"></span></fieldset>
  <span>K <input id="kslider" type="range" min="1" step="1"> <input id="kbox" type="number" min="1" step="1"></span>
  <label><input type="checkbox" id="cubes"> show 0.5 m cubes</label>
  <label><input type="checkbox" id="cloud" checked> show point cloud</label>
  <label><input type="checkbox" id="allc"> show all candidates</label>
  <label><input type="checkbox" id="byrank"> colour selected by rank (default: pixel colour)</label>
  <span title="points farther forward than this are outside the 3D view (they stay in the statistics)">view depth <input id="dmax" type="range" min="2" step="0.5"> <span id="dmaxv"></span> m</span>
</div>
<div id="readout"></div>
<div id="panels">
  <div id="scene"></div>
  <div id="imgwrap"><canvas id="img" width="640" height="480"></canvas></div>
</div>
<script>
const D = __DATA__;
const JPG = "data:image/jpeg;base64,__JPG__";
const C = D.cands, NC = C.u.length, CUBE = 0.5;
const st = { picker: D.pickers.includes('sd') ? 'sd' : D.pickers[0], K: Math.min(100, D.kmax), dmax: 0 };
// default view depth: twice the median candidate depth, between 4 and 30 m, so the near scene fills the view
{ const ys = C.y.slice().sort((a, b) => a - b); st.dmax = Math.round(Math.min(30, Math.max(4, 2 * ys[ys.length >> 1])) * 2) / 2; }

// trace order in the 3D scene
const T_CLOUD = 0, T_ALL = 1, T_CUBES = 2, T_CAM = 3, T_SEL = 4;

function selection() {
  const pk = D.perK[st.picker];
  if (pk) return pk[Math.min(st.K, pk.length) - 1] || [];
  return D.sel[st.picker].slice(0, st.K);       // prefix-consistent pickers
}

// camera pyramid at the origin (plot axes: x right, y forward, z up), depth 0.5 m
function camTrace() {
  const d = 0.5, hx = D.W / 2 / D.fx * d, hz = D.H / 2 / D.fy * d;
  const c = [[-hx, d, hz], [hx, d, hz], [hx, d, -hz], [-hx, d, -hz]];
  const x = [], y = [], z = [];
  const seg = (a, b) => { x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null); };
  for (let i = 0; i < 4; i++) { seg([0, 0, 0], c[i]); seg(c[i], c[(i + 1) % 4]); }
  seg([-hx, d, hz], [0, d, hz * 1.5]); seg([0, d, hz * 1.5], [hx, d, hz]);   // "up" tick on top
  return { type: 'scatter3d', mode: 'lines', x, y, z, line: { color: '#d62728', width: 4 }, hoverinfo: 'skip', name: 'camera' };
}

function cubeKeys(S) {
  const keys = new Map();
  for (const i of S) {
    const k = [Math.floor(C.cx[i] / CUBE), Math.floor(C.cy[i] / CUBE), Math.floor(C.cz[i] / CUBE)];
    keys.set(k.join(','), k);
  }
  return keys;
}

function cubeLines(keys) {
  const x = [], y = [], z = [];
  // camera frame corner -> plot axes
  const P = (a, b, c) => [a * CUBE, c * CUBE, -b * CUBE];
  for (const [i, j, k] of keys.values()) {
    const v = [];
    for (let b = 0; b < 8; b++) v.push(P(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)));
    for (let a = 0; a < 8; a++) for (const bit of [1, 2, 4]) if (!(a & bit)) {
      const p = v[a], q = v[a | bit];
      x.push(p[0], q[0], null); y.push(p[1], q[1], null); z.push(p[2], q[2], null);
    }
  }
  return { x, y, z };
}

function median(a) {
  if (!a.length) return NaN;
  const s = a.slice().sort((p, q) => p - q), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function nnStats(S) {
  const px = [], m = [];
  for (const i of S) {
    let bp = Infinity, bm = Infinity;
    for (const j of S) if (j !== i) {
      const du = C.u[i] - C.u[j], dv = C.v[i] - C.v[j];
      const dx = C.cx[i] - C.cx[j], dy = C.cy[i] - C.cy[j], dz = C.cz[i] - C.cz[j];
      bp = Math.min(bp, du * du + dv * dv); bm = Math.min(bm, dx * dx + dy * dy + dz * dz);
    }
    if (bp < Infinity) { px.push(Math.sqrt(bp)); m.push(Math.sqrt(bm)); }
  }
  return [median(px), median(m)];
}

function rankColour(r) {   // rank 0..kmax -> viridis-like ramp (dark purple = strongest pick)
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const t = Math.min(1, r / Math.max(1, D.kmax - 1)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return 'rgb(' + c.join(',') + ')';
}

// ---- 2D panel
const cv = document.getElementById('img'), ctx = cv.getContext('2d'), im = new Image();
im.onload = draw2d; im.src = JPG;

function draw2d() {
  ctx.drawImage(im, 0, 0, D.W, D.H);
  if (document.getElementById('allc').checked) {
    ctx.fillStyle = 'rgba(150,150,150,0.85)';
    for (let i = 0; i < NC; i++) ctx.fillRect(C.u[i] - 1, C.v[i] - 1, 2, 2);
  }
  const S = selection(), byr = document.getElementById('byrank').checked;
  ctx.lineWidth = 2;
  S.forEach((i, n) => {
    ctx.beginPath(); ctx.arc(C.u[i], C.v[i], 3.5, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.strokeStyle = byr ? rankColour(n) : C.col[i]; ctx.lineWidth = 1.2; ctx.stroke();
  });
}

// ---- 3D scene
function selTrace(S) {
  const byr = document.getElementById('byrank').checked;
  return {
    x: S.map(i => C.x[i]), y: S.map(i => C.y[i]), z: S.map(i => C.z[i]),
    text: S.map((i, n) => 'pick ' + (n + 1) + '<br>rank ' + C.rank[i] + '<br>depth ' + C.d[i] + ' m'),
    color: byr ? S.map((i, n) => rankColour(n)) : S.map(i => C.col[i])
  };
}

function extent(a, b, ma, mb) {
  let lo = Infinity, hi = -Infinity;
  for (const [arr, m] of [[a, ma], [b, mb]]) arr.forEach((t, i) => { if (m[i]) { if (t < lo) lo = t; if (t > hi) hi = t; } });
  return [lo - CUBE, hi + CUBE];
}

// axis ranges for the points with forward distance <= dmax, and the matching aspect ratio
function sceneRanges(dmax) {
  const mc = D.cloud.y.map(y => y <= dmax), mk = C.y.map(y => y <= dmax);
  const rx = extent(D.cloud.x, C.x, mc, mk), ry = extent(D.cloud.y, C.y, mc, mk), rz = extent(D.cloud.z, C.z, mc, mk);
  rx[0] = Math.min(rx[0], -0.5); ry[0] = Math.min(ry[0], -0.5); rz[0] = Math.min(rz[0], -0.5);
  const sx = rx[1] - rx[0], sy = ry[1] - ry[0], sz = rz[1] - rz[0], mx = Math.max(sx, sy, sz);
  return { rx, ry, rz, aspect: { x: sx / mx, y: sy / mx, z: sz / mx } };
}

function init() {
  const S = selection(), s = selTrace(S), cb = cubeLines(cubeKeys(S));
  const traces = [
    { type: 'scatter3d', mode: 'markers', x: D.cloud.x, y: D.cloud.y, z: D.cloud.z, name: 'cloud',
      marker: { size: 1.5, color: D.cloud.c }, hoverinfo: 'skip', visible: true },
    { type: 'scatter3d', mode: 'markers', x: C.x, y: C.y, z: C.z, name: 'candidates',
      marker: { size: 2, color: '#888' }, hoverinfo: 'skip', visible: false },
    { type: 'scatter3d', mode: 'lines', x: cb.x, y: cb.y, z: cb.z, name: 'cubes',
      line: { color: '#777', width: 1.5 }, hoverinfo: 'skip', visible: false },
    camTrace(),
    { type: 'scatter3d', mode: 'markers', x: s.x, y: s.y, z: s.z, text: s.text, name: 'selected',
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 3.5, color: s.color, line: { color: '#000', width: 0.8 } } }
  ];
  // Fixed axis ranges (points within the view depth, padded by one cube) so adding points or cubes never rescales
  // the scene; the aspect ratio is set to the range proportions, i.e. what aspectmode 'data' would give for this data.
  const { rx, ry, rz, aspect } = sceneRanges(st.dmax);
  const ax = (t, r) => ({ title: t, range: r, backgroundcolor: '#fff', gridcolor: '#e5e5e5', zerolinecolor: '#bbb' });
  const layout = {
    uirevision: 'keep', margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, paper_bgcolor: '#fff',
    scene: {
      uirevision: 'keep',
      xaxis: ax('x right (m)', rx), yaxis: ax('y forward (m)', ry), zaxis: ax('z up (m)', rz),
      aspectmode: 'manual', aspectratio: aspect,
      camera: { eye: { x: 0.9, y: -1.5, z: 0.8 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: -0.1 } }
    }
  };
  Plotly.newPlot('scene', traces, layout, { responsive: true, displaylogo: false });
}

function update() {
  const S = selection(), s = selTrace(S), keys = cubeKeys(S);
  Plotly.restyle('scene', { x: [s.x], y: [s.y], z: [s.z], text: [s.text], 'marker.color': [s.color] }, [T_SEL]);
  if (document.getElementById('cubes').checked) {
    const cb = cubeLines(keys);
    Plotly.restyle('scene', { x: [cb.x], y: [cb.y], z: [cb.z], visible: true }, [T_CUBES]);
  } else {
    Plotly.restyle('scene', { visible: false }, [T_CUBES]);
  }
  const [npx, nm] = nnStats(S);
  document.getElementById('readout').textContent =
    'K selected ' + S.length + ' | occupied 0.5 m cubes ' + keys.size +
    ' | median NN distance ' + (isNaN(npx) ? '-' : npx.toFixed(1)) + ' px, ' + (isNaN(nm) ? '-' : nm.toFixed(2)) + ' m' +
    ' | candidates with depth ' + NC + (D.dropped[st.picker] ? ' | ' + D.dropped[st.picker] + ' picks without depth hidden' : '');
  draw2d();
}

// ---- controls
const ks = document.getElementById('kslider'), kb = document.getElementById('kbox');
ks.max = kb.max = D.kmax; ks.value = kb.value = st.K;
function setK(k) {
  k = Math.max(1, Math.min(D.kmax, Math.round(+k || 1)));
  if (k === st.K && +ks.value === k && +kb.value === k) return;
  st.K = k; ks.value = k; kb.value = k; update();
}
ks.addEventListener('input', () => setK(ks.value));
kb.addEventListener('input', () => setK(kb.value));
const box = document.getElementById('pickers');
for (const p of D.pickers) {
  const l = document.createElement('label'), r = document.createElement('input');
  r.type = 'radio'; r.name = 'picker'; r.value = p; r.checked = p === st.picker;
  r.addEventListener('change', () => { st.picker = p; update(); });
  l.append(r, ' ' + p + ' '); box.append(l);
}
document.getElementById('cloud').addEventListener('change', e => Plotly.restyle('scene', { visible: e.target.checked }, [T_CLOUD]));
document.getElementById('allc').addEventListener('change', e => { Plotly.restyle('scene', { visible: e.target.checked }, [T_ALL]); draw2d(); });
document.getElementById('cubes').addEventListener('change', update);
const ds = document.getElementById('dmax'), dv = document.getElementById('dmaxv');
ds.max = Math.ceil(Math.max(...C.y, ...D.cloud.y)); ds.value = st.dmax; dv.textContent = st.dmax;
ds.addEventListener('input', () => {
  st.dmax = +ds.value; dv.textContent = st.dmax;
  const { rx, ry, rz, aspect } = sceneRanges(st.dmax);
  Plotly.relayout('scene', { 'scene.xaxis.range': rx, 'scene.yaxis.range': ry, 'scene.zaxis.range': rz, 'scene.aspectratio': aspect });
});
document.getElementById('byrank').addEventListener('change', update);

init(); update();
</script>
</body>
</html>
'''


def main():
    ap = argparse.ArgumentParser(description='standalone 3D frame viewer with picker selections')
    ap.add_argument('--flight', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--det', default='raco')
    ap.add_argument('--pickers', default='sd,cube0.5,far2d,far3d,far3dw,far2dp400')
    ap.add_argument('--kmax', type=int, default=2000)
    ap.add_argument('--step', type=int, default=4)
    a = ap.parse_args()
    pickers = [p.strip() for p in a.pickers.split(',') if p.strip()]
    data, jpg = build(a.flight, a.frame, a.det, pickers, a.kmax, a.step)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, '%s_%d.html' % (data['flight'].replace('/', '_'), a.frame))
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(render(data, jpg))
    print('%s  %.2f MB  cloud %d  candidates %d  picks %s' % (out, os.path.getsize(out) / 1e6, len(data['cloud']['x']),
          len(data['cands']['u']), {k: len(v) for k, v in data['sel'].items()}))


if __name__ == '__main__':
    main()
