"""Frame viewer: a standalone HTML page that shows one frame in 3D with the keypoints each picker keeps.

  python -m kpbench.viewer --flight oldtown/P001 --frame 334 [--det raco] [--pickers sd,cube0.5,far2d,far3d]
                           [--kmax 2000] [--step 4]
-> docs/frames/<slug>_<frame>.html (plotly.js from the CDN, all data inline as JSON).

DPP pages (fresh stock-RaCo run, 3000 candidates with depth, slider = first k picks of each picker's order):
  python -m kpbench.viewer --flight oldtown/P001 --frame 334 --ncand 3000 --pickers dpp0.5,sd,cube0.5 --default dpp0.5
                           --out dpp_tartanair_oldtown_P001_334.html --title "..."
  Hypersim: set KPBENCH_INTR="554.256258 554.256258 320 240" first (see data.py).

Plot axes follow the usual convention: x = right (camera x), y = forward (camera z), z = up (-camera y).
"""
import argparse
import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

from .data import Flight, to3d, DEPTH_MAX, W, H, FX, FY, CX, CY
from .cache import Cands
from .pickers import parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'docs', 'frames')
PLOTLY = 'https://cdn.plot.ly/plotly-2.35.2.min.js'
# Pickers whose selection for K is NOT the first K entries of the selection for kmax (stored per K).
NOT_PREFIX = ('grid',)
# plain picker names and colours for --ncand pages
LABELS = {'dpp0.5': 'DPP', 'sd': 'strongest first', 'cube0.5': 'cubes 0.5 m'}
COLOURS = {'dpp0.5': '#1f9bff', 'sd': '#ff3b30', 'cube0.5': '#ff9500'}
DETNAMES = {'raco': 'RaCo', 'sp': 'SuperPoint', 'aliked': 'ALIKED'}
DET_COLOURS = {'raco': '#ff3b30', 'sp': '#ffd60a', 'aliked': '#00c7be'}


def plain_label(key, det, zc, multi):
    """picker key -> plain name (and colour) for --ncand pages; zc = median candidate depth that scaled pickers use"""
    d, name = key.split(':', 1) if ':' in key else (det, key)
    m = re.fullmatch(r'dppmr([0-9.]+)a[0-9.]+b[0-9.]+t[0-9.]+', name)
    if m:
        c = float(m.group(1))
        return 'DPP %g\u00d7median depth (%.2f m) + re-found bonus' % (c, c * zc), '#b04dff'
    m = re.fullmatch(r'cubem([0-9.]+)', name)
    if m:
        c = float(m.group(1))
        return 'cubes %g\u00d7median depth (%.2f m)' % (c, c * zc), '#34c759'
    if name in ('sd', 'score') and multi:
        return DETNAMES.get(d, d) + ' strongest', DET_COLOURS.get(d, '#ff3b30')
    m = re.fullmatch(r'dpp([0-9.]+)', name)
    if m and multi:
        return 'DPP %s m' % m.group(1), COLOURS['dpp0.5']
    return LABELS.get(name, name), COLOURS.get(name, '#ffd400')


def plot_axes(P):
    """camera frame (x right, y down, z forward) -> plot axes (x right, y forward, z up)"""
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def r3(a):
    return [round(float(x), 3) for x in a]


class Detectors:
    """Fresh detector runs for one page (instead of the 2000-point caches). cands(det, img, dep, n): the top m detections in
    score order (RaCo: ranker score, as extract.py; SuperPoint: keypoint score, nms 4, no threshold, as extract.py; ALIKED:
    aliked-n16 keypoint score, nms 2, top-k), m grown until n of them have depth -> (u, v, s, z) of those n in score order.
    desc(img, u, v): SuperPoint's dense descriptor sampled at the points (as the cache). past(img): RaCo's strongest 2000 and
    their descriptors exactly as extract.py caches them (for dppmr's track bonus over the previous frames)."""
    def __init__(self):
        import torch
        from . import extract
        self.torch = torch
        self.dev = extract.DEV if torch.cuda.is_available() else 'cpu'
        extract.DEV = self.dev
        self.sp_dense, self.sp_sample, self.det2000 = extract.load_models({'raco'})
        self.raco_dir = extract.RACO_DIR
        self.models = {}

    def _run(self, det, img, m):
        torch = self.torch
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.dev)
        with torch.no_grad():
            if det == 'raco':
                net = torch.hub.load(self.raco_dir, 'raco', source='local', pretrained=True, max_num_keypoints=m,
                                     ranker=True, covariance_estimator=True).eval().to(self.dev)
                p = net({'image': x}); sc = p['ranker_scores'][0]
            else:
                if det not in self.models:
                    from lightglue import SuperPoint, ALIKED
                    net = (SuperPoint(max_num_keypoints=None, detection_threshold=0.0, nms_radius=4) if det == 'sp' else
                           ALIKED(model_name='aliked-n16', max_num_keypoints=m, detection_threshold=-1, nms_radius=2))
                    net.preprocess_conf['resize'] = None
                    self.models[det] = net.eval().to(self.dev)
                net = self.models[det]; net.conf.max_num_keypoints = m
                if det == 'aliked':
                    net.dkd.top_k = m
                p = net.extract(x[0]); sc = p['keypoint_scores'][0]
        return p['keypoints'][0].cpu().numpy(), sc.cpu().numpy()

    def cands(self, det, img, dep, n):
        m = n
        while True:
            kp, sc = self._run(det, img, m)
            o = np.argsort(-sc, kind='stable'); kp, sc = kp[o], sc[o]
            u = np.clip(kp[:, 0], 0, W - 1); v = np.clip(kp[:, 1], 0, H - 1)
            z = dep[np.clip(np.rint(v).astype(int), 0, H - 1), np.clip(np.rint(u).astype(int), 0, W - 1)]
            ok = np.flatnonzero((z > 0) & (z < DEPTH_MAX))
            if len(ok) >= n or len(kp) < m:
                ok = ok[:n]
                return u[ok].astype(np.float64), v[ok].astype(np.float64), sc[ok].astype(np.float64), z[ok].astype(np.float64)
            m = int(np.ceil(m * n / max(1, len(ok)) * 1.05))

    def desc(self, img, u, v):
        return self.sp_sample(np.stack([u, v], 1), self.sp_dense(img))

    def past(self, img):
        kp, sc = self.det2000(img, 'raco')
        u = np.clip(kp[:, 0], 0, W - 1).astype(np.float32); v = np.clip(kp[:, 1], 0, H - 1).astype(np.float32)
        return u, v, sc, None, self.desc(img, u, v)


class PastFrames:
    """the Cands stand-in pickers.track_count reads: frame(g)[4] = descriptors of frame g's strongest RaCo candidates"""
    def __init__(self, F, dets):
        self.F, self.dets, self.memo = F, dets, {}

    def frame(self, g):
        if g not in self.memo:
            self.memo[g] = self.dets.past(self.F.img(g))
        return self.memo[g]


class FrameDir:
    """a flight-layout folder (tools/frame_export.py output): image_left/, depth_left/, intrinsics.txt; no pose needed"""
    def __init__(self, d):
        self.dir = d; self.slug = os.path.basename(os.path.normpath(d))
        ip = os.path.join(d, 'intrinsics.txt')
        intr = np.loadtxt(ip) if os.path.isfile(ip) else np.array([320.0, 320.0, 320.0, 240.0])
        if not np.allclose(intr, [FX, FY, CX, CY], atol=1e-3):
            raise RuntimeError('%s has intrinsics %s, process uses %s: set KPBENCH_INTR' % (d, intr, (FX, FY, CX, CY)))

    def img(self, f):
        return np.array(Image.open(os.path.join(self.dir, 'image_left', '%06d_left.png' % f)).convert('RGB'))

    def depth(self, f):
        return np.load(os.path.join(self.dir, 'depth_left', '%06d_left_depth.npy' % f)).astype(np.float64)


def run_counting_fill(fn, *args, **kw):
    """run a picker and return (selection, number of picks it made itself before pickers.fill topped it up strongest
    first; None if it never called fill). pickers.fill is wrapped for the call, so the count is exact."""
    from . import pickers as pk
    orig, seen = pk.fill, []
    def fill(sel, n, K, pool=None):
        seen.append(len(sel)); return orig(sel, n, K, pool)
    pk.fill = fill
    try:
        idx = fn(*args, **kw)
    finally:
        pk.fill = orig
    return idx, (seen[-1] if seen else None)


def cand_list(u, v, z, img, valid):
    """embedded candidate list (rank = index in score order)"""
    Pc = to3d(u[valid], v[valid], z[valid])
    Q = plot_axes(Pc)
    return {'u': [round(float(x), 1) for x in u[valid]], 'v': [round(float(x), 1) for x in v[valid]],
            'x': r3(Q[:, 0]), 'y': r3(Q[:, 1]), 'z': r3(Q[:, 2]),
            'cx': r3(Pc[:, 0]), 'cy': r3(Pc[:, 1]), 'cz': r3(Pc[:, 2]),   # camera frame, for the 0.5 m cubes
            'rank': [int(j) for j in valid], 'd': [round(float(x), 2) for x in z[valid]],
            'col': ['#%02x%02x%02x' % tuple(int(c) for c in img[int(min(H - 1, max(0, round(b)))), int(min(W - 1, max(0, round(a))))]) for a, b in zip(u[valid], v[valid])]}


def build(fid, f, det='raco', pickers=('sd', 'cube0.5', 'far2d', 'far3d'), kmax=2000, step=4, ncand=0, fdir=None):
    """pickers: names, or '<det>:<name>' to run a picker on another detector's candidates (needs ncand, e.g. 'sp:sd').
    ncand > 0: fresh detector runs, the top ncand candidates with depth per detector (Detectors); fdir: a FrameDir folder
    instead of a flight id."""
    F = FrameDir(fdir) if fdir else Flight(fid)
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
    dv = dep[(dep > 0) & (dep < DEPTH_MAX)]
    zmed = round(float(np.median(dv)), 3) if len(dv) else None

    # 2. candidates per detector
    spec = [tuple(p.split(':', 1)) if ':' in p else (det, p) for p in pickers]
    dets = list(dict.fromkeys(d for d, _ in spec))
    assert ncand or dets == [det], 'pickers on other detectors need ncand'
    D = Detectors() if ncand else None
    raw = {}
    for d in dets:
        if ncand:     # fresh run for this frame, ncand candidates, all with depth
            u, v, s, z = D.cands(d, img, dep, ncand)
            raw[d] = (u, v, s, z, D.desc(img, u, v) if d == 'raco' else None)
        else:
            raw[d] = Cands(F.slug, d).frame(f)
    sets, poss, zcand = {}, {}, {}
    for d, (u, v, s, z, _) in raw.items():
        valid = np.flatnonzero((z > 0) & (z < DEPTH_MAX))
        poss[d] = {int(j): i for i, j in enumerate(valid)}     # candidate index -> position in the embedded list
        sets[d] = cand_list(u, v, z, img, valid)
        zcand[d] = round(float(np.median(z[valid])), 3) if len(valid) else None

    # 3. selections. sd, cube<s>, far2d/far3d, dpp* are prefix-consistent: the first K entries of the selection for
    # kmax are exactly the selection for K, so one ordered list per picker covers every K. grid is not, so its
    # selection is stored separately for every K (perK[name][K-1]). Indices point into the embedded candidate list;
    # candidates without depth (possible only through a picker's top-up fill) are dropped.
    sel, perK, dropped, own, set_of = {}, {}, {}, {}, {}
    past = None
    for key, (d, name) in zip(pickers, spec):
        u, v, s, z, de = raw[d]; pos = poss[d]
        fn = parse(name); kw = {}
        if getattr(fn, 'needs_past', False):     # dppr*: descriptors of the previous frames only (no pose, no future)
            assert ncand and d == 'raco', 'past-frame pickers need ncand and raco candidates'
            past = past or PastFrames(F, D)
            kw = {'ctx': (F, past, f), 'de': de}
        elif getattr(fn, 'needs_ctx', False) or getattr(fn, 'needs_desc', False):
            raise ValueError('picker %s needs data the viewer does not provide' % name)
        idx, made = run_counting_fill(fn, u, v, s, z, kmax, **kw)
        sel[key] = [pos[int(j)] for j in idx if int(j) in pos]
        dropped[key] = len(idx) - len(sel[key])
        set_of[key] = d
        if name.startswith('dpp'):    # greedy DPP stops when no gain is left; pickers.fill adds the rest strongest first
            own[key] = min(made, kmax) if made is not None else len(idx)
        if name in NOT_PREFIX:
            perK[key] = [[pos[int(j)] for j in fn(u, v, s, z, K, **kw) if int(j) in pos] for K in range(1, kmax + 1)]

    # 4. image as base64 JPEG
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=80)
    jpg = base64.b64encode(buf.getvalue()).decode('ascii')

    data = {'flight': fid or F.slug, 'frame': int(f), 'det': det, 'kmax': int(kmax), 'step': int(step),
            'W': W, 'H': H, 'fx': FX, 'fy': FY, 'pickers': list(pickers), 'cloud': cloud,
            'cands': sets[det] if len(sets) == 1 and not ncand else sets, 'setOf': set_of,
            'sel': sel, 'perK': perK, 'dropped': dropped, 'own': own, 'ncand': int(ncand), 'zmed': zmed, 'zcand': zcand}
    return data, jpg


def render(data, jpg):
    title = data.get('title') or '%s frame %d (%s)' % (data['flight'], data['frame'], data['det'])
    blob = json.dumps(data, separators=(',', ':'))
    return (HTML.replace('__H1__', data.get('h1') or title).replace('__TITLE__', title).replace('__PLOTLY__', PLOTLY)
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
  #controls fieldset { max-width: 100%; box-sizing: border-box; }
  #pickers { display: flex; flex-wrap: wrap; gap: 2px 10px; }
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
<h1>__H1__</h1>
<div id="source" style="color:#888;font-size:11px;margin:-4px 0 6px"></div>
<div id="controls">
  <fieldset><legend>picker</legend><span id="pickers"></span></fieldset>
  <span>K <input id="kslider" type="range" min="1" step="1"> <input id="kbox" type="number" min="1" step="1"></span>
  <label><input type="checkbox" id="cubes"> show 0.5 m cubes</label>
  <label><input type="checkbox" id="cloud" checked> show point cloud</label>
  <label id="allcl"><input type="checkbox" id="allc"> show all candidates</label>
  <label id="byrankl"><input type="checkbox" id="byrank"> colour selected by rank (default: pixel colour)</label>
  <span title="points farther forward than this are outside the 3D view (they stay in the statistics)">view depth <input id="dmax" type="range" min="2" step="0.5"> <span id="dmaxv"></span> m</span>
</div>
<div id="note" style="color:#555;margin-bottom:6px;max-width:1200px"></div>
<div id="pose" style="color:#333;margin-bottom:6px;max-width:1200px"></div>
<div id="readout"></div>
<div id="foot" style="color:#888;font-size:11px;margin:-3px 0 6px"></div>
<div id="panels">
  <div id="scene"></div>
  <div id="imgwrap"><canvas id="img" width="640" height="480"></canvas></div>
</div>
<script>
const D = __DATA__;
const JPG = "data:image/jpeg;base64,__JPG__";
const CUBE = 0.5, KMIN = D.kmin !== undefined ? D.kmin : 1;
// candidate sets: one per detector (D.setOf: picker -> detector); older pages carry a single set
const SETS = D.cands.u ? { _: D.cands } : D.cands;
const setOf = p => (D.setOf && SETS[D.setOf[p]]) ? D.setOf[p] : Object.keys(SETS)[0];
let C, NC, CSET;
function useSet(p) { CSET = setOf(p); C = SETS[CSET]; NC = C.u.length; }
const st = { picker: D.default || (D.pickers.includes('sd') ? 'sd' : D.pickers[0]), K: Math.min(D.kdefault || 100, D.kmax), dmax: 0 };
useSet(st.picker);
// 'order' pages (one fresh candidate set, pickers compared at the same k): faint unpicked candidates, one colour per picker
const ORDER = D.mode === 'order', LAB = D.labels || {}, HL = D.colours || {};
const lab = p => LAB[p] || p, hl = p => HL[p] || '#ffd400';
document.getElementById('note').textContent = D.note || '';
document.getElementById('source').textContent = D.source || '';
{ const pe = document.getElementById('pose'); pe.textContent = D.pose || ''; if (D.poseTitle) { pe.title = D.poseTitle; pe.style.cursor = 'help'; } if (!D.pose) pe.style.display = 'none'; }
if (ORDER) { document.getElementById('allc').checked = false;
  document.getElementById('allcl').lastChild.textContent = ' show unpicked';
  document.getElementById('byrankl').lastChild.textContent = ' colour by pick order'; }
// default view depth: the farthest point, so the whole scene is in view
st.dmax = Math.ceil(Math.max(...C.y, ...D.cloud.y));

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

function ordinal(n) {
  const t = n % 100, e = n % 10;
  return n + (t >= 11 && t <= 13 ? 'th' : e === 1 ? 'st' : e === 2 ? 'nd' : e === 3 ? 'rd' : 'th');
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
  const S = selection(), byr = document.getElementById('byrank').checked;
  if (document.getElementById('allc').checked) {
    if (ORDER) {
      const inS = new Uint8Array(NC); for (const i of S) inS[i] = 1;
      for (let i = 0; i < NC; i++) if (!inS[i]) {
        ctx.fillStyle = 'rgba(255,255,255,0.45)'; ctx.fillRect(C.u[i] - 1.5, C.v[i] - 1.5, 3, 3);
        ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(C.u[i] - 0.75, C.v[i] - 0.75, 1.5, 1.5);
      }
    } else {
      ctx.fillStyle = 'rgba(150,150,150,0.85)';
      for (let i = 0; i < NC; i++) ctx.fillRect(C.u[i] - 1, C.v[i] - 1, 2, 2);
    }
  }
  ctx.lineWidth = 2;
  if (ORDER) {
    S.forEach((i, n) => {
      ctx.beginPath(); ctx.arc(C.u[i], C.v[i], 3.2, 0, 2 * Math.PI);
      ctx.fillStyle = byr ? rankColour(n) : hl(st.picker); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.9)'; ctx.lineWidth = 1; ctx.stroke();
    });
    return;
  }
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
    text: S.map((i, n) => 'pick ' + (n + 1) + (ORDER ? '<br>' + ordinal(C.rank[i] + 1) + ' strongest' : '<br>rank ' + C.rank[i]) + '<br>depth ' + C.d[i] + ' m'),
    color: byr ? S.map((i, n) => rankColour(n)) : ORDER ? hl(st.picker) : S.map(i => C.col[i])
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
      marker: ORDER ? { size: 1.6, color: '#555', opacity: 0.35 } : { size: 2, color: '#888' }, hoverinfo: 'skip',
      visible: document.getElementById('allc').checked },
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
  if (ORDER) {
    const cell = 20, gx = Math.ceil(D.W / cell), gy = Math.ceil(D.H / cell), cells = new Set();
    let rs = 0;
    for (const i of S) { cells.add(Math.floor(C.u[i] / cell) * gy + Math.floor(C.v[i] / cell)); rs += C.rank[i] + 1; }
    const own = D.own[st.picker], n = S.length, sp = (t, x) => '<span title="' + t + '">' + x + '</span>';
    document.getElementById('readout').innerHTML = [
      sp('mean position of the picks in the detector strength order (1 = strongest of its 3000 points)',
         'avg strength rank ' + (n ? Math.round(rs / n) : '-')),
      sp('share of the 20 px image squares that hold at least one pick (' + cells.size + ' of ' + gx * gy + ')',
         'image coverage ' + Math.round(100 * cells.size / (gx * gy)) + ' %'),
      sp('number of distinct 0.5 m cubes in 3D that hold at least one pick', '0.5 m cubes ' + keys.size)
    ].join(' \u00b7 ');
    document.getElementById('foot').textContent = own !== undefined && own < D.kmax ?
      lab(st.picker) + ' stops choosing at ' + own + '; rest = strongest' : '';
    draw2d();
    return;
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
ks.min = kb.min = KMIN; ks.max = kb.max = D.kmax; ks.value = kb.value = st.K;
function setK(k) {
  k = Math.max(KMIN, Math.min(D.kmax, Math.round(+k || 0)));
  if (k === st.K && +ks.value === k && +kb.value === k) return;
  st.K = k; ks.value = k; kb.value = k; update();
}
ks.addEventListener('input', () => setK(ks.value));
kb.addEventListener('input', () => setK(kb.value));
const box = document.getElementById('pickers');
for (const p of D.pickers) {
  const l = document.createElement('label'), r = document.createElement('input');
  r.type = 'radio'; r.name = 'picker'; r.value = p; r.checked = p === st.picker;
  r.addEventListener('change', () => {
    st.picker = p; const old = CSET; useSet(p);
    if (CSET !== old) Plotly.restyle('scene', { x: [C.x], y: [C.y], z: [C.z] }, [T_ALL]);   // that detector's candidates
    update();
  });
  l.append(r, ' ' + lab(p) + ' '); box.append(l);
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
    ap.add_argument('--flight', default='')
    ap.add_argument('--dir', default='', help='a frame folder (tools/frame_export.py) instead of --flight')
    ap.add_argument('--kmin', type=int, default=1)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--det', default='raco')
    ap.add_argument('--pickers', default='sd,cube0.5,far2d,far3d,far3dw,far2dp400')
    ap.add_argument('--kmax', type=int, default=2000)
    ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--ncand', type=int, default=0,
                    help='>0: fresh detector runs, the top ncand candidates with depth per detector, pickers compared at the '
                         'same k; pickers may name a detector: raco:sd, sp:sd, aliked:sd')
    ap.add_argument('--default', default='', help='picker shown first')
    ap.add_argument('--title', default='')
    ap.add_argument('--note', default='', help='one plain sentence under the title')
    ap.add_argument('--h1', default='', help='page heading (default: the title)')
    ap.add_argument('--source', default='', help='small grey line under the heading')
    ap.add_argument('--labels', default='', help='button labels, key=label,key=label')
    ap.add_argument('--out', default='', help='file name in docs/frames (default <flight>_<frame>.html)')
    ap.add_argument('--outdir', default=OUT_DIR, help='folder for the page (default docs/frames)')
    ap.add_argument('--pose', default='', help='one extra line under the note (e.g. per-picker pose errors)')
    ap.add_argument('--pose-title', default='', help='hover text of the --pose line')
    ap.add_argument('--k-default', type=int, default=100, help='K shown first')
    a = ap.parse_args()
    pickers = [p.strip() for p in a.pickers.split(',') if p.strip()]
    assert a.flight or a.dir, 'need --flight or --dir'
    data, jpg = build(a.flight, a.frame, a.det, pickers, a.kmax, a.step, a.ncand, a.dir or None)
    data['kmin'] = a.kmin
    if a.ncand:
        multi = len(set(data['setOf'].values())) > 1
        lc = {p: plain_label(p, a.det, data['zcand'][data['setOf'][p]], multi) for p in pickers}
        data.update(mode='order', labels={p: v[0] for p, v in lc.items()}, colours={p: v[1] for p, v in lc.items()},
                    detLabels=DETNAMES if multi else {})
    if a.default:
        data['default'] = a.default
    if a.title:
        data['title'] = a.title
    if a.note:
        data['note'] = a.note
    data['kdefault'] = a.k_default
    if a.pose:
        data['pose'] = a.pose
    if a.pose_title:
        data['poseTitle'] = a.pose_title
    for k in ('h1', 'source'):
        if getattr(a, k):
            data[k] = getattr(a, k)
    if a.labels:
        data.setdefault('labels', {}).update(dict(x.split('=', 1) for x in a.labels.split(',')))
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, a.out or '%s_%d.html' % (data['flight'].replace('/', '_'), a.frame))
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(render(data, jpg))
    print('%s  %.2f MB  cloud %d  picks %s  own %s' % (out, os.path.getsize(out) / 1e6, len(data['cloud']['x']),
          {k: len(v) for k, v in data['sel'].items()}, data['own']))


if __name__ == '__main__':
    main()
