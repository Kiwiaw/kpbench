"""Stage 1: detector candidates -> cache (see cache.py). Needs torch, lightglue (SuperPoint) and the RaCo repo.

  python -m kpbench.extract --flights oldtown/P001 --dets raco,sp
  python -m kpbench.extract --import-pkl <dir>      # convert the earlier c23chk cache (<traj>_all.pkl) instead of re-running

RaCo: keypoints + ranker scores (strongest 2000). SuperPoint: keypoint scores (strongest 2000, nms 4, no threshold).
Both get SuperPoint's dense descriptor sampled at the point, so the matcher is the same for every detector.
Env: RACO_DIR (torch.hub local repo), DEV (cuda | cpu).
"""
import os, sys, time, pickle
import numpy as np
from .data import Flight, parse_flights, W, H, DEPTH_MAX
from . import cache

RACO_DIR = os.environ.get('RACO_DIR', r'C:\Users\Kinga\Desktop\ETH_lab\local_tools\RaCo')
DEV = os.environ.get('DEV', 'cuda')


def import_pkl(d, flights):
    """earlier format: {'raco'|'sp': [(u, v, z, desc fp16) per frame]} in score order; scores were not stored (rank only)"""
    for fid in flights:
        F = Flight(fid)
        p = os.path.join(d, F.traj + '_all.pkl')
        if not os.path.isfile(p):
            print('skip', fid, 'no', p); continue
        R = pickle.load(open(p, 'rb'))
        for det in ('raco', 'sp'):
            if os.path.isfile(cache.path(F.slug, det)):
                print('exists', F.slug, det); continue
            frames = []
            for f in range(F.N):
                u, v, z, de = R[det][f]
                s = -np.arange(len(u), dtype=np.float32)        # rank as score
                frames.append((np.asarray(u, np.float32), np.asarray(v, np.float32), s, np.asarray(z, np.float32), de))
            cache.save(F.slug, det, frames); print('imported', F.slug, det, F.N, flush=True)


def load_models(dets):
    import torch, torch.nn.functional as TF
    from lightglue import SuperPoint
    sp = SuperPoint(max_num_keypoints=None, detection_threshold=0.0, nms_radius=4).eval().to(DEV)
    sp.preprocess_conf['resize'] = None
    raco = None
    if 'raco' in dets or 'racoc' in dets:
        raco = torch.hub.load(RACO_DIR, 'raco', source='local', pretrained=True, max_num_keypoints=2000,
                              ranker=True, covariance_estimator=True).eval().to(DEV)
        if 'racoc' in dets:   # racoc = RaCo with a retrained ranker head (MegaDepth cube label, 7 Oct); weights from RACO_WEIGHTS
            w = os.environ['RACO_WEIGHTS']; sd = torch.load(w, map_location=DEV)
            sd = {k: v for k, v in sd.items() if k.startswith('ranker_head')}
            miss = raco.load_state_dict(sd, strict=False)
            assert not miss.unexpected_keys and len(sd) > 0, (miss.unexpected_keys, len(sd))
            print('racoc: loaded %d ranker-head tensors from %s' % (len(sd), w), flush=True)

    @torch.no_grad()
    def sp_dense(img):
        g = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
        x = torch.from_numpy(g.astype(np.float32))[None, None].to(DEV); m = sp
        x = m.relu(m.conv1a(x)); x = m.relu(m.conv1b(x)); x = m.pool(x)
        x = m.relu(m.conv2a(x)); x = m.relu(m.conv2b(x)); x = m.pool(x)
        x = m.relu(m.conv3a(x)); x = m.relu(m.conv3b(x)); x = m.pool(x)
        x = m.relu(m.conv4a(x)); x = m.relu(m.conv4b(x))
        return TF.normalize(m.convDb(m.relu(m.convDa(x))), p=2, dim=1)

    @torch.no_grad()
    def sp_sample(uv, dmap, s=8):
        kp = torch.from_numpy(uv.astype(np.float32))[None].to(DEV); b, c, h, w = dmap.shape
        kp = (kp - s / 2 + 0.5) / kp.new_tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)]) * 2 - 1
        d = TF.grid_sample(dmap, kp.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
        return TF.normalize(d.reshape(c, -1), p=2, dim=0).T.cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def det(img, d):
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).float().to(DEV)
        if d == 'sp':
            sp.conf.max_num_keypoints = 2000; f = sp.extract(x)
            kp = f['keypoints'][0].cpu().numpy(); sc = f['keypoint_scores'][0].cpu().numpy()
        else:
            try:
                p = raco({'image': x[None]})
            except Exception:
                p = raco(x[None])
            kp = p['keypoints'][0].cpu().numpy(); sc = p['ranker_scores'][0].cpu().numpy()
        o = np.argsort(-sc, kind='stable')[:cache.NCAND]
        return kp[o], sc[o]
    return sp_dense, sp_sample, det


def extract(flights, dets):
    todo = [(fid, d) for fid in flights for d in dets if not os.path.isfile(cache.path(Flight(fid).slug, d))]
    if not todo:
        print('all caches exist'); return
    sp_dense, sp_sample, det = load_models(set(d for _, d in todo))
    for fid in flights:
        F = Flight(fid); want = [d for d in dets if (fid, d) in todo]
        if not want:
            continue
        frames = {d: [] for d in want}; t0 = time.time()
        for f in range(F.N):
            img = F.img(f); dp = F.depth(f); dm = sp_dense(img)
            for d in want:
                kp, sc = det(img, d)
                u = np.clip(kp[:, 0], 0, W - 1).astype(np.float32); v = np.clip(kp[:, 1], 0, H - 1).astype(np.float32)
                z = dp[np.clip(np.rint(v).astype(int), 0, H - 1), np.clip(np.rint(u).astype(int), 0, W - 1)].astype(np.float32)
                frames[d].append((u, v, sc.astype(np.float32), z, sp_sample(np.stack([u, v], 1), dm)))
            if f % 100 == 0:
                print(F.id, f, F.N, '%.0fs' % (time.time() - t0), flush=True)
        for d in want:
            cache.save(F.slug, d, frames[d]); print(F.id, d, 'cached', '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--flights', default='all'); ap.add_argument('--dets', default='raco,sp'); ap.add_argument('--import-pkl', default='')
    a = ap.parse_args(); fl = parse_flights(a.flights)
    if a.import_pkl:
        import_pkl(a.import_pkl, fl)
    else:
        extract(fl, a.dets.split(','))
