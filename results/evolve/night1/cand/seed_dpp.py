# seed: greedy DPP MAP (dpp0.5): L = diag(q) K diag(q), K = Gaussian kernel on camera-frame 3D distance (sigma 0.5 m),
# quality q_i = exp(-alpha * rank_i / 100), alpha = 1; each pick maximises the log-det gain (Chen, Zhang, Zhou 2018)
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

SIGMA, ALPHA = 0.5, 1.0


def pick(u, v, s, z, K):
    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(min(K, len(u)))
    P = to3d(u[ok], v[ok], z[ok]); n = len(ok); q = np.exp(-ALPHA * np.arange(n) / 100.0)
    d2 = q ** 2; c = np.zeros((0, n)); sel = []
    for _ in range(min(K, n)):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        dj = np.sqrt(d2[j]); Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, 1) / (2 * SIGMA ** 2))
        e = (Lj - (c[:, j] @ c if len(c) else 0.0)) / dj; c = np.vstack([c, e]); d2 = d2 - e ** 2; d2[j] = -np.inf; sel.append(j)
    sel = list(ok[np.array(sel, int)])
    if len(sel) < min(K, len(u)):
        used = set(sel); sel += [i for i in range(len(u)) if i not in used][:min(K, len(u)) - len(sel)]
    return np.array(sel, int)
