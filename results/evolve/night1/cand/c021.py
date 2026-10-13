# idea: inside the parent's greedy-DPP, scale each point's quality q_i by a soft "off dominant-plane" boost from a fast PCA/SVD plane fit, so the determinant-maximizing diversity search itself prefers physically off-plane, well-conditioned points for PnP while keeping DPP's proven high frame-to-frame repeatability (rep5 0.359) intact.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

SIGMA, ALPHA, BETA = 0.5, 1.0, 1.5


def pick(u, v, s, z, K):
    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(min(K, len(u)))
    P = to3d(u[ok], v[ok], z[ok])
    n = len(ok)

    # --- fast robust dominant-plane estimate (PCA/SVD) ---
    if n >= 4:
        c = np.median(P, axis=0)
        Pc = P - c
        try:
            _, _, vt = np.linalg.svd(Pc, full_matrices=False)
            normal = vt[-1]
            dist = np.abs(Pc @ normal)
            dmax = dist.max()
            boost = 1.0 + BETA * (dist / dmax) if dmax > 1e-9 else np.ones(n)
        except np.linalg.LinAlgError:
            boost = np.ones(n)
    else:
        boost = np.ones(n)

    q = np.exp(-ALPHA * np.arange(n) / 100.0) * boost
    d2 = q ** 2
    c_rows = np.zeros((0, n))
    sel = []
    for _ in range(min(K, n)):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        dj = np.sqrt(d2[j])
        Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, 1) / (2 * SIGMA ** 2))
        e = (Lj - (c_rows[:, j] @ c_rows if len(c_rows) else 0.0)) / dj
        c_rows = np.vstack([c_rows, e])
        d2 = d2 - e ** 2
        d2[j] = -np.inf
        sel.append(j)

    sel = list(ok[np.array(sel, int)])
    if len(sel) < min(K, len(u)):
        used = set(sel)
        sel += [i for i in range(len(u)) if i not in used][:min(K, len(u)) - len(sel)]
    return np.array(sel, int)
