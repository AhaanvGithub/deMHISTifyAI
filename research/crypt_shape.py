"""Shape of one crypt outline (boolean mask, ~2 um/px): centre line, bend, T-branch, width evenness, lumen count."""
import numpy as np


def centre_line(mask):
    """Longest path through the skeleton (pixel coords) and the skeleton graph's branch points."""
    from skimage.morphology import skeletonize
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    sk = skeletonize(mask)
    pts = np.argwhere(sk)
    if len(pts) < 3:
        return pts, []
    idx = -np.ones(mask.shape, int)
    idx[tuple(pts.T)] = np.arange(len(pts))
    r, c, w = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            q = pts + (dy, dx)
            ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < mask.shape[0]) & (q[:, 1] < mask.shape[1])
            j = np.full(len(pts), -1)
            j[ok] = idx[q[ok, 0], q[ok, 1]]
            m = j >= 0
            r += list(np.flatnonzero(m)); c += list(j[m]); w += [np.hypot(dy, dx)] * int(m.sum())
    g = coo_matrix((w, (r, c)), shape=(len(pts), len(pts))).tocsr()
    deg = np.diff(g.indptr)
    d0 = shortest_path(g, indices=[0], directed=False)[0]
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = shortest_path(g, indices=[a], directed=False, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da[0]), da[0], -1)))
    path = [b]
    while path[-1] != a and pred[0, path[-1]] >= 0:
        path.append(pred[0, path[-1]])
    return pts[path], pts[deg >= 3]


def describe(mask, lumen=None, um_per_px=2.0):
    from scipy import ndimage as ndi
    from skimage import measure
    dist = ndi.distance_transform_edt(mask)
    path, branch = centre_line(mask)
    out = {"length_um": 0.0, "width_um": float(2 * dist.max() * um_per_px), "bend_deg": 0.0, "width_cv": None,
           "t_arms": 0, "lumens": None}
    if len(path) >= 6:
        seg = np.hypot(*np.diff(path, axis=0).T)
        L = seg.sum()
        out["length_um"] = float(L * um_per_px)
        widths = 2 * dist[tuple(path.T)]
        core = widths[len(widths) // 6: len(widths) - len(widths) // 6]
        out["width_cv"] = float(core.std() / max(core.mean(), 1e-6)) if len(core) > 2 else None
        # bend: direction of the first vs last third of the centre line
        n = len(path)
        a1 = path[n // 3] - path[0]
        a2 = path[-1] - path[2 * n // 3]
        cos = np.dot(a1, a2) / max(np.linalg.norm(a1) * np.linalg.norm(a2), 1e-6)
        out["bend_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        # inverted T: a side arm leaving the centre line that is itself long
        from skimage.morphology import skeletonize, dilation, disk
        rest = skeletonize(mask)
        on = np.zeros_like(mask)
        on[tuple(path.T)] = True
        rest &= ~dilation(on, disk(2))
        if rest.any():
            arms = [p.area for p in measure.regionprops(measure.label(rest, connectivity=2))]
            out["t_arms"] = int(sum(a * um_per_px >= 1.2 * out["width_um"] for a in arms))
    if lumen is not None:
        lab = measure.label(lumen & mask)
        big = [p for p in measure.regionprops(lab) if p.area * um_per_px ** 2 >= 40]   # >= ~7 um across
        out["lumens"] = len(big)
    return out
