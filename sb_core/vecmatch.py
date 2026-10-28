"""Exact vector matching of outlined text.

CAD exports draw every character as a set of thin stroke paths. The same label is therefore the *same set of
vertices* wherever it appears (shifted, and possibly turned by 90 degrees), independent of screen resolution or
anti-aliasing. Matching those vertices directly is exact, unlike matching rendered pixels.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from .detect import S, K_ANGLE

# upright frame -> orientation k (page frame, y down)  and the reverse
_TO_UPRIGHT = {0: lambda p: p, 1: lambda p: np.c_[-p[:, 1], p[:, 0]], 2: lambda p: -p, 3: lambda p: np.c_[p[:, 1], -p[:, 0]]}
_FROM_UPRIGHT = {0: lambda p: p, 1: lambda p: np.c_[p[:, 1], -p[:, 0]], 2: lambda p: -p, 3: lambda p: np.c_[-p[:, 1], p[:, 0]]}


def thin_vertices(page, bold_width, max_obj=40.0):
    """Vertices (x, top) of small, non-bold stroked paths = text glyph strokes."""
    pts = []
    for objs in (page.lines, page.curves):
        for o in objs:
            if not o.get("stroke") or o.get("linewidth", 0) >= bold_width - 0.01:
                continue
            if (o["x1"] - o["x0"]) > max_obj or (o["bottom"] - o["top"]) > max_obj:
                continue
            for p in (o.get("pts") or []):
                pts.append((p[0], p[1]))
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def template_vertices(page, bbox_pt, bold_width, pad=0.6):
    """Vertices of the thin strokes that lie inside the label's box (page frame)."""
    x0, y0, x1, y1 = bbox_pt
    pts = []
    for objs in (page.lines, page.curves):
        for o in objs:
            if not o.get("stroke") or o.get("linewidth", 0) >= bold_width - 0.01:
                continue
            if o["x0"] >= x0 - pad and o["x1"] <= x1 + pad and o["top"] >= y0 - pad and o["bottom"] <= y1 + pad:
                pts.extend((p[0], p[1]) for p in (o.get("pts") or []))
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def build_template(page, tpl, bold_width):
    """Template as vertices in the upright frame, or None when the label is not made of vector strokes."""
    pb = tpl.get("page_bbox_px")
    if pb is None:
        return None
    bbox_pt = tuple(v / S for v in pb)
    P = template_vertices(page, bbox_pt, bold_width)
    P = np.unique(np.round(P, 3), axis=0)
    if len(P) < 8:
        return None
    U = _TO_UPRIGHT[tpl["k"]](P)
    return U - U.min(axis=0)


def match(V, U, tol=0.30, max_missing=0.02, max_extra=0.10):
    """Find every copy of the upright vertex set U among page vertices V, in the 4 right-angle orientations.

    Returns dicts compatible with detect.match_orthogonal (score = fraction of template vertices found).

    True copies match every vertex (1.000). A label that differs in even one glyph misses several vertices and is rejected.
    """
    if len(V) == 0 or U is None:
        return []
    tree = cKDTree(V)
    out = []
    n = len(U)
    for k in range(4):
        P = _FROM_UPRIGHT[k](U)
        a = 0
        probes = np.unique(np.linspace(0, n - 1, 7).astype(int))
        T = V - P[a]                               # one translation hypothesis per page vertex
        for j in probes:
            if len(T) == 0:
                break
            d, _ = tree.query(P[j] + T, distance_upper_bound=tol)
            T = T[np.isfinite(d)]
        for t in T:
            d, _ = tree.query(P + t, distance_upper_bound=tol)
            ratio = float(np.isfinite(d).mean())
            if (1.0 - ratio) * n > max_missing * n:       # a different last glyph (SB3 vs SB1) leaves several vertices unmatched
                continue
            Q = P + t
            x0, y0 = Q.min(axis=0)
            x1, y1 = Q.max(axis=0)
            inside = int(((V[:, 0] >= x0 - 0.05) & (V[:, 0] <= x1 + 0.05) & (V[:, 1] >= y0 - 0.05) & (V[:, 1] <= y1 + 0.05)).sum())
            extra = max(0, inside - int(np.isfinite(d).sum()))
            if extra > max_extra * n:
                continue
            L = (x1 - x0) if k in (0, 2) else (y1 - y0)
            H = (y1 - y0) if k in (0, 2) else (x1 - x0)
            out.append(dict(k=k, rotated=False, angle=K_ANGLE[k], score=ratio, scale=1.0, method="vector",
                            bbox=(float(x0), float(y0), float(x1), float(y1)), cx=float((x0 + x1) / 2), cy=float((y0 + y1) / 2),
                            text_len=float(L), text_h=float(H)))
    out.sort(key=lambda c: -c["score"])
    kept = []
    for c in out:
        if all(math.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) > 1.0 for o in kept):
            kept.append(c)
    return kept


# --------------------------------------------------------------------------- labels at any angle (diagonal text)
def _rotm(phi):
    c, s_ = math.cos(phi), math.sin(phi)
    return np.array([[c, -s_], [s_, c]])


def match_rotated(V, U, tol=0.30, max_missing=0.02, max_extra=0.10, skip_orth_deg=0.6):
    """Find copies of the upright vertex set U at ANY rotation (page frame, y down), not only multiples of 90 degrees.

    Two far-apart template vertices A, B are paired with every pair of page vertices at the same distance; each pair
    fixes a rotation + shift, which is checked on a few more template vertices, refined by least squares and finally
    required to explain (almost) every template vertex. Exact copies therefore match at 1.000, as for right-angle labels.
    """
    if len(V) == 0 or U is None or len(U) < 8:
        return []
    n = len(U)
    D = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=2)
    a, b = np.unravel_index(D.argmax(), D.shape)
    A, B, Dl = U[a], U[b], float(D[a, b])
    tree = cKDTree(V)
    pairs = tree.query_pairs(Dl + tol, output_type="ndarray")
    if len(pairs) == 0:
        return []
    w = V[pairs[:, 1]] - V[pairs[:, 0]]
    pairs = pairs[np.abs(np.hypot(w[:, 0], w[:, 1]) - Dl) <= tol]
    if len(pairs) == 0:
        return []
    q1 = np.concatenate([V[pairs[:, 0]], V[pairs[:, 1]]])
    q2 = np.concatenate([V[pairs[:, 1]], V[pairs[:, 0]]])
    phi = np.arctan2(q2[:, 1] - q1[:, 1], q2[:, 0] - q1[:, 0]) - math.atan2(B[1] - A[1], B[0] - A[0])
    cs, sn = np.cos(phi), np.sin(phi)
    tx = q1[:, 0] - (cs * A[0] - sn * A[1])
    ty = q1[:, 1] - (sn * A[0] + cs * A[1])
    ph_deg = np.degrees(phi) % 90.0
    keep = np.minimum(ph_deg, 90.0 - ph_deg) > skip_orth_deg          # right angles are handled by match()
    cs, sn, tx, ty, phi = cs[keep], sn[keep], tx[keep], ty[keep], phi[keep]
    probes = [j for j in np.linspace(0, n - 1, 8).astype(int) if j not in (a, b)][:6]
    for j in probes:                                                    # progressive filtering on a few more vertices
        if len(phi) == 0:
            break
        X = cs * U[j, 0] - sn * U[j, 1] + tx
        Y = sn * U[j, 0] + cs * U[j, 1] + ty
        d, _ = tree.query(np.c_[X, Y], distance_upper_bound=0.55)
        ok = np.isfinite(d)
        cs, sn, tx, ty, phi = cs[ok], sn[ok], tx[ok], ty[ok], phi[ok]
    Wu, Hu = float(U[:, 0].max()), float(U[:, 1].max())
    out = []
    for c_, s_, x_, y_ in zip(cs, sn, tx, ty):
        R = np.array([[c_, -s_], [s_, c_]])
        t = np.array([x_, y_])
        for _ in range(3):                                              # least-squares refinement (Kabsch)
            Q = U @ R.T + t
            d, idx = tree.query(Q, distance_upper_bound=0.6)
            ok = np.isfinite(d)
            if ok.sum() < 0.8 * n:
                break
            P, M = U[ok], V[idx[ok]]
            pc, mc = P.mean(axis=0), M.mean(axis=0)
            Hm = (P - pc).T @ (M - mc)
            u_, _, vt = np.linalg.svd(Hm)
            R = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u_.T))]) @ u_.T
            t = mc - R @ pc
        Q = U @ R.T + t
        d, _ = tree.query(Q, distance_upper_bound=tol)
        matched = int(np.isfinite(d).sum())
        if (n - matched) > max_missing * n:
            continue
        ang = math.atan2(R[1, 0], R[0, 0])
        deg = math.degrees(ang) % 90.0
        if min(deg, 90.0 - deg) <= skip_orth_deg:
            continue
        ctr = R @ np.array([Wu / 2, Hu / 2]) + t
        near = tree.query_ball_point(ctr, r=max(Wu, Hu) + 3.0)
        Vu = (V[near] - t) @ R                                           # page -> upright frame
        inside = int(((Vu[:, 0] >= -0.05) & (Vu[:, 0] <= Wu + 0.05) & (Vu[:, 1] >= -0.05) & (Vu[:, 1] <= Hu + 0.05)).sum())
        if max(0, inside - matched) > max_extra * n:
            continue
        x0, y0 = Q.min(axis=0)
        x1, y1 = Q.max(axis=0)
        out.append(dict(k=None, rotated=True, angle=float((-math.degrees(ang)) % 360.0), score=matched / n, scale=1.0, method="vector-rotated",
                        bbox=(float(x0), float(y0), float(x1), float(y1)), cx=float(ctr[0]), cy=float(ctr[1]),
                        text_len=Wu, text_h=Hu, _R=R, _t=t))
    out.sort(key=lambda c: -c["score"])
    kept = []
    for c in out:
        if all(math.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) > 1.5 for o in kept):
            kept.append(c)
    return kept


def trailing_rotated(V, cand, n_chars):
    """True if glyph strokes follow a rotated label in its reading direction (a longer mark such as SB91 vs SB9)."""
    if len(V) == 0:
        return False
    R, t, Wu, Hu = cand["_R"], cand["_t"], cand["text_len"], cand["text_h"]
    tree = cKDTree(V)
    near = tree.query_ball_point([cand["cx"], cand["cy"]], r=2.5 * Wu)
    if not near:
        return False
    Vu = (V[near] - t) @ R
    cw = Wu / max(1, n_chars)
    sel = (Vu[:, 0] >= Wu + 0.1 * cw) & (Vu[:, 0] <= Wu + 1.5 * cw) & (Vu[:, 1] >= -0.25 * Hu) & (Vu[:, 1] <= 1.25 * Hu)
    if not sel.any():
        return False
    return float(Vu[sel, 1].max() - Vu[sel, 1].min()) >= 0.4 * Hu
