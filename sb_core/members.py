"""Bold-line (steel member) extraction and label -> member association. Coordinates: PDF units, origin top-left (y down)."""
import math
from collections import Counter


def _is_black(c):
    if c is None:
        return False
    if isinstance(c, (int, float)):
        return c == 0
    try:
        return all(abs(v) < 0.01 for v in c)
    except TypeError:
        return False


def _stroked(page):
    for kind, objs in (("line", page.lines), ("curve", page.curves), ("rect", page.rects)):
        for o in objs:
            if o.get("stroke") and _is_black(o.get("stroking_color")):
                yield kind, o


def detect_bold_width(page, min_total_length=200.0, max_width=4.0):
    """Thickest black stroke width whose strokes add up to at least `min_total_length` PDF units.

    Using total length (not segment count) keeps tiny text strokes from outvoting a few long member lines.
    Widths above `max_width` (title-block frames, 12-24 pt border strips ...) are ignored; if nothing is left the
    old behaviour (any width) is used.
    """
    total = Counter()
    for kind, o in _stroked(page):
        pts = o.get("pts") or []
        if kind == "rect" and not pts:
            pts = [(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]), (o["x0"], o["bottom"]), (o["x0"], o["top"])]
        L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))
        total[round(o["linewidth"], 2)] += L
    cands = [w for w, L in total.items() if L >= min_total_length and w > 0]
    sane = [w for w in cands if w <= max_width]
    cands = sane or cands
    return max(cands) if cands else (max(total) if total else 1.0)


def bold_segments(page, width):
    segs = []
    for kind, o in _stroked(page):
        if abs(o["linewidth"] - width) > 0.01:
            continue
        pts = o.get("pts") or []
        if kind == "rect" and not pts:
            pts = [(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]), (o["x0"], o["bottom"]), (o["x0"], o["top"])]
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append(dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1]))
    return segs


def merge_runs(segs, min_seg=1.0, ang_tol=0.4, rho_tol=0.7, gap=4.0):
    """Merge collinear, touching segments (any angle) into runs."""
    items = []
    for s in segs:
        dx, dy = s["x2"] - s["x1"], s["y2"] - s["y1"]
        L = math.hypot(dx, dy)
        if L < min_seg:
            continue
        items.append((math.degrees(math.atan2(dy, dx)) % 180.0, s))
    items.sort(key=lambda t: t[0])
    clusters = []
    for th, s in items:
        if clusters and abs(th - clusters[-1][0]) < ang_tol:
            clusters[-1][1].append((th, s))
        elif clusters and th > 180 - ang_tol and clusters[0][0] < ang_tol:
            clusters[0][1].append((th, s))
        else:
            clusters.append([th, [(th, s)]])
    runs = []
    for th0, members in clusters:
        th = sum(m[0] if m[0] - th0 < 90 else m[0] - 180 for m in members) / len(members)
        c, sn = math.cos(math.radians(th)), math.sin(math.radians(th))
        rows = []
        for _, s in members:
            r1 = -s["x1"] * sn + s["y1"] * c
            r2 = -s["x2"] * sn + s["y2"] * c
            t1 = s["x1"] * c + s["y1"] * sn
            t2 = s["x2"] * c + s["y2"] * sn
            rows.append(((r1 + r2) / 2, min(t1, t2), max(t1, t2)))
        rows.sort()
        groups = []
        for r in rows:
            if groups and abs(r[0] - groups[-1][0][0]) < rho_tol:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            g.sort(key=lambda r: r[1])
            cur_a, cur_b, n, rr = g[0][1], g[0][2], 1, [g[0][0]]
            for r, a, b in g[1:]:
                if a <= cur_b + gap:
                    cur_b = max(cur_b, b); n += 1; rr.append(r)
                else:
                    runs.append((th, sum(rr) / len(rr), cur_a, cur_b, n)); cur_a, cur_b, n, rr = a, b, 1, [r]
            runs.append((th, sum(rr) / len(rr), cur_a, cur_b, n))
    out = []
    for th, rho, t0, t1, n in runs:
        c, sn = math.cos(math.radians(th)), math.sin(math.radians(th))
        out.append(dict(theta=th, rho=rho, t0=t0, t1=t1, n=n, len=t1 - t0,
                        p0=(t0 * c - rho * sn, t0 * sn + rho * c), p1=(t1 * c - rho * sn, t1 * sn + rho * c)))
    return out


def _pt_seg(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def bbox_dist(bb, a, b):
    x0, y0, x1, y1 = bb
    return min(_pt_seg(x0 + (x1 - x0) * i / 6, y0 + (y1 - y0) * j / 6, a, b) for i in range(7) for j in range(7))


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def associate(cand, runs, max_dist=8.0, parallel_tol=12.0, penalty=8.0, max_dist_nonparallel=3.0, along_margin=5.0):
    """Pick the member (bold run) a label belongs to.

    Runs parallel to the label text (and under/next to it) are accepted up to `max_dist`; a non-parallel run
    (e.g. a diagonal member with horizontal text) is only accepted when the label practically touches it.
    """
    x0, y0, x1, y1 = cand["bbox"]
    lab_dir = (-cand["angle"]) % 180.0   # label angle is counter-clockwise (y up); run angles are clockwise (y down)
    scored = []
    for r in runs:
        (ax, ay), (bx, by) = r["p0"], r["p1"]
        pad = max_dist + 2
        if max(ax, bx) < x0 - pad or min(ax, bx) > x1 + pad or max(ay, by) < y0 - pad or min(ay, by) > y1 + pad:
            continue
        d = bbox_dist(cand["bbox"], r["p0"], r["p1"])
        if d > max_dist:
            continue
        par = _angle_diff(r["theta"], lab_dir) <= parallel_tol
        if not par and d > max_dist_nonparallel:
            continue
        if par:
            c, sn = math.cos(math.radians(r["theta"])), math.sin(math.radians(r["theta"]))
            t = cand["cx"] * c + cand["cy"] * sn
            if not (r["t0"] - along_margin <= t <= r["t1"] + along_margin):
                continue
        scored.append((d + (0.0 if par else penalty), d, par, r))
    scored.sort(key=lambda s: s[0])
    if not scored:
        return None
    best = scored[0]
    nxt = next((s[1] for s in scored[1:] if s[3] is not best[3]), None)
    return dict(run=best[3], dist=best[1], parallel=best[2], next_dist=nxt)
