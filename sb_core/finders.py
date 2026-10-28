"""Extra ways to find the member line that belongs to a label.

The standard finder (``bold``) only sees *black* lines of the *thickest* width.  Real drawings often draw steel
members differently - for example as purple **dashed** lines (every dash is a tiny filled polygon), in another line
weight, or far from the label with a thin leader line in between.  Every label that the standard finder cannot place
is therefore tried, in order, with the fallback methods below.  The first method that finds a member wins and the
result is annotated with a **dotted line in that method's own colour**, so you can see at a glance how each member
was found (standard = solid blue).

    method   what it looks for                                              default look
    -------  ---------------------------------------------------------      -------------------
    bold     thickest black line parallel to the label (standard)           solid  blue
    colour   coloured lines, including dashed lines made of small polygons  dotted green
    width    dark lines of other widths (default 0.9 - 4 pt)                dotted orange
    leader   thin line that starts at the label and ends on a member        dotted brown
    wide     any of the above, but up to N x farther from the label         dotted cyan  (opt-in)
"""
import math
import re

import cv2
import numpy as np

from . import members as mem

# fallback order: most specific / most reliable first
EXTRA_ORDER = ("colour", "width", "leader", "wide")
DEFAULT_EXTRA = ("colour", "width", "leader")          # "wide" is opt-in: it is the least certain method
ALL_METHODS = ("bold",) + EXTRA_ORDER

TITLES = {
    "bold": "Bold black line (standard)",
    "colour": "Coloured / dashed line",
    "width": "Other line width",
    "leader": "Via leader line",
    "wide": "Wider search",
    "unmatched": "Label found, no member (not measured)",
}

DEFAULT_COLORS = {          # hex RGB
    "bold": "#0000FF",
    "colour": "#00A63F",
    "width": "#FF7F00",
    "leader": "#8B4A0B",
    "wide": "#00B8D9",
    "unmatched": "#E000E0",
}


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def style_for(method, opts):
    """Colour (0-1 RGB) and dash pattern name for a method. Standard = solid; everything else follows opts.dash_style."""
    colors = dict(DEFAULT_COLORS)
    colors.update({k: v for k, v in (getattr(opts, "method_colors", None) or {}).items() if v})
    try:
        rgb = hex_to_rgb(colors[method])
    except (ValueError, KeyError):
        rgb = hex_to_rgb(DEFAULT_COLORS[method])
    dash = "solid" if method == "bold" else getattr(opts, "dash_style", "dotted")
    return dict(color="#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb), rgb=rgb, dash=dash)


# --------------------------------------------------------------------------- colour helpers
def _rgb(c):
    """pdfplumber colour value (None / number / 1, 3 or 4 numbers) -> (r, g, b) or None."""
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    try:
        v = [float(x) for x in c]
    except (TypeError, ValueError):
        return None
    if len(v) == 1:
        return (v[0],) * 3
    if len(v) == 3:
        return tuple(v)
    if len(v) == 4:                       # CMYK
        cc, m, y, k = v
        return ((1 - cc) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def parse_colors(spec):
    """'auto' / '' -> None (auto-detect).  Otherwise colours separated by ; or | or spaces:  '#C78FFF;0.26,0.31,0.94;199,143,255'."""
    if spec is None:
        return None
    s = spec.strip().lower()
    if s in ("", "auto"):
        return None
    out = []
    for tok in re.split(r"[;|\s]+", s):
        if not tok:
            continue
        if tok.startswith("#"):
            out.append(hex_to_rgb(tok))
            continue
        try:
            v = [float(x) for x in tok.split(",")]
        except ValueError:
            raise ValueError(f"cannot read colour '{tok}' (use #RRGGBB or r,g,b)")
        if len(v) != 3:
            raise ValueError(f"colour '{tok}' needs three numbers (r,g,b)")
        if max(v) > 1.0:
            v = [x / 255.0 for x in v]
        out.append(tuple(v))
    return out or None


def parse_widths(spec):
    if not spec or not str(spec).strip():
        return []
    try:
        return [float(x) for x in re.split(r"[;,\s]+", str(spec).strip()) if x]
    except ValueError:
        raise ValueError(f"cannot read line widths '{spec}' (use numbers like 0.96,1.0)")


def _is_chromatic(rgb, min_sat=0.25):
    return rgb is not None and (max(rgb) - min(rgb)) >= min_sat


def _is_red(rgb):
    return rgb is not None and rgb[0] > 0.85 and rgb[1] < 0.2 and rgb[2] < 0.2      # revision clouds


def _is_dark(rgb, lim=0.35):
    return rgb is not None and max(rgb) <= lim


# --------------------------------------------------------------------------- segment extraction
def _pts(o, kind):
    pts = o.get("pts") or []
    if kind == "rect" and not pts:
        pts = [(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]), (o["x0"], o["bottom"]), (o["x0"], o["top"])]
    return [(float(p[0]), float(p[1])) for p in pts]


def _objects(page):
    for kind, objs in (("line", page.lines), ("curve", page.curves), ("rect", page.rects)):
        for o in objs:
            yield kind, o


def dark_width_segments(page, wmin, wmax, extra_widths=()):
    """Segments of dark (black / dark grey) strokes whose width is in [wmin, wmax] or equals one of extra_widths."""
    segs = []
    for kind, o in _objects(page):
        if not o.get("stroke"):
            continue
        lw = float(o.get("linewidth") or 0)
        ok = wmin <= lw <= wmax or any(abs(lw - w) <= 0.02 for w in extra_widths)
        if not ok or not _is_dark(_rgb(o.get("stroking_color"))):
            continue
        pts = _pts(o, kind)
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append(dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1]))
    return segs


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def colour_segments(page, colors=None, tol=0.06, max_thick=6.0, min_elong=2.0, max_len=400.0):
    """Centre-line segments of coloured member lines.

    * stroked open lines / polylines -> their own segments
    * filled or closed shapes (a dashed line is a row of tiny filled polygons, each one often split in 2-3 pieces) ->
      touching pieces of one colour are grouped, and every thin elongated group becomes one segment along its long axis.
    `colors` = list of RGB tuples, or None = every saturated colour except red (red = revision clouds).
    """
    def match(rgb):
        if rgb is None:
            return None
        if colors is None:
            return rgb if (_is_chromatic(rgb) and not _is_red(rgb)) else None
        for c in colors:
            if max(abs(a - b) for a, b in zip(rgb, c)) <= tol:
                return rgb
        return None

    segs, pieces = [], []                 # pieces: (colour key, points)
    for kind, o in _objects(page):
        pts = _pts(o, kind)
        if len(pts) < 2:
            continue
        stroke_rgb = match(_rgb(o.get("stroking_color"))) if o.get("stroke") else None
        fill_rgb = match(_rgb(o.get("non_stroking_color"))) if o.get("fill") else None
        if fill_rgb is None and stroke_rgb is None:
            continue
        closed = bool(o.get("fill")) or (len(pts) >= 3 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.05)
        if not closed:
            for a, b in zip(pts[:-1], pts[1:]):
                segs.append(dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1]))
            continue
        rgb = fill_rgb or stroke_rgb
        pieces.append((tuple(round(v, 2) for v in rgb), pts))

    # group touching pieces of the same colour
    n = len(pieces)
    if n:
        boxes = np.array([[min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)] for _, pts in pieces])
        uf = _UF(n)
        cell, pad = 24.0, 0.4
        grid = {}
        for i, b in enumerate(boxes):
            for gx in range(int((b[0] - pad) // cell), int((b[2] + pad) // cell) + 1):
                for gy in range(int((b[1] - pad) // cell), int((b[3] + pad) // cell) + 1):
                    grid.setdefault((gx, gy), []).append(i)
        for ids in grid.values():
            for ii, i in enumerate(ids):
                for j in ids[ii + 1:]:
                    if pieces[i][0] != pieces[j][0] or uf.find(i) == uf.find(j):
                        continue
                    a, b = boxes[i], boxes[j]
                    if a[0] <= b[2] + pad and b[0] <= a[2] + pad and a[1] <= b[3] + pad and b[1] <= a[3] + pad:
                        uf.union(i, j)
        groups = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)
        for ids in groups.values():
            P = np.array([p for i in ids for p in pieces[i][1]], dtype=np.float32)
            if len(P) < 2:
                continue
            (cx, cy), (w, h), _ = cv2.minAreaRect(P)
            long_, short = max(w, h), min(w, h)
            if long_ < 2.0 or long_ > max_len or short > max_thick or long_ < min_elong * max(short, 0.3):
                continue
            box = cv2.boxPoints(((cx, cy), (w, h), _))
            e0, e1 = box[1] - box[0], box[2] - box[1]
            u = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
            u = u / (np.hypot(*u) or 1.0)
            segs.append(dict(x1=float(cx - u[0] * long_ / 2), y1=float(cy - u[1] * long_ / 2),
                             x2=float(cx + u[0] * long_ / 2), y2=float(cy + u[1] * long_ / 2)))
    return segs


def thin_dark_segments(page, bold_width, min_len=6.0):
    """Thin black strokes long enough to be a leader line (not a letter stroke): list of ((x1,y1),(x2,y2))."""
    out = []
    for kind, o in _objects(page):
        if not o.get("stroke") or float(o.get("linewidth") or 0) >= bold_width - 0.01:
            continue
        if not _is_dark(_rgb(o.get("stroking_color"))):
            continue
        pts = _pts(o, kind)
        for a, b in zip(pts[:-1], pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
                out.append((a, b))
    return out


# --------------------------------------------------------------------------- the finder
def _pt_seg_dist(p, a, b):
    return mem._pt_seg(p[0], p[1], a, b)


class MemberFinder:
    """Per-page helper: builds the line runs of every method lazily and pairs a label with a member."""

    def __init__(self, page, bold_width, opts):
        self.page, self.bw, self.o = page, bold_width, opts
        self._runs, self._thin = {}, None
        self.colors = parse_colors(opts.member_colors)
        self.extra_widths = parse_widths(opts.extra_widths)

    # -- run sets
    def runs(self, name):
        if name in self._runs:
            return self._runs[name]
        o, L = self.o, self.o.min_member_len
        if name == "bold":
            r = mem.merge_runs(mem.bold_segments(self.page, self.bw))
        elif name == "colour":
            r = mem.merge_runs(colour_segments(self.page, self.colors, o.color_tol), gap=o.dash_gap, ang_tol=1.0, rho_tol=1.2)
        elif name == "width":
            r = mem.merge_runs(dark_width_segments(self.page, o.width_min, o.width_max, self.extra_widths))
        elif name == "wide":
            r = self.runs("bold") + self.runs("colour") + self.runs("width")
            self._runs[name] = r
            return r
        else:
            r = []
        r = [x for x in r if x["len"] >= L]
        self._runs[name] = r
        return r

    def thin(self):
        if self._thin is None:
            self._thin = thin_dark_segments(self.page, self.bw)
        return self._thin

    # -- association
    def _leader(self, cand):
        o = self.o
        x0, y0, x1, y1 = cand["bbox"]
        pad = 2.0
        h = max(cand.get("text_h", 5.0), 3.0)
        minlen = max(8.0, 1.8 * h)
        segs = self.thin()
        allruns = self.runs("wide")
        if not allruns:
            return None
        best = None
        for si, (a, b) in enumerate(segs):
            for tip, root in ((b, a), (a, b)):
                # root end must touch the label box
                if not (x0 - pad <= root[0] <= x1 + pad and y0 - pad <= root[1] <= y1 + pad):
                    continue
                if x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1:
                    continue
                path, used, cur = [root, tip], {si}, tip
                for _ in range(4):                  # follow a bent leader (up to 4 more pieces)
                    nxt = None
                    for sj, (c, d) in enumerate(segs):
                        if sj in used:
                            continue
                        for p, q in ((c, d), (d, c)):
                            if math.hypot(p[0] - cur[0], p[1] - cur[1]) <= 0.8:
                                nxt = (sj, q)
                                break
                        if nxt:
                            break
                    if not nxt:
                        break
                    used.add(nxt[0]); cur = nxt[1]; path.append(cur)
                for r in allruns:
                    d = _pt_seg_dist(cur, r["p0"], r["p1"])
                    if d <= o.leader_tip_dist and (best is None or d < best[0]):
                        best = (d, r, path)
        if best is None:
            return None
        d, r, path = best
        return dict(run=r, dist=d, next_dist=None, parallel=False, leader=[[round(x, 2), round(y, 2)] for x, y in path])

    def find(self, cand, methods):
        """Try 'bold' then the enabled fallbacks. Returns dict(run, dist, next_dist, parallel, method) or None."""
        o = self.o
        for name in methods:
            if name == "leader":
                a = self._leader(cand)
            elif name == "wide":
                a = mem.associate(cand, self.runs("wide"), o.max_dist * o.wide_factor, parallel_tol=20.0, penalty=o.max_dist,
                                  max_dist_nonparallel=o.max_dist * 0.6, along_margin=15.0)
            else:
                a = mem.associate(cand, self.runs(name), o.max_dist)
            if a is not None:
                a["method"] = name
                return a
        return None


def same_member(r1, r2, spacing=4.0, min_overlap=0.7, ang_tol=1.0):
    """True when two runs are the two edges (or a double drawing) of one member: parallel, closer than `spacing`, and overlapping along their length."""
    d = abs(r1["theta"] - r2["theta"]) % 180.0
    if min(d, 180.0 - d) > ang_tol:
        return False
    c, sn = math.cos(math.radians(r1["theta"])), math.sin(math.radians(r1["theta"]))
    rho2 = (-r2["p0"][0] * sn + r2["p0"][1] * c)
    rho1 = (-r1["p0"][0] * sn + r1["p0"][1] * c)
    if abs(rho2 - rho1) > spacing:
        return False
    a0, a1 = sorted((r1["p0"][0] * c + r1["p0"][1] * sn, r1["p1"][0] * c + r1["p1"][1] * sn))
    b0, b1 = sorted((r2["p0"][0] * c + r2["p0"][1] * sn, r2["p1"][0] * c + r2["p1"][1] * sn))
    ov = min(a1, b1) - max(a0, b0)
    return ov >= min_overlap * min(a1 - a0, b1 - b0)
