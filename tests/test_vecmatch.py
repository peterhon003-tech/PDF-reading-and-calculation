"""Exact vector matcher test on a synthetic page: python tests/test_vecmatch.py  (needs reportlab + pdfplumber)"""
import math
import os
import sys
import tempfile

import numpy as np
import pdfplumber
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sb_core import vecmatch  # noqa: E402
from sb_core.detect import S  # noqa: E402

# a made-up "word": three glyph polylines (y down, local coordinates, ~20 x 7 units)
GLYPHS = [[(0, 1), (2, 0), (4, 1), (1, 4), (3, 5), (4, 7), (2, 7), (0, 6)],
          [(6, 0), (6, 7), (9, 7), (10, 5), (9, 3.5), (6, 3.5)],
          [(14, 0), (12, 4.5), (17, 4.5), (15.5, 0), (15.5, 7)]]


def place(c, ox, oy, k, glyphs=GLYPHS):
    """draw the word at (ox, oy) rotated by k*90 degrees (page coordinates, y down)"""
    for g in glyphs:
        pts = np.array(g, float)
        if k == 1:
            pts = np.c_[pts[:, 1], -pts[:, 0]]
        elif k == 2:
            pts = -pts
        elif k == 3:
            pts = np.c_[-pts[:, 1], pts[:, 0]]
        pts = pts + (ox, oy)
        c.setLineWidth(0.72)
        p = c.beginPath()
        p.moveTo(pts[0][0], 400 - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, 400 - y)
        c.drawPath(p, stroke=1, fill=0)


def test_finds_all_orientations_and_ignores_decoys():
    with tempfile.TemporaryDirectory() as td:
        pdf = os.path.join(td, "t.pdf")
        c = canvas.Canvas(pdf, pagesize=(600, 400))
        place(c, 50, 50, 0)       # the sample
        place(c, 200, 120, 0)
        place(c, 300, 300, 1)     # reads bottom-to-top
        place(c, 450, 200, 2)     # upside down
        place(c, 100, 250, 3)     # reads top-to-bottom
        bad = [list(g) for g in GLYPHS]
        bad[1] = [(x + 1.2, y) for x, y in bad[1]]           # decoy: one glyph shifted by 1.2 units
        place(c, 500, 50, 0, bad)
        c.setLineWidth(1.44)
        c.line(10, 20, 590, 20)   # bold line must be ignored
        c.save()
        with pdfplumber.open(pdf) as doc:
            page = doc.pages[0]
            V = vecmatch.thin_vertices(page, 1.44)
            tpl = dict(page_bbox_px=tuple(v * S for v in (49, 49, 68, 58)), k=0)
            U = vecmatch.build_template(page, tpl, 1.44)
            assert U is not None
            found = vecmatch.match(V, U)
        pos = sorted((round(f["cx"]), round(f["cy"]), f["k"]) for f in found)
        assert len(found) == 5, pos                              # sample + 4 copies, decoy rejected
        assert sorted(f["k"] for f in found) == [0, 0, 1, 2, 3], pos
        assert min(f["score"] for f in found) > 0.95


if __name__ == "__main__":
    test_finds_all_orientations_and_ignores_decoys()
    print("ok test_finds_all_orientations_and_ignores_decoys")
