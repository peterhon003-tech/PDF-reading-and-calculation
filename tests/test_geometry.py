"""Fast self-checks of the geometry code (no OCR / rendering needed):  python tests/test_geometry.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sb_core import annotate, members as mem  # noqa: E402


def seg(x1, y1, x2, y2):
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


def test_merge_and_length():
    # a horizontal beam drawn in two touching pieces plus a separate far piece
    runs = mem.merge_runs([seg(0, 100, 60, 100), seg(62, 100, 120, 100), seg(300, 100, 340, 100)])
    lens = sorted(round(r["len"], 2) for r in runs)
    assert lens == [40.0, 120.0], lens


def test_annotation_same_length_right_and_below():
    h = 1000.0
    for start, end, side in [((100, 500), (100, 300), "right"), ((10, 50), (210, 50), "below")]:
        run = dict(p0=start, p1=end)
        s, e, th = annotate.canonical(run)
        ann = annotate.make_annotation(s, e, th, 4.0, h)
        L = math.hypot(e[0] - s[0], e[1] - s[1])
        La = math.hypot(ann["b"][0] - ann["a"][0], ann["b"][1] - ann["a"][1])
        assert abs(L - La) < 1e-9 and ann["side"] == side, (L, La, ann["side"])


def test_diagonal_offset_is_perpendicular():
    s, e, th = annotate.canonical(dict(p0=(0, 300), p1=(173.2, 200)))   # ~30 deg rising to the right
    ann = annotate.make_annotation(s, e, th, 4.0, 1000.0)
    dx, dy = ann["a"][0] - s[0], ann["a"][1] - (1000 - s[1])
    assert abs(math.hypot(dx, dy) - 4.0) < 1e-9
    ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
    assert abs(dx * ux + dy * uy) < 1e-9          # offset is perpendicular to the member


def test_association_prefers_parallel():
    runs = mem.merge_runs([seg(0, 100, 200, 100), seg(100, 98, 100, 60)])     # horizontal beam + vertical stub just below the label
    cand = dict(bbox=(90, 90, 110, 98), cx=100, cy=94, angle=0.0)
    a = mem.associate(cand, [r for r in runs if r["len"] >= 30])
    th = a["run"]["theta"]
    assert a and a["parallel"] and min(abs(th), abs(th - 180)) < 1, a


def test_diagonal_label_picks_its_own_arm_of_a_v():
    # a V: one arm rising to the right, one falling to the right, meeting at (100, 200) in page coordinates (y down)
    rising = seg(100, 200, 200, 130)      # up and to the right  -> 35 deg counter-clockwise
    falling = seg(100, 200, 200, 270)     # down and to the right -> 145 deg counter-clockwise
    runs = [r for r in mem.merge_runs([rising, falling]) if r["len"] >= 30]
    # label text rising at 35 deg (counter-clockwise), lying right next to the RISING arm
    cand = dict(bbox=(140, 160, 160, 175), cx=150, cy=167, angle=35.0)
    a = mem.associate(cand, runs)
    (x0, y0), (x1, y1) = a["run"]["p0"], a["run"]["p1"]
    assert a["parallel"], "label and member must count as parallel"
    assert (y1 - y0) / (x1 - x0) < 0, "must be the rising arm (y decreases as x grows)"


def test_id_text_stays_on_the_member_for_every_direction():
    import numpy as np
    page_h = 400.0
    img = np.full((int(page_h * annotate.S), int(500 * annotate.S)), 255, np.uint8)          # blank page
    for (x0, y0), (x1, y1) in [((50, 300), (150, 230)),      # rising to the right   (angle ~35)
                               ((50, 100), (150, 170)),      # falling to the right  (angle ~145)
                               ((50, 100), (50, 200)),       # vertical
                               ((50, 100), (200, 100))]:     # horizontal
        s_, e_, th = annotate.canonical(dict(p0=(x0, y0), p1=(x1, y1)))
        ann = annotate.make_annotation(s_, e_, th, 4.0, page_h)
        t = annotate.place_text(img, ann, "SB9-012", page_h)
        bx, by = t["base"]
        ax, ay, bx2, by2 = ann["a"][0], ann["a"][1], ann["b"][0], ann["b"][1]
        L2 = (bx2 - ax) ** 2 + (by2 - ay) ** 2
        tt = ((bx - ax) * (bx2 - ax) + (by - ay) * (by2 - ay)) / L2                            # position along the blue line
        assert -0.01 <= tt <= 1.01, f"ID text is {tt:.2f} of the way along the line (must be within 0..1) for angle {th:.0f}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
