"""Self-checks of the extra member finders (no OCR needed):  python tests/test_finders.py"""
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import pdfplumber
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sb_core import annotate, finders, members as mem  # noqa: E402
from sb_core.pipeline import Options  # noqa: E402

PURPLE = (0.78, 0.56, 1.0)


def make_page(draw, w=600, h=400):
    path = os.path.join(tempfile.mkdtemp(), "t.pdf")
    c = canvas.Canvas(path, pagesize=(w, h))
    draw(c)
    c.save()
    return pdfplumber.open(path)


def dashed_purple(c, x0, x1, y, dash=8, gap=3, thick=2.0):
    """a dashed member line: every dash is a small filled polygon, split in two triangles like the CAD export does"""
    c.setFillColorRGB(*PURPLE)
    c.setStrokeColorRGB(*PURPLE)
    c.setLineWidth(0)
    x = x0
    while x + dash <= x1 + 1e-6:
        for tri in ([(x, y), (x + dash, y), (x + dash, y + thick)], [(x, y), (x + dash, y + thick), (x, y + thick)]):
            p = c.beginPath()
            p.moveTo(*tri[0]); p.lineTo(*tri[1]); p.lineTo(*tri[2]); p.close()
            c.drawPath(p, stroke=1, fill=1)
        x += dash + gap


def test_dashed_coloured_member_is_one_run_of_full_length():
    pdf = make_page(lambda c: dashed_purple(c, 50, 391, 200))          # 31 dashes of 8 + 30 gaps of 3 = 338 long (50 -> 388)
    page = pdf.pages[0]
    segs = finders.colour_segments(page, None)
    runs = [r for r in mem.merge_runs(segs, gap=6.0, ang_tol=1.0, rho_tol=1.2) if r["len"] > 30]
    assert len(runs) == 1, [r["len"] for r in runs]
    assert abs(runs[0]["len"] - 338) < 1.5, runs[0]["len"]
    assert min(abs(runs[0]["theta"]), abs(runs[0]["theta"] - 180)) < 1.0


def test_colour_filter_and_red_clouds_ignored():
    def draw(c):
        dashed_purple(c, 50, 200, 100)
        c.setStrokeColorRGB(1, 0, 0); c.setLineWidth(1); c.line(50, 300, 400, 300)      # red revision cloud line
    page = make_page(draw).pages[0]
    assert len(finders.colour_segments(page, None)) > 0
    near_red = [s for s in finders.colour_segments(page, None) if abs(s["y1"] - 100) < 5]        # the red line is at y(top)=100
    assert not near_red, near_red
    assert finders.colour_segments(page, [(0.0, 0.5, 0.0)]) == []                    # a colour that is not on the page


def test_other_widths_found_when_standard_width_is_elsewhere():
    def draw(c):
        c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(1.0); c.line(50, 100, 300, 100)   # the real member, 1.0 pt
        c.setLineWidth(2.5); c.line(50, 300, 300, 300)                                 # something heavier elsewhere
    page = make_page(draw).pages[0]
    segs = finders.dark_width_segments(page, 0.9, 1.5)
    assert len(segs) == 1 and abs(segs[0]["x2"] - segs[0]["x1"]) == 250, segs
    assert len(finders.dark_width_segments(page, 0.1, 0.5, extra_widths=[2.5])) == 1


def test_bold_width_ignores_huge_strokes():
    def draw(c):
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(1.44); c.line(10, 50, 400, 50)
        c.setLineWidth(24); c.line(10, 300, 400, 300)
    page = make_page(draw).pages[0]
    assert abs(mem.detect_bold_width(page) - 1.44) < 0.01
    assert abs(mem.detect_bold_width(page, max_width=100) - 24) < 0.01            # old behaviour still reachable


def test_leader_line_reaches_member():
    def draw(c):
        c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(1.44); c.line(100, 200, 400, 200)    # member (y top = 200)
        c.setLineWidth(0.36); c.line(300, 330, 330, 330); c.line(330, 330, 330, 203)        # leader: label at (300..330, 330) -> down to the member
    page = make_page(draw).pages[0]
    o = Options(member="SB1")
    f = finders.MemberFinder(page, 1.44, o)
    cand = dict(bbox=(290, 62, 322, 69), cx=306, cy=65.5, angle=0.0, text_h=7.0, text_len=32.0, rotated=False, k=0)
    a = f.find(cand, ["leader"])
    assert a and a["method"] == "leader" and abs(a["run"]["len"] - 300) < 1, a
    assert f.find(cand, ["bold"]) is None                                                # too far for the standard finder


def test_wide_search_is_farther_than_standard():
    def draw(c):
        c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(1.44); c.line(100, 200, 400, 200)
    page = make_page(draw).pages[0]
    o = Options(member="SB1")
    f = finders.MemberFinder(page, 1.44, o)
    cand = dict(bbox=(240, 180, 272, 187), cx=256, cy=183.5, angle=0.0, text_h=7.0, text_len=32.0, rotated=False, k=0)   # 13 units above the member (standard limit 8, wide limit 20)
    assert f.find(cand, ["bold"]) is None
    a = f.find(cand, ["bold", "wide"])
    assert a and a["method"] == "wide", a


def test_two_edges_of_one_member_count_once():
    r = lambda y, x0=0.0, x1=300.0: dict(theta=0.0, p0=(x0, y), p1=(x1, y), len=x1 - x0)
    assert finders.same_member(r(100.0), r(102.5), 4.0)                       # both edges of a 2.5-unit-wide member
    assert not finders.same_member(r(100.0), r(110.0), 4.0)                    # a different member 10 units away
    assert not finders.same_member(r(100.0), r(101.0, 250, 550), 4.0)          # collinear-ish but barely overlapping
    assert not finders.same_member(r(100.0), dict(theta=45.0, p0=(0, 100), p1=(100, 200), len=141), 4.0)


def test_parse_colors_and_widths():
    assert finders.parse_colors("auto") is None and finders.parse_colors("") is None
    c = finders.parse_colors("#C78FFF; 0.26,0.31,0.94 | 199,143,255")
    assert len(c) == 3 and abs(c[0][0] - 0.78) < 0.01 and abs(c[2][2] - 1.0) < 0.01
    assert finders.parse_widths("0.5, 6") == [0.5, 6.0]
    for bad in ("zzz", "1,2"):
        try:
            finders.parse_colors(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_pdf_lines_use_method_colour_and_dots():
    src = os.path.join(tempfile.mkdtemp(), "src.pdf")
    c = canvas.Canvas(src, pagesize=(300, 200)); c.drawString(5, 190, "blank"); c.showPage(); c.save()
    dst = src.replace("src.pdf", "dst.pdf")
    mk = lambda meth, y: dict(id=f"X-{meth}", annotation=dict(
        style=dict(method=meth, color=finders.style_for(meth, Options(member="X"))["color"], dash=finders.style_for(meth, Options(member="X"))["dash"]),
        start=dict(x_pdf=20, y_pdf=y), end=dict(x_pdf=220, y_pdf=y), id_text=dict(base_x_pdf=100, base_y_pdf=y + 6, rotation_deg=0)))
    ms = [mk("bold", 50), mk("colour", 100), mk("width", 150)]
    annotate.write_annotated_pdf(src, dst, {1: ms}, {1: (300, 200)}, {1: [(10, 10, 40, 20)]})
    page = pdfplumber.open(dst).pages[0]
    lines = {round(l["pts"][0][1]): l for l in page.lines if l.get("stroke") and l["linewidth"] > 1}
    got = {}
    for l in page.lines:
        if l.get("stroke") and abs(l["linewidth"] - 1.44) < 0.01 and abs(l["pts"][-1][0] - l["pts"][0][0]) > 100:
            col = tuple(round(v, 2) for v in l["stroking_color"])
            got[col] = l.get("dash")
    assert (0.0, 0.0, 1.0) in got and not (got[(0.0, 0.0, 1.0)] and got[(0.0, 0.0, 1.0)][0]), got       # standard: solid blue
    green = tuple(round(v, 2) for v in finders.hex_to_rgb("#00A63F"))
    assert green in got and got[green] and got[green][0], got                                           # colour method: dotted green
    orange = tuple(round(v, 2) for v in finders.hex_to_rgb("#FF7F00"))
    assert orange in got and got[orange] and got[orange][0], got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
