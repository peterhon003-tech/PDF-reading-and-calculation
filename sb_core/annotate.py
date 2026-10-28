"""Blue annotation geometry, ID text placement, vector overlay and PNG rendering."""
import io
import math
import os
import subprocess

import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from .detect import S

FONT, FSIZE = "Helvetica-Bold", 8.0
BLUE = (0, 0, 1)
LINE_W = 1.44
unmatched_color = "#E000E0"       # dotted box around a label with no member


def canonical(run):
    """Start = left end (lower end for vertical). Returns start, end in y-down coordinates and the direction angle (deg CCW, y-up, [0,180))."""
    (x0, y0), (x1, y1) = run["p0"], run["p1"]
    if (x1 < x0 - 1e-6) or (abs(x1 - x0) < 1e-6 and y1 > y0):
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
    if abs(y1 - y0) < 0.01:
        y1 = y0
    if abs(x1 - x0) < 0.01:
        x1 = x0
    theta = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180.0
    if theta > 179.999:
        theta = 0.0
    return (x0, y0), (x1, y1), theta


def orientation(theta, tol=0.5):
    if min(theta, 180 - theta) < tol:
        return "horizontal"
    if abs(theta - 90) < tol:
        return "vertical"
    return "diagonal"


def make_annotation(start, end, theta, offset, page_h):
    """Offset the member to the right / below. Returns geometry in y-up (PDF native) coordinates."""
    n = (math.sin(math.radians(theta)), -math.cos(math.radians(theta)))  # outer normal, y-up
    sx, sy = start[0], page_h - start[1]
    ex, ey = end[0], page_h - end[1]
    a = (sx + n[0] * offset, sy + n[1] * offset)
    b = (ex + n[0] * offset, ey + n[1] * offset)
    side = "below" if abs(n[1]) > 0.999 else ("right" if abs(n[0]) > 0.999 else "lower-right/right-up")
    return dict(a=a, b=b, n=n, side=side, theta=theta, offset=offset)


def place_text(img, ann, text, page_h):
    """Choose where along the blue line the ID goes (outer side), avoiding existing drawing ink."""
    n = ann["n"]
    L = math.hypot(ann["b"][0] - ann["a"][0], ann["b"][1] - ann["a"][1])
    u = ((ann["b"][0] - ann["a"][0]) / L, (ann["b"][1] - ann["a"][1]) / L)   # real start -> end direction (the angle alone is ambiguous by 180 degrees)
    w = stringWidth(text, FONT, FSIZE)
    cap = FSIZE * 0.72
    gap = 3.0
    H, W = img.shape
    best = None
    fracs = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.12, 0.88]
    for f in fracs:
        along = f * L
        if L > w + 4 and (along < w / 2 + 1 or along > L - w / 2 - 1):
            continue
        p = (ann["a"][0] + u[0] * along, ann["a"][1] + u[1] * along)
        c = (p[0] + n[0] * (gap + cap / 2), p[1] + n[1] * (gap + cap / 2))
        xs = np.arange(-w / 2 - 1, w / 2 + 1.01, 1.0)
        ys = np.arange(-cap / 2 - 1, cap / 2 + 1.01, 1.0)
        gx, gy = np.meshgrid(xs, ys)
        px = c[0] + gx * u[0] + gy * n[0]
        py = c[1] + gx * u[1] + gy * n[1]
        ix = np.clip((px * S).astype(int), 0, W - 1)
        iy = np.clip(((page_h - py) * S).astype(int), 0, H - 1)
        dark = int((img[iy, ix] < 160).sum())
        if best is None or dark < best[0]:
            best = (dark, f, p)
        if dark == 0:
            break
    if best is None:
        f = 0.5
        best = (0, f, (ann["a"][0] + u[0] * L / 2, ann["a"][1] + u[1] * L / 2))
    dark, f, p = best
    if ann["theta"] <= 90 + 1e-6:
        rot = ann["theta"]
        base = (p[0] + n[0] * (gap + cap), p[1] + n[1] * (gap + cap))
    else:
        rot = ann["theta"] - 180
        base = (p[0] + n[0] * gap, p[1] + n[1] * gap)
    return dict(base=base, rot=rot, fraction=f, overlap_pixels=dark)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def set_dash(c, dash, lw=LINE_W, gap=None):
    """'solid' | 'dotted' (round dots) | 'dashed'. `gap` = distance between dot centres / dash pitch."""
    if dash == "dotted":
        c.setLineCap(1)
        c.setDash([0.01, gap or max(3.2, lw * 2.4)])
    elif dash == "dashed":
        c.setLineCap(0)
        c.setDash([6, 3])
    else:
        c.setLineCap(0)
        c.setDash([])


def _draw_text(c, text, x, y, rot, rgb=BLUE):
    c.saveState()
    c.translate(x, y)
    c.rotate(rot)
    c.setFont(FONT, FSIZE)
    w = stringWidth(text, FONT, FSIZE)
    c.setFillColorRGB(1, 1, 1)  # white halo so the ID stays readable on top of drawing lines
    for dx, dy in [(-.9, 0), (.9, 0), (0, -.9), (0, .9), (-.7, -.7), (.7, .7), (-.7, .7), (.7, -.7)]:
        c.drawString(-w / 2 + dx, dy, text)
    c.setFillColorRGB(*rgb)
    c.drawString(-w / 2, 0, text)
    c.restoreState()


def write_annotated_pdf(src_pdf, dst_pdf, members_by_page, page_sizes, unmatched_by_page=None, legend=None, unmatched_hex=None):
    """Overlay every member's line + ID (colour / dash taken from member["annotation"]["style"]).

    unmatched_by_page: {page: [(x0, y0, x1, y1), ...]} labels found but not measured -> dotted box.
    legend: list of dict(method, title, color, dash, count) -> one extra last page explaining the colours.
    """
    unmatched_by_page = unmatched_by_page or {}
    reader = PdfReader(src_pdf)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages, 1):
        ms = members_by_page.get(i)
        um = unmatched_by_page.get(i)
        if ms or um:
            w, h = page_sizes[i]
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=(w, h))
            for m in ms or []:
                a = m["annotation"]
                st = a.get("style") or {}
                rgb = hex_rgb(st["color"]) if st.get("color") else BLUE
                c.saveState()
                c.setStrokeColorRGB(*rgb)
                c.setLineWidth(LINE_W)
                set_dash(c, st.get("dash", "solid"))
                c.line(a["start"]["x_pdf"], a["start"]["y_pdf"], a["end"]["x_pdf"], a["end"]["y_pdf"])
                if st.get("dash") == "dotted":          # a round dot exactly on the end point, so the dotted line is as long as the member
                    c.setDash([])
                    c.setFillColorRGB(*rgb)
                    c.circle(a["end"]["x_pdf"], a["end"]["y_pdf"], LINE_W / 2, stroke=0, fill=1)
                c.restoreState()
                t = a["id_text"]
                _draw_text(c, m["id"], t["base_x_pdf"], t["base_y_pdf"], t["rotation_deg"], rgb)
            for x0, y0, x1, y1 in um or []:
                c.saveState()
                c.setStrokeColorRGB(*hex_rgb(unmatched_hex or unmatched_color))
                c.setLineWidth(1.0)
                set_dash(c, "dotted", 1.0, 2.6)
                c.rect(x0 - 3, h - (y1 + 3), (x1 - x0) + 6, (y1 - y0) + 6, stroke=1, fill=0)
                c.restoreState()
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    if legend:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(595, 842))
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 790, "Annotation legend - how each member was found")
        y = 750
        for row in legend:
            rgb = hex_rgb(row["color"])
            c.saveState()
            c.setStrokeColorRGB(*rgb)
            c.setLineWidth(LINE_W)
            set_dash(c, row["dash"])
            c.line(40, y + 3, 140, y + 3)
            c.restoreState()
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(155, y, f"{row['title']}  -  {row['count']} item(s)")
            y -= 24
        c.setFont("Helvetica", 8.5)
        for line in ["Solid blue = standard finder (thickest black line). Dotted lines = a fallback finder was needed;",
                     "please check those members (and the dotted boxes = labels with no member found) against the drawing."]:
            c.drawString(40, y - 10, line)
            y -= 12
        c.save()
        buf.seek(0)
        writer.add_page(PdfReader(buf).pages[0])
    writer.add_metadata({"/Title": "Steel member annotation"})
    with open(dst_pdf, "wb") as f:
        writer.write(f)


def render_png(pdf_path, page_no, out_png, dpi=300, review_boxes=(), legend=None):
    """Render a page to PNG. `review_boxes` = (x0, y0, x1, y1, kind) in page units (y down) for labels found but not annotated.

    `legend` (rows of dict(title, color, dash, count)) adds a strip under the page explaining the line colours."""
    prefix = out_png[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(page_no), "-l", str(page_no), "-singlefile", pdf_path, prefix],
                   check=True, capture_output=True)
    if review_boxes or legend:
        from PIL import Image, ImageDraw, ImageFont
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(out_png).convert("RGB")
        dr = ImageDraw.Draw(im)
        f = dpi / 72.0
        for x0, y0, x1, y1, kind in review_boxes:
            col = (170, 170, 170) if kind == "longer" else (255, 140, 0)    # grey = longer mark skipped on purpose, orange = please review
            dr.rectangle([(x0 - 3) * f, (y0 - 3) * f, (x1 + 3) * f, (y1 + 3) * f], outline=col, width=max(2, int(dpi / 100)))
        if legend:
            fs = int(11 * f)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", fs)
            except OSError:
                font = ImageFont.load_default()
            band = int(len(legend) * fs * 1.9 + fs)
            out = Image.new("RGB", (im.width, im.height + band), "white")
            out.paste(im, (0, 0))
            d2 = ImageDraw.Draw(out)
            y = im.height + fs // 2
            for row in legend:
                rgb = tuple(int(row["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                x0, x1 = int(20 * f), int(110 * f)
                yy = y + fs // 2
                if row["dash"] == "dotted":
                    r = max(2, int(0.72 * f))
                    for x in range(x0, x1, int(3.4 * f)):
                        d2.ellipse([x - r, yy - r, x + r, yy + r], fill=rgb)
                elif row["dash"] == "dashed":
                    for x in range(x0, x1, int(9 * f)):
                        d2.line([x, yy, min(x + int(6 * f), x1), yy], fill=rgb, width=max(2, int(1.44 * f)))
                else:
                    d2.line([x0, yy, x1, yy], fill=rgb, width=max(2, int(1.44 * f)))
                d2.text((x1 + int(10 * f), y), f"{row['title']}  ({row['count']})", fill=(0, 0, 0), font=font)
                y += int(fs * 1.9)
            im = out
        im.save(out_png, optimize=True)
    return out_png
