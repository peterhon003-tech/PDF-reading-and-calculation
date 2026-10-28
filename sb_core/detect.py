"""Label detection.

CAD-exported PDFs usually have *outlined* text (no text layer), so labels are found visually:
1. OCR (tesseract) locates one clean sample of the typed label -> used as a template.
2. Ink-based template matching finds every other copy of that label in 4 orientations
   (and optionally at other angles for diagonal text).
3. Each candidate is re-read with OCR to reject longer marks (e.g. SB4A when searching SB4).
"""
import math
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DPI = 200
S = DPI / 72.0  # pixels per PDF unit

# image op that makes text of orientation k upright
# k=0 upright, k=1 text reads bottom->top, k=2 upside-down, k=3 text reads top->bottom
ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
# op that turns an upright template into orientation k
TPL_ROT = {0: None, 1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}
K_ANGLE = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}  # text direction, degrees CCW, y-up


def _rot(img, code):
    return img if code is None else cv2.rotate(img, code)


def tok(s):
    return re.sub(r"[^A-Z0-9#]", "", s.upper())


def render_page(pdf_path, page_no, dpi=DPI):
    """Render one page (1-based) to a grey uint8 array using poppler."""
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "p")
        subprocess.run(
            ["pdftoppm", "-png", "-gray", "-r", str(dpi), "-f", str(page_no), "-l", str(page_no), "-singlefile", pdf_path, prefix],
            check=True, capture_output=True,
        )
        img = cv2.imread(prefix + ".png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not render page {page_no}")
    return img


# --------------------------------------------------------------------------- template
def ocr_scan(img, target, orientations=(0, 1, 3), margin=3, stop_at_first=False):
    """OCR the page in the given text orientations; return every exact occurrence of `target`, best first.

    Each hit: upright template crop, OCR confidence, ink on the crop border, orientation k and the box in page pixels.
    """
    T = tok(target)
    Hh, Ww = img.shape
    hits = []
    for k in orientations:
        up = _rot(img, ROT[k])
        d = pytesseract.image_to_data(up, config="--psm 11", output_type=pytesseract.Output.DICT)
        found = False
        for i, t in enumerate(d["text"]):
            if not t.strip() or tok(t) != T:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            y0, y1, x0, x1 = max(0, y - margin), min(up.shape[0], y + h + margin), max(0, x - margin), min(up.shape[1], x + w + margin)
            crop = up[y0:y1, x0:x1]
            if crop.size == 0 or min(crop.shape) < 8:
                continue
            ring = np.concatenate([crop[:margin].ravel(), crop[-margin:].ravel(), crop[:, :margin].ravel(), crop[:, -margin:].ravel()])
            if k == 0:
                pb = (x, y, x + w, y + h)
            elif k == 1:      # image was rotated clockwise: page x = y_up, page y = H-1-x_up
                pb = (y, Hh - 1 - (x + w), y + h, Hh - 1 - x)
            else:             # k == 3, image was rotated counter-clockwise
                pb = (Ww - 1 - (y + h), x, Ww - 1 - y, x + w)
            hits.append(dict(crop=crop.copy(), conf=float(d["conf"][i]), ring=float((ring < 128).mean()), margin=margin, k=k, page_bbox_px=pb))
            found = True
        if found and stop_at_first:
            break
    hits.sort(key=lambda h: (-round(h["conf"] / 10), h["ring"]))
    return hits


def find_template(img, target, orientations=(0, 1, 3), margin=3):
    """Best clean sample of `target` (for template matching), or None."""
    hits = ocr_scan(img, target, orientations, margin, stop_at_first=True)
    return hits[0] if hits else None


# --------------------------------------------------------------------------- matching
def _peaks(res, thr, r):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    mx = cv2.dilate(res, k)
    ys, xs = np.where((res >= thr) & (res >= mx))
    return [(float(res[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]


def _nms(cands, dist_pt):
    cands = sorted(cands, key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(math.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"]) > dist_pt for k in kept):
            kept.append(c)
    return kept


def match_orthogonal(img, tpl, thr=0.90, scales=(1.0,)):
    """Find the template in the 4 right-angle orientations."""
    inv = 255 - img
    crop, m = tpl["crop"], tpl["margin"]
    out = []
    for sc in scales:
        t0 = crop if sc == 1.0 else cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        mm = max(1, int(round(m * sc)))
        for k in range(4):
            t = _rot(t0, TPL_ROT[k])
            th, tw = t.shape
            res = cv2.matchTemplate(inv, 255 - t, cv2.TM_CCOEFF_NORMED)
            for score, x, y in _peaks(res, thr, max(3, min(tw, th) // 2)):
                x0, y0, x1, y1 = (x + mm) / S, (y + mm) / S, (x + tw - mm) / S, (y + th - mm) / S
                out.append(dict(k=k, rotated=False, angle=K_ANGLE[k], score=score, scale=sc,
                                bbox=(x0, y0, x1, y1), cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                                text_len=((x1 - x0) if k in (0, 2) else (y1 - y0)),
                                text_h=((y1 - y0) if k in (0, 2) else (x1 - x0))))
    return _nms(out, 6.0)


def _rotated_template(tf, ang):
    h, w = tf.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, float(ang), 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sn + w * cs), int(h * cs + w * sn)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(tf, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.uint8)


def match_rotated(img, tpl, thr=0.80, step=5, down=2, coarse_thr=0.45, max_coarse=100):
    """Find the template at in-between angles (diagonal text).

    Stage 1: coarse search on a reduced image every `step` degrees with a permissive threshold.
    Stage 2: refine angle (+-4 deg, 1 deg steps) and position at full resolution; keep if score >= thr.
    """
    inv_full = 255 - img
    small = cv2.resize(img, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    inv = 255 - small
    crop = tpl["crop"]
    tf_full = (255 - crop).astype(np.float32)
    tf_small = (255 - cv2.resize(crop, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)).astype(np.float32)
    m_full = tpl["margin"]
    h0, w0 = crop.shape
    coarse = []
    for ang in range(step, 360, step):
        if ang % 90 == 0:
            continue
        t = _rotated_template(tf_small, ang)
        nh, nw = t.shape
        res = cv2.matchTemplate(inv, t, cv2.TM_CCOEFF_NORMED)
        for score, x, y in _peaks(res, coarse_thr, max(3, min(nw, nh) // 2)):
            coarse.append((score, ang, (x + nw / 2) * down, (y + nh / 2) * down))   # centre in full-res px
    coarse.sort(reverse=True)
    out = []
    Hh, Ww = inv_full.shape
    tpl_inv = (255 - crop).astype(np.uint8)
    for score, ang0, cx, cy in coarse[:max_coarse]:
        R = int(1.3 * max(w0, h0) + 12)
        xa, ya = int(max(0, cx - R)), int(max(0, cy - R))
        win = inv_full[ya:int(min(Hh, cy + R)), xa:int(min(Ww, cx + R))]
        if win.shape[0] <= h0 or win.shape[1] <= w0:
            continue
        c = (cx - xa, cy - ya)
        best = None
        for ang in range(ang0 - 4, ang0 + 5):
            # rotate the image patch so text becomes upright, then compare with the upright template
            M = cv2.getRotationMatrix2D(c, -float(ang), 1.0)
            rw = cv2.warpAffine(win, M, (win.shape[1], win.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
            r = cv2.matchTemplate(rw, tpl_inv, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if best is None or mv > best[0]:
                inv_M = cv2.invertAffineTransform(M)
                pc = inv_M @ np.array([ml[0] + w0 / 2, ml[1] + h0 / 2, 1.0])
                best = (mv, ang, xa + pc[0], ya + pc[1])
        if best is None or best[0] < thr:
            continue
        sc, ang, fx, fy = best
        a = math.radians(ang)
        tl, th_ = (w0 - 2 * m_full) / S, (h0 - 2 * m_full) / S
        hw = (abs(tl * math.cos(a)) + abs(th_ * math.sin(a))) / 2
        hh = (abs(tl * math.sin(a)) + abs(th_ * math.cos(a))) / 2
        cxp, cyp = fx / S, fy / S
        a_mod = ang % 360
        if min(a_mod % 90, 90 - a_mod % 90) <= 4.5:
            continue  # (near-)right-angle text is handled by the right-angle search
        out.append(dict(k=None, rotated=True, angle=float(a_mod), score=float(sc), scale=1.0,
                        bbox=(cxp - hw, cyp - hh, cxp + hw, cyp + hh), cx=cxp, cy=cyp, text_len=tl, text_h=th_))
    return _nms(out, 8.0)


# --------------------------------------------------------------------------- vector check for longer marks
def build_glyph_index(page, bold_width):
    """Boxes (x0, y0, x1, y1; y down) of thin stroked paths: candidate text glyph strokes of the page."""
    boxes = []
    for objs in (page.lines, page.curves):
        for o in objs:
            if o.get("stroke") and o.get("linewidth", 0) < bold_width - 0.01:
                boxes.append((o["x0"], o["top"], o["x1"], o["bottom"]))
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def has_trailing_glyphs(index, cand, n_chars):
    """True if glyph-sized strokes start right after the label in its reading direction (i.e. a longer mark such as SB12 vs SB1)."""
    if index is None or len(index) == 0 or cand["rotated"]:
        return False
    x0, y0, x1, y1 = cand["bbox"]
    h = cand["text_h"]
    cw = cand["text_len"] / max(1, n_chars)
    k = cand["k"]
    bx0, by0, bx1, by1 = index[:, 0], index[:, 1], index[:, 2], index[:, 3]
    small = ((bx1 - bx0) <= 1.5 * h) & ((by1 - by0) <= 1.5 * h)
    pad = 0.25 * h
    if k in (0, 2):
        start = (bx0 - x1) if k == 0 else (x0 - bx1)            # distance from label end to the glyph's near edge
        band = (by1 >= y0 - pad) & (by0 <= y1 + pad)
    else:
        start = (y0 - by1) if k == 1 else (by0 - y1)
        band = (bx1 >= x0 - pad) & (bx0 <= x1 + pad)
    sel = small & band & (start >= 0.1 * cw) & (start <= 1.5 * cw)
    if not sel.any():
        return False
    ux0, uy0, ux1, uy1 = bx0[sel].min(), by0[sel].min(), bx1[sel].max(), by1[sel].max()
    cross = (uy1 - uy0) if k in (0, 2) else (ux1 - ux0)          # extent across the reading direction
    return cross >= 0.4 * h


# --------------------------------------------------------------------------- verification
_IGNORABLE_TAIL = set("IL")  # a vertical line touching the label is often read as I / L


def _read(crop):
    crop = cv2.copyMakeBorder(crop, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(crop, config="--psm 7").strip()


def _judge(text, target, require_confirm, veto_other=False):
    T = tok(target).lstrip("#")
    fam = re.match(r"[A-Z]+", T)
    fam = fam.group(0) if fam else ""
    tokens = [tok(t).lstrip("#") for t in re.split(r"\s+", text.upper()) if tok(t).lstrip("#")]
    if any(t == T for t in tokens):
        return True, "confirmed"
    for t in tokens:
        if t.startswith(T) and len(t) > len(T):
            tail = t[len(T):]
            if not set(tail) <= _IGNORABLE_TAIL:
                return False, f"longer mark '{t}'"
    if veto_other and fam:
        for t in tokens:
            if t.startswith(fam) and any(ch.isdigit() for ch in t) and t != T and not t.startswith(T):
                return False, f"OCR reads a different mark '{t}'"
    if require_confirm:
        return False, "OCR could not confirm label"
    return True, "unconfirmed (kept)"


def verify(img, cand, target, suffix_check=True, glyph_index=None):
    veto_other = cand.get("method") != "vector"          # exact vector matches are trusted over OCR; weaker detectors are not
    """Re-read the candidate with its surroundings. Returns (ok, reason, ocr_text)."""
    H, W = img.shape
    x0, y0, x1, y1 = [v * S for v in cand["bbox"]]
    L, hgt = cand["text_len"] * S, cand["text_h"] * S
    if not cand["rotated"]:
        k = cand["k"]
        es, ee, ec = 0.6 * L, 1.3 * L, 0.5 * hgt
        if k == 0:
            r = (x0 - es, y0 - ec, x1 + ee, y1 + ec)
        elif k == 2:
            r = (x0 - ee, y0 - ec, x1 + es, y1 + ec)
        elif k == 1:
            r = (x0 - ec, y0 - ee, x1 + ec, y1 + es)
        else:
            r = (x0 - ec, y0 - es, x1 + ec, y1 + ee)
        xa, ya, xb, yb = int(max(0, r[0])), int(max(0, r[1])), int(min(W, r[2])), int(min(H, r[3]))
        crop = _rot(img[ya:yb, xa:xb], ROT[k])
        if crop.size == 0:
            return False, "empty crop", ""
        text = _read(crop)
        if not suffix_check:
            return True, "suffix check disabled", text
        if has_trailing_glyphs(glyph_index, cand, len(tok(target).lstrip("#"))):
            return False, "longer mark (extra characters after the label)", text
        ok, why = _judge(text, target, require_confirm=False, veto_other=veto_other)
        return ok, why, text
    # rotated text: straighten then read; confirmation required
    cx, cy = cand["cx"] * S, cand["cy"] * S
    R = int(1.6 * max(L, hgt) + 10)
    xa, ya = int(max(0, cx - R)), int(max(0, cy - R))
    sub = img[ya:int(min(H, cy + R)), xa:int(min(W, cx + R))]
    if sub.size == 0:
        return False, "empty crop", ""
    c = (cx - xa, cy - ya)
    M = cv2.getRotationMatrix2D(c, -cand["angle"], 1.0)
    rot = cv2.warpAffine(sub, M, (sub.shape[1], sub.shape[0]), flags=cv2.INTER_CUBIC, borderValue=255)
    bx0, bx1 = int(max(0, c[0] - 0.6 * L - 0.6 * L)), int(min(rot.shape[1], c[0] + 0.5 * L + 1.3 * L))
    by0, by1 = int(max(0, c[1] - 1.1 * hgt)), int(min(rot.shape[0], c[1] + 1.1 * hgt))
    text = _read(rot[by0:by1, bx0:bx1])
    ok, why = _judge(text, target, require_confirm=True, veto_other=True)
    return ok, why, text
