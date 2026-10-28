"""Independent audit of a finished run.

    python tests/audit_run.py <members.json> <annotated.pdf> [source.pdf]

Re-reads the OUTPUT files (not the detection code) and checks:
  1. accounting   - every label found is either annotated or listed as skipped, on every page
  2. unique IDs   - IDs are unique and consecutive
  3. wrong marks  - no annotated member's OCR text names a different mark (e.g. SB3 annotated while searching SB1)
  4. drawn lines  - every member has a line (blue = standard finder, other colour + dotted = extra finder) of identical length at the stated
                    offset in the PDF, and the PDF has no stray lines of those colours
  5. one run each - no two members share the same member line
Exit code 0 = all passed.
"""
import json
import math
import re
import sys

import pdfplumber


_CONFUSABLE = str.maketrans({"I": "1", "L": "1", "T": "1", "|": "1", "!": "1", "O": "0", "$": "S"})


def norm(t):
    """upper-case, map the usual OCR look-alikes (I/L/T/| for 1, O for 0, $ for S) and drop punctuation"""
    return re.sub(r"[^A-Z0-9]", "", t.upper().translate(_CONFUSABLE))


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(round(int(h[i:i + 2], 16) / 255.0, 2) for i in (0, 2, 4))


def blue_lines(page, rgb=(0, 0, 1)):
    """annotation lines of one colour (default blue = standard finder; dotted lines of the extra finders have other colours)"""
    out = []
    for o in page.lines:
        c = o.get("stroking_color")
        if c is not None and tuple(round(v, 2) for v in (c if isinstance(c, (list, tuple)) else (c,))) == tuple(round(v, 2) for v in rgb) and o["linewidth"] > 1:
            (x1, y1), (x2, y2) = o["pts"][0], o["pts"][-1]
            out.append((x1, y1, x2, y2))
    return out


def _member_rgb(m):
    st = (m.get("annotation") or {}).get("style") or {}
    return _hex_rgb(st["color"]) if st.get("color") else (0, 0, 1)


def main():
    js, pdf = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else None
    s = json.load(open(js, encoding="utf-8"))
    target = norm(s["member"])
    fam = re.match(r"[A-Z]+", target)
    fam = fam.group(0) if fam else ""
    fails = []

    # 1. accounting
    for p in s["pages"]:
        if p["candidates"] != p.get("labels_accepted", 0) + len(p["rejected"]):
            fails.append(f"page {p['page']}: {p['candidates']} labels found but {p.get('labels_accepted', 0)} annotated + {len(p['rejected'])} skipped")
    print(f"[1] accounting       : {'ok' if not any('labels found' in f for f in fails) else 'FAIL'}  ({sum(p['candidates'] for p in s['pages'])} labels found)")

    # 2. IDs
    ids = [m["id"] for m in s["members"]]
    exp = [f"{ids[0].rsplit('-', 1)[0]}-{i:03d}" for i in range(1, len(ids) + 1)] if ids else []
    if ids != exp or len(set(ids)) != len(ids):
        fails.append("IDs are not unique / consecutive")
    print(f"[2] unique IDs       : {'ok' if ids == exp else 'FAIL'}  ({len(ids)} members)")

    # 3. wrong marks
    wrong = []
    for m in s["members"]:
        for t in re.split(r"\s+", m["label_match"]["ocr_text"].upper()):
            t = norm(t)
            if fam and t.startswith(fam) and any(ch.isdigit() for ch in t) and not t.startswith(target):
                wrong.append((m["id"], m["label_match"]["ocr_text"]))
    for w in wrong:
        fails.append(f"{w[0]}: OCR text {w[1]!r} names a different mark")
    print(f"[3] wrong marks      : {'ok' if not wrong else 'FAIL ' + str(wrong)}")

    # 4. drawn lines in the output PDF
    drawn_ok = True
    with pdfplumber.open(pdf) as doc, (pdfplumber.open(src) if src else open(pdf, "rb")) as base:
        for pno, page in enumerate(doc.pages, 1):
            if pno > len(base.pages if src else doc.pages):
                break                                                   # legend page added at the end
            mine = [m for m in s["members"] if m["page"] == pno]
            colours = {_member_rgb(m) for m in mine} | {(0, 0, 1)}
            lines = {c: blue_lines(page, c) for c in colours}
            if src:
                for c in colours:
                    for b in blue_lines(base.pages[pno - 1], c):         # lines of that colour that were already in the drawing
                        for i, l in enumerate(lines[c]):
                            if all(abs(a - x) < 0.05 for a, x in zip(l, b)):
                                lines[c].pop(i)
                                break
            for m in mine:
                a = m["annotation"]
                pool = lines[_member_rgb(m)]
                want = (a["start"]["x_top"], a["start"]["y_top"], a["end"]["x_top"], a["end"]["y_top"])
                hit = next((l for l in pool if all(abs(x - y) < 0.05 for x, y in zip(l, want))), None)
                if hit is None:
                    fails.append(f"{m['id']}: no line at the stated position on page {pno}")
                    drawn_ok = False
                    continue
                pool.remove(hit)
                L = math.hypot(hit[2] - hit[0], hit[3] - hit[1])
                if abs(L - m["length_pdf_units"]) > 0.01:
                    fails.append(f"{m['id']}: drawn line length {L:.2f} != member length {m['length_pdf_units']:.2f}")
                    drawn_ok = False
                gap = math.hypot(a["start"]["x_top"] - m["start"]["x_top"], a["start"]["y_top"] - m["start"]["y_top"])
                if abs(gap - a["offset"]) > 0.01:
                    fails.append(f"{m['id']}: offset {gap:.2f} != {a['offset']}")
                    drawn_ok = False
            stray = sum(len(v) for v in lines.values())
            if stray and src:
                fails.append(f"page {pno}: {stray} stray annotation line(s) not belonging to any member")
                drawn_ok = False
    print(f"[4] drawn lines      : {'ok' if drawn_ok else 'FAIL'}")

    # 5. one member line each
    keys = [(m["page"], round(m["start"]["x_top"], 1), round(m["start"]["y_top"], 1), round(m["end"]["x_top"], 1), round(m["end"]["y_top"], 1)) for m in s["members"]]
    dup = len(keys) - len(set(keys))
    if dup:
        fails.append(f"{dup} member(s) share the same member line")
    print(f"[5] one run each     : {'ok' if not dup else 'FAIL'}")

    print("\nPASS" if not fails else "\nFAILED:\n  " + "\n  ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
