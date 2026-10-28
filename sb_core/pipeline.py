"""End-to-end processing of one uploaded PDF."""
import datetime
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import pdfplumber
from pypdf import PdfReader

from . import annotate, detect, export, finders, members as mem, vecmatch


@dataclass
class Options:
    member: str
    offset: float = 4.0            # PDF units between member and blue line
    scale: float = 100.0           # drawing scale denominator (1:N)
    mm_per_unit: float = 0.352778  # 1 PDF unit = 1/72 inch
    threshold: float = 0.90        # pixel template match score (supplementary raster detector)
    rotated_threshold: float = 0.65  # diagonal-text candidates; each must also be confirmed by OCR
    max_dist: float = 8.0          # max label -> parallel member distance (PDF units); non-parallel must touch (<=3)
    min_member_len: float = 30.0   # ignore bold runs shorter than this (PDF units)
    suffix_check: bool = True      # reject longer marks such as SB4A when searching SB4
    diagonal_text: bool = True     # also search text rotated at in-between angles (adds ~15 s per A1 page)
    ocr_audit: bool = True         # OCR the page in 3 orientations as an independent cross-check (adds ~30 s per A1 page)
    bold_width: Optional[float] = None  # None = auto-detect thickest black stroke
    max_pages: int = 60
    png_dpi: int = 300
    # ---- extra ways to find the member line (fallbacks, drawn as dotted lines in their own colour)
    extra_methods: tuple = finders.DEFAULT_EXTRA   # any of "colour", "width", "leader", "wide" (opt-in); () = standard finder only
    bold_max_width: float = 4.0    # auto-detected member line width ignores strokes thicker than this (frames, border strips)
    member_colors: str = "auto"    # "colour" method: "auto" = any saturated colour except red, or e.g. "#C78FFF;0.26,0.31,0.94"
    color_tol: float = 0.06        # how far a colour may differ from member_colors (0-1 per channel)
    dash_gap: float = 6.0          # dashes of a dashed member line closer than this are joined (PDF units)
    width_min: float = 0.9         # "width" method: dark lines from this width ...
    width_max: float = 4.0         # ... to this width (pt)
    extra_widths: str = ""         # "width" method: further exact widths, e.g. "0.5,6"
    wide_factor: float = 2.5       # "wide" method: search this many times max_dist
    leader_tip_dist: float = 6.0   # "leader" method: leader end must be within this of a member line
    dup_spacing: float = 4.0       # extra finders: two parallel overlapping lines closer than this count as ONE member (both edges of a section)
    dash_style: str = "dotted"     # look of fallback lines: "dotted" | "dashed" | "solid"
    method_colors: dict = field(default_factory=dict)   # e.g. {"colour": "#00A63F"}; see finders.DEFAULT_COLORS
    mark_unmatched: bool = True    # dotted box around labels for which no member was found
    legend_page: bool = True       # add a legend page to the annotated PDF


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "", s) or "member"


def _noop(frac, msg):
    pass


def _near(c, others, d=3.0):
    return any(math.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) <= d for o in others)


def _from_ocr_hit(h):
    x0, y0, x1, y1 = [v / detect.S for v in h["page_bbox_px"]]
    k = h["k"]
    return dict(k=k, rotated=False, angle=detect.K_ANGLE[k], score=h["conf"] / 100.0, scale=1.0, method="ocr",
                bbox=(x0, y0, x1, y1), cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                text_len=((x1 - x0) if k in (0, 2) else (y1 - y0)), text_h=((y1 - y0) if k in (0, 2) else (x1 - x0)))


def run_job(pdf_path: str, out_dir: str, base: str, opts: Options, progress: Callable[[float, str], None] = _noop):
    os.makedirs(out_dir, exist_ok=True)
    label = opts.member.strip()
    tag = _safe(label)
    n_pages = len(PdfReader(pdf_path).pages)
    if n_pages > opts.max_pages:
        raise ValueError(f"PDF has {n_pages} pages; the limit is {opts.max_pages}")

    pages_info, all_members, page_sizes, rejected_all = [], [], {}, []
    members_by_page, review_by_page, unmatched_by_page = {}, {}, {}
    g_raster, g_vec = None, None      # templates remembered from an earlier page (same font on every sheet)
    next_id = 1
    with pdfplumber.open(pdf_path) as pdf:
        for pno in range(1, n_pages + 1):
            f0 = (pno - 1) / n_pages
            span = 0.8 / n_pages
            info = dict(page=pno, warnings=[], rejected=[], candidates=0, members=0, bold_width=None,
                        sources=dict(vector=0, vector_rotated=0, raster=0, ocr=0, rotated=0))
            page = pdf.pages[pno - 1]
            w, h = float(page.width), float(page.height)
            info.update(width=w, height=h)
            page_sizes[pno] = (w, h)
            if page.rotation:
                info["warnings"].append(f"page is rotated {page.rotation} deg; coordinates may not line up")
            progress(f0, f"Page {pno}/{n_pages}: rendering")
            img = detect.render_page(pdf_path, pno)
            bw = opts.bold_width or mem.detect_bold_width(page, max_width=opts.bold_max_width)
            info["bold_width"] = bw

            progress(f0 + span * 0.1, f"Page {pno}/{n_pages}: reading text with OCR")
            hits = detect.ocr_scan(img, label, stop_at_first=not opts.ocr_audit)
            tpl_r = hits[0] if hits else g_raster
            if hits:
                g_raster = hits[0]

            # ---- 1. exact vector matching (primary)
            progress(f0 + span * 0.3, f"Page {pno}/{n_pages}: matching labels")
            V = vecmatch.thin_vertices(page, bw)
            cands, U = [], None
            for hh in hits[:5]:                       # the sample that yields the most exact copies is the cleanest one
                u = vecmatch.build_template(page, hh, bw)
                if u is None:
                    continue
                m = vecmatch.match(V, u)
                if len(m) > len(cands):
                    cands, U = m, u
            if U is not None:
                g_vec = U
            elif g_vec is not None:
                cands = vecmatch.match(V, g_vec)
            for c in cands:
                c["method"] = "vector"
            info["sources"]["vector"] = len(cands)

            # ---- 1b. exact vector matching of labels at any angle (diagonal members)
            U_any = U if U is not None else g_vec
            if opts.diagonal_text and U_any is not None:
                progress(f0 + span * 0.4, f"Page {pno}/{n_pages}: matching diagonal labels")
                for c in vecmatch.match_rotated(V, U_any):
                    if not _near(c, cands):
                        cands.append(c)
                        info["sources"]["vector_rotated"] += 1

            if tpl_r is None and not cands:
                info["warnings"].append(f"'{label}' was not found on this page")
                pages_info.append(info)
                continue

            # ---- 2. supplements: pixel template matching, then raw OCR hits
            if tpl_r is not None:
                for c in detect.match_orthogonal(img, tpl_r, opts.threshold):
                    if not _near(c, cands):
                        c["method"] = "raster"
                        cands.append(c)
                        info["sources"]["raster"] += 1
            for hh in hits:
                c = _from_ocr_hit(hh)
                if not _near(c, cands):
                    cands.append(c)
                    info["sources"]["ocr"] += 1
            if opts.diagonal_text and tpl_r is not None:
                progress(f0 + span * 0.5, f"Page {pno}/{n_pages}: searching diagonal labels")
                for c in detect.match_rotated(img, tpl_r, opts.rotated_threshold):
                    if not _near(c, cands):
                        c["method"] = "raster-rotated"
                        cands.append(c)
                        info["sources"]["rotated"] += 1
            info["candidates"] = len(cands)

            # ---- 3. verify each label, then pair it with a member
            progress(f0 + span * 0.7, f"Page {pno}/{n_pages}: pairing labels with members")
            finder = finders.MemberFinder(page, bw, opts)
            methods = ["bold"] + [m for m in finders.EXTRA_ORDER if m in tuple(opts.extra_methods)]
            rank = {m: i for i, m in enumerate(finders.ALL_METHODS)}
            glyphs = detect.build_glyph_index(page, bw)
            found = {}
            n_chars = len(detect.tok(label).lstrip("#"))
            for c in cands:
                bb = [round(v, 1) for v in c["bbox"]]
                if c["method"] == "vector-rotated":          # exact stroke match: trusted over OCR, which cannot read rotated text well
                    if opts.suffix_check and vecmatch.trailing_rotated(V, c, n_chars):
                        ok, why, text = False, "longer mark (extra characters after the label)", ""
                    else:
                        ok, why, text = True, "exact vector match", ""
                else:
                    ok, why, text = detect.verify(img, c, label, opts.suffix_check, glyphs)
                if not ok:
                    info["rejected"].append(dict(reason=why, method=c["method"], score=round(c["score"], 3), ocr_text=text, bbox=bb))
                    continue
                a = finder.find(c, methods)
                if a is None:
                    info["rejected"].append(dict(reason=f"no member found within {opts.max_dist:g} units by: {', '.join(methods)} (e.g. schedule table or note)",
                                                 method=c["method"], score=round(c["score"], 3), ocr_text=text, bbox=bb))
                    continue
                key = (round(a["run"]["p0"][0], 1), round(a["run"]["p0"][1], 1), round(a["run"]["p1"][0], 1), round(a["run"]["p1"][1], 1))
                key = min(key[:2] + key[2:], key[2:] + key[:2])
                found.setdefault(key, []).append((c, a, why, text))

            # extra finders can see both edges of one member (section views, double lines): keep one, count both labels
            keys = list(found)
            best_of = {k: min(found[k], key=lambda g: (rank[g[1]["method"]], -g[0]["score"])) for k in keys}
            for i, ki in enumerate(keys):
                for kj in keys[i + 1:]:
                    if ki not in found or kj not in found:
                        continue
                    ai, aj = best_of[ki][1], best_of[kj][1]
                    if ai["method"] == "bold" and aj["method"] == "bold":
                        continue                                  # the standard finder keeps its old behaviour
                    if finders.same_member(ai["run"], aj["run"], opts.dup_spacing):
                        keep, drop = (ki, kj) if (rank[ai["method"]], ai["dist"]) <= (rank[aj["method"]], aj["dist"]) else (kj, ki)
                        found[keep].extend(found.pop(drop))
                        best_of[keep] = min(found[keep], key=lambda g: (rank[g[1]["method"]], -g[0]["score"]))

            # accounting: every label found must end as a member or as a recorded rejection
            accepted_labels = sum(len(g) for g in found.values())
            if accepted_labels + len(info["rejected"]) != len(cands):
                info["warnings"].append("internal accounting mismatch: some labels were not recorded")
            info["labels_accepted"] = accepted_labels

            page_members = []
            for key, group in found.items():
                c, a, why, text = min(group, key=lambda g: (rank[g[1]["method"]], -g[0]["score"]))
                start, end, theta = annotate.canonical(a["run"])
                L = math.hypot(end[0] - start[0], end[1] - start[1])
                page_members.append(dict(
                    label=label, page=pno, orientation=annotate.orientation(theta), angle_deg=round(theta, 3),
                    start=dict(x_top=round(start[0], 3), y_top=round(start[1], 3), x_pdf=round(start[0], 3), y_pdf=round(h - start[1], 3)),
                    end=dict(x_top=round(end[0], 3), y_top=round(end[1], 3), x_pdf=round(end[0], 3), y_pdf=round(h - end[1], 3)),
                    length_pdf_units=round(L, 3),
                    length_paper_mm=round(L * opts.mm_per_unit, 3),
                    length_real_mm=round(L * opts.mm_per_unit * opts.scale, 1),
                    n_segments_merged=a["run"]["n"],
                    label_match=dict(method=c["method"], bbox_top=[round(v, 2) for v in c["bbox"]], text_direction_deg=c["angle"],
                                     score=round(c["score"], 3), ocr_status=why, ocr_text=text, labels_pointing_here=len(group)),
                    found_by=a["method"],
                    leader_path=a.get("leader"),
                    label_to_member_distance=round(a["dist"], 2),
                    next_nearest_member_distance=None if a["next_dist"] is None else round(a["next_dist"], 2),
                    label_parallel_to_member=a["parallel"],
                    _run=a["run"],
                ))
            page_members.sort(key=lambda m: (round(min(m["start"]["x_top"], m["end"]["x_top"])), min(m["start"]["y_top"], m["end"]["y_top"])))

            # IDs run through the whole document, so the page can be finished now and its raster released
            for m in page_members:
                m["id"] = f"{tag}-{next_id:03d}"
                next_id += 1
                run = m.pop("_run")
                start, end, theta = annotate.canonical(run)
                ann = annotate.make_annotation(start, end, theta, opts.offset, h)
                t = annotate.place_text(img, ann, m["id"], h)
                st = finders.style_for(m["found_by"], opts)
                m["annotation"] = dict(
                    style=dict(method=m["found_by"], color=st["color"], dash=st["dash"]),
                    side=ann["side"], offset=ann["offset"],
                    start=dict(x_pdf=round(ann["a"][0], 3), y_pdf=round(ann["a"][1], 3), x_top=round(ann["a"][0], 3), y_top=round(h - ann["a"][1], 3)),
                    end=dict(x_pdf=round(ann["b"][0], 3), y_pdf=round(ann["b"][1], 3), x_top=round(ann["b"][0], 3), y_top=round(h - ann["b"][1], 3)),
                    length_pdf_units=round(math.hypot(ann["b"][0] - ann["a"][0], ann["b"][1] - ann["a"][1]), 3),
                    id_text=dict(base_x_pdf=round(t["base"][0], 3), base_y_pdf=round(t["base"][1], 3), rotation_deg=round(t["rot"], 2),
                                 position_fraction=t["fraction"], overlapped_ink_pixels=t["overlap_pixels"]))
            info["members"] = len(page_members)
            info["by_method"] = {k: sum(1 for m in page_members if m["found_by"] == k) for k in finders.ALL_METHODS}
            if page_members:
                members_by_page[pno] = page_members
                all_members.extend(page_members)
            review_by_page[pno] = [tuple(r["bbox"]) + ("longer" if r["reason"].startswith("longer mark") else "review",) for r in info["rejected"]]
            if opts.mark_unmatched:
                um = [tuple(r["bbox"]) for r in info["rejected"] if not r["reason"].startswith("longer mark")]
                if um:
                    unmatched_by_page[pno] = um
            rejected_all.extend([dict(page=pno, **r) for r in info["rejected"]])
            pages_info.append(info)
            del img

    # ---- outputs
    progress(0.82, "Writing annotated PDF")
    outputs = {}
    pdf_out = os.path.join(out_dir, f"{base}_{tag}_annotated.pdf")
    legend = []
    for meth in finders.ALL_METHODS:
        n_m = sum(1 for m in all_members if m["found_by"] == meth)
        if n_m:
            st = finders.style_for(meth, opts)
            legend.append(dict(method=meth, title=finders.TITLES[meth], color=st["color"], dash=st["dash"], count=n_m))
    n_um = sum(len(v) for v in unmatched_by_page.values())
    if n_um:
        legend.append(dict(method="unmatched", title=finders.TITLES["unmatched"], color=finders.style_for("unmatched", opts)["color"], dash="dotted", count=n_um))
    annotate.write_annotated_pdf(pdf_path, pdf_out, members_by_page, page_sizes, unmatched_by_page,
                                 legend if (opts.legend_page and any(l["method"] != "bold" for l in legend)) else None,
                                 finders.style_for("unmatched", opts)["color"])
    outputs["annotated_pdf"] = pdf_out

    pngs = []
    for k, pno in enumerate(sorted(members_by_page)):
        progress(0.86 + 0.08 * k / max(1, len(members_by_page)), f"Rendering PNG preview for page {pno}")
        pngs.append(annotate.render_png(pdf_out, pno, os.path.join(out_dir, f"{base}_{tag}_annotated_p{pno}_{opts.png_dpi}dpi.png"),
                                        opts.png_dpi,
                                        [b for b in review_by_page.get(pno, ()) if not (opts.mark_unmatched and b[4] == "review")],
                                        legend if any(l["method"] != "bold" for l in legend) else None))
    outputs["png"] = pngs

    total = round(sum(m["length_pdf_units"] for m in all_members), 3)
    summary = dict(
        input=os.path.basename(base), member=label, generated=datetime.datetime.now().isoformat(timespec="seconds"),
        options={k: v for k, v in asdict(opts).items()},
        parameters=dict(units="lengths and offsets are in PDF units (1 unit = 1/72 inch = 0.352778 mm on paper)",
                        real_length="length_real_mm = length_paper_mm x scale denominator",
                        coordinates="x_pdf,y_pdf: PDF native (origin bottom-left, y up); x_top,y_top: origin top-left (y down)",
                        member_definition="bold black stroke; collinear touching segments (gap<=4 units) merged; nearest parallel run to the label wins",
                        detection="labels found by exact vector-vertex matching (primary), pixel template matching and OCR (cross-checks); "
                                 "every label found is either paired with a member or listed in 'rejected' with the reason (PNG: orange box = please review, grey box = longer mark skipped on purpose)"),
        pages=[{k: v for k, v in p.items()} for p in pages_info],
        members=all_members,
        legend=legend,
        totals=dict(members=len(all_members), length_pdf_units=total,
                    by_method={k: sum(1 for m in all_members if m["found_by"] == k) for k in finders.ALL_METHODS},
                    length_paper_mm=round(total * opts.mm_per_unit, 2), length_real_mm=round(total * opts.mm_per_unit * opts.scale, 1)),
        rejected=rejected_all,
    )
    progress(0.95, "Writing Excel and JSON")
    xlsx = os.path.join(out_dir, f"{base}_{tag}_members.xlsx")
    js = os.path.join(out_dir, f"{base}_{tag}_members.json")
    export.write_xlsx(summary, xlsx)
    export.write_json(summary, js)
    outputs["excel"], outputs["json"] = xlsx, js
    zp = os.path.join(out_dir, f"{base}_{tag}_all_outputs.zip")
    export.write_zip([pdf_out, *pngs, xlsx, js], zp)
    outputs["zip"] = zp
    progress(1.0, "Done")
    summary["outputs"] = {k: ([os.path.basename(x) for x in v] if isinstance(v, list) else os.path.basename(v)) for k, v in outputs.items()}
    export.write_json(summary, js)
    return summary
