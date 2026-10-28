"""Command line use:  python -m sb_core.cli input.pdf SB4 --out ./out [--extra colour,width,leader,wide]"""
import argparse
import os

from . import finders
from .pipeline import Options, run_job


def _extra(v):
    v = v.strip().lower()
    if v in ("none", "off", ""):
        return ()
    if v == "all":
        return finders.EXTRA_ORDER
    out = tuple(x.strip() for x in v.split(",") if x.strip())
    bad = [x for x in out if x not in finders.EXTRA_ORDER]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown method {bad}; choose from {', '.join(finders.EXTRA_ORDER)}, all, none")
    return out


def _colors(v):
    out = {}
    for part in v.split(","):
        if not part.strip():
            continue
        k, _, h = part.partition("=")
        if k.strip() not in finders.DEFAULT_COLORS or not h:
            raise argparse.ArgumentTypeError(f"use name=#RRGGBB with name in {', '.join(finders.DEFAULT_COLORS)}")
        finders.hex_to_rgb(h)
        out[k.strip()] = h.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Annotate and measure steel members in a drawing PDF")
    ap.add_argument("pdf")
    ap.add_argument("member", help="member label text, e.g. SB4")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--offset", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--max-dist", type=float, default=8.0, help="label -> member distance for the standard finder (PDF units)")
    ap.add_argument("--no-ocr-audit", action="store_true", help="skip the OCR cross-check (faster)")
    ap.add_argument("--no-diagonal-text", action="store_true", help="skip the search for rotated (diagonal) labels")
    g = ap.add_argument_group("extra ways to find the member (fallbacks; drawn dotted in their own colour)")
    g.add_argument("--extra", type=_extra, default=finders.DEFAULT_EXTRA,
                   help="comma list of colour,width,leader,wide | all | none  (default colour,width,leader; wide is opt-in)")
    g.add_argument("--member-colors", default="auto", help='colour method: "auto" or e.g. "#C78FFF;0.26,0.31,0.94"')
    g.add_argument("--color-tol", type=float, default=0.06)
    g.add_argument("--dash-gap", type=float, default=6.0, help="join dashes closer than this (PDF units)")
    g.add_argument("--width-min", type=float, default=0.9)
    g.add_argument("--width-max", type=float, default=4.0)
    g.add_argument("--extra-widths", default="", help='width method: more exact widths, e.g. "0.5,6"')
    g.add_argument("--wide-factor", type=float, default=2.5)
    g.add_argument("--leader-tip-dist", type=float, default=6.0)
    g.add_argument("--bold-max-width", type=float, default=4.0, help="auto-detected member width ignores strokes thicker than this")
    g.add_argument("--dup-spacing", type=float, default=4.0, help="parallel overlapping lines closer than this count as one member (extra finders)")
    g.add_argument("--dash-style", choices=("dotted", "dashed", "solid"), default="dotted")
    g.add_argument("--colors", type=_colors, default={}, help="override line colours, e.g. colour=#FF00FF,width=#FFA500")
    g.add_argument("--no-unmatched-box", action="store_true", help="do not draw a dotted box around labels with no member")
    g.add_argument("--no-legend-page", action="store_true")
    a = ap.parse_args()
    base = os.path.splitext(os.path.basename(a.pdf))[0]
    opts = Options(member=a.member, offset=a.offset, scale=a.scale, threshold=a.threshold, max_dist=a.max_dist,
                   diagonal_text=not a.no_diagonal_text, ocr_audit=not a.no_ocr_audit,
                   extra_methods=a.extra, member_colors=a.member_colors, color_tol=a.color_tol, dash_gap=a.dash_gap,
                   width_min=a.width_min, width_max=a.width_max, extra_widths=a.extra_widths, wide_factor=a.wide_factor,
                   leader_tip_dist=a.leader_tip_dist, dup_spacing=a.dup_spacing, bold_max_width=a.bold_max_width, dash_style=a.dash_style,
                   method_colors=a.colors, mark_unmatched=not a.no_unmatched_box, legend_page=not a.no_legend_page)
    s = run_job(a.pdf, a.out, base, opts, lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    print(f"\n{s['totals']['members']} members; outputs in {a.out}")
    print("found by:", {k: v for k, v in s["totals"]["by_method"].items() if v})
    for m in s["members"]:
        print(m["id"], "page", m["page"], m["orientation"], m["length_pdf_units"], m["found_by"])


if __name__ == "__main__":
    main()
