"""Web UI + JSON API for the steel member annotator.

POST /api/jobs            multipart: file (PDF), member, [offset, scale, threshold, max_dist, diagonal, suffix_check, bold_width,
                          extra_colour, extra_width, extra_leader, extra_wide (checkboxes = "on"), member_colors, color_tol, dash_gap,
                          width_min, width_max, extra_widths, wide_factor, leader_tip_dist, dash_style, mark_unmatched, legend_page,
                          col_colour, col_width, col_leader, col_wide (hex colours)]
GET  /api/jobs/<id>       job status / progress / results
GET  /api/jobs/<id>/files/<name>   download one output file
GET  /healthz             liveness probe
"""
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from sb_core import Options, finders, run_job

DATA_DIR = os.path.abspath(os.environ.get("SB_DATA_DIR", "./data"))
MAX_UPLOAD_MB = int(os.environ.get("SB_MAX_UPLOAD_MB", "200"))
MAX_PAGES = int(os.environ.get("SB_MAX_PAGES", "60"))
JOB_TTL_HOURS = float(os.environ.get("SB_JOB_TTL_HOURS", "24"))
WORKERS = int(os.environ.get("SB_WORKERS", "1"))

JOBS_DIR = os.path.join(DATA_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=WORKERS)
_lock = threading.Lock()

JOB_RE = re.compile(r"^[0-9a-f]{32}$")
MEMBER_RE = re.compile(r"^[A-Za-z0-9#][A-Za-z0-9#_\-./]{0,39}$")


# ----------------------------------------------------------------------------- job state on disk
def _job_dir(job_id):
    if not JOB_RE.match(job_id):
        abort(404)
    return os.path.join(JOBS_DIR, job_id)


def _write_status(job_id, **upd):
    path = os.path.join(JOBS_DIR, job_id, "status.json")
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                st = json.load(f)
        except FileNotFoundError:
            st = {}
        st.update(upd)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, path)


def _read_status(job_id):
    try:
        with open(os.path.join(_job_dir(job_id), "status.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        abort(404)


def _cleanup():
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for name in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, name)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _worker(job_id, opts, base):
    d = os.path.join(JOBS_DIR, job_id)
    last = [0.0]

    def progress(frac, msg):
        now = time.time()
        if frac >= 1.0 or now - last[0] > 0.4:
            last[0] = now
            _write_status(job_id, state="running", progress=round(frac, 3), message=msg)

    try:
        _write_status(job_id, state="running", progress=0.0, message="Starting")
        s = run_job(os.path.join(d, "input.pdf"), os.path.join(d, "out"), base, opts, progress)
        brief = dict(
            member=s["member"], totals=s["totals"], outputs=s["outputs"], legend=s.get("legend", []),
            pages=[dict(page=p["page"], candidates=p["candidates"], members=p["members"], warnings=p["warnings"], by_method=p.get("by_method", {}),
                        sources=p["sources"], accepted=p.get("labels_accepted", 0),
                        rejected=[r["reason"] for r in p["rejected"]]) for p in s["pages"]],
            members=[dict(id=m["id"], page=m["page"], orientation=m["orientation"], found_by=m["found_by"], length_pdf_units=m["length_pdf_units"],
                          length_paper_mm=m["length_paper_mm"], length_real_mm=m["length_real_mm"]) for m in s["members"]],
        )
        _write_status(job_id, state="done", progress=1.0, message="Done", result=brief)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _write_status(job_id, state="error", message=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB, max_pages=MAX_PAGES)


@app.get("/healthz")
def healthz():
    return "ok"


def _num(name, default, lo, hi):
    raw = (request.form.get(name) or "").strip()
    if raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        abort(400, f"{name} must be a number")
    if not lo <= v <= hi:
        abort(400, f"{name} must be between {lo} and {hi}")
    return v


@app.post("/api/jobs")
def create_job():
    f = request.files.get("file")
    member = (request.form.get("member") or "").strip()
    if not f or not f.filename:
        abort(400, "Please choose a PDF file")
    if not MEMBER_RE.match(member):
        abort(400, "Member name: 1-40 characters, letters/digits and # _ - . / only (e.g. SB4)")
    head = f.stream.read(5)
    f.stream.seek(0)
    if head != b"%PDF-":
        abort(400, "The uploaded file is not a PDF")
    extra = tuple(m for m in finders.EXTRA_ORDER if request.form.get(f"extra_{m}") == "on")
    member_colors = (request.form.get("member_colors") or "auto").strip()
    extra_widths = (request.form.get("extra_widths") or "").strip()
    try:
        finders.parse_colors(member_colors)
        finders.parse_widths(extra_widths)
    except ValueError as e:
        abort(400, str(e))
    method_colors = {}
    for m in finders.DEFAULT_COLORS:
        h = (request.form.get(f"col_{m}") or "").strip()
        if h:
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", h):
                abort(400, f"colour for {m} must look like #RRGGBB")
            method_colors[m] = h
    dash_style = request.form.get("dash_style") or "dotted"
    if dash_style not in ("dotted", "dashed", "solid"):
        abort(400, "dash_style must be dotted, dashed or solid")
    opts = Options(
        member=member,
        extra_methods=extra, member_colors=member_colors, extra_widths=extra_widths, dash_style=dash_style, method_colors=method_colors,
        color_tol=_num("color_tol", 0.06, 0.0, 0.5), dash_gap=_num("dash_gap", 6.0, 0.5, 50),
        width_min=_num("width_min", 0.9, 0.0, 50), width_max=_num("width_max", 4.0, 0.1, 50),
        wide_factor=_num("wide_factor", 2.5, 1, 10), leader_tip_dist=_num("leader_tip_dist", 6.0, 0.5, 50),
        mark_unmatched=request.form.get("mark_unmatched") == "on", legend_page=request.form.get("legend_page") == "on",
        offset=_num("offset", 4.0, 0.5, 100),
        scale=_num("scale", 100.0, 1, 100000),
        threshold=_num("threshold", 0.90, 0.60, 0.99),
        max_dist=_num("max_dist", 8.0, 2, 100),
        diagonal_text=request.form.get("diagonal") == "on",
        suffix_check=request.form.get("suffix_check") == "on",
        ocr_audit=request.form.get("ocr_audit") == "on",
        bold_width=(_num("bold_width", 0, 0, 50) or None),
        max_pages=MAX_PAGES,
    )
    _cleanup()
    job_id = uuid.uuid4().hex
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d)
    f.save(os.path.join(d, "input.pdf"))
    base = os.path.splitext(secure_filename(f.filename))[0] or "drawing"
    _write_status(job_id, state="queued", progress=0.0, message="Queued", member=member, filename=f.filename)
    executor.submit(_worker, job_id, opts, base)
    return jsonify(job_id=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    return jsonify(_read_status(job_id))


@app.get("/api/jobs/<job_id>/files/<path:name>")
def job_file(job_id, name):
    st = _read_status(job_id)
    allowed = set()
    for v in (st.get("result", {}).get("outputs") or {}).values():
        allowed.update(v if isinstance(v, list) else [v])
    if name not in allowed:
        abort(404)
    inline = name.lower().endswith(".png")
    return send_from_directory(os.path.join(_job_dir(job_id), "out"), name, as_attachment=not inline)


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=getattr(e, "description", "Bad request")), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify(error=f"File too large (limit {MAX_UPLOAD_MB} MB)"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False, threaded=True)
