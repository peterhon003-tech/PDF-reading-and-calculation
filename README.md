# Steel Member Annotator

Upload a multi-page drawing PDF, type a member name (for example `SB4`) and get:

| Output | Content |
|---|---|
| **Annotated PDF** | The original pages with a line (same length as the member) drawn beside every matching member, plus a unique ID (`SB4-001`, `SB4-002` ...). **Solid blue** = found by the standard finder; **dotted lines in other colours** = found by one of the extra finders (see section 3c). A legend page is added at the end. Vector annotations - sharp at any zoom. |
| **PNG** | 300 dpi preview of every page that has annotations. |
| **Excel** | One row per member: coordinates, length in PDF units, length on paper (mm) and real length (mm) as live formulas driven by an Assumptions sheet, plus a per-page summary. |
| **JSON** | Everything above plus label match scores, distances, annotation geometry and the list of rejected candidates, for checking. |
| **ZIP** | All of the above in one download. |

Lengths are measured in **PDF units** (1 unit = 1/72 inch = 0.352778 mm on paper). Real length = paper length x drawing scale (default 1:100, editable).

---

## 1. Run with Docker (recommended)

Requirements: Docker 24+ (Docker Desktop on Windows/macOS is fine). Give the container at least **2 GB RAM** (about 1 GB is used while processing an A1 page).

```bash
cd sb_annotator

# build the image
docker build -t sb-annotator:1.0 .

# run it (results are kept in the named volume "sb_data")
docker run -d --name sb-annotator \
  -p 8000:8000 \
  -v sb_data:/data \
  --memory 4g \
  --restart unless-stopped \
  sb-annotator:1.0
```

Open **http://localhost:8000**, choose the PDF, type the member name, press **Annotate & calculate**.

Check it is healthy / see logs:

```bash
docker ps                                  # STATUS should say "healthy" after ~20 s
curl http://localhost:8000/healthz         # -> ok
docker logs -f sb-annotator
```

### Docker Compose

```bash
docker compose up -d --build      # start
docker compose logs -f            # logs
docker compose down               # stop (keeps the data volume)
```

### Update / stop / clean up

```bash
docker compose up -d --build              # rebuild after changing the code
docker stop sb-annotator && docker rm sb-annotator
docker volume rm sb_data                  # also deletes stored results
```

### Optional: pre-calculated Excel values
Excel and LibreOffice calculate the formulas when the file is opened. Some previewers (phone/e-mail viewers, pandas) show empty cells for formulas that were never calculated. To ship calculated values inside the file, build with LibreOffice included (adds ~400 MB to the image):

```bash
docker build --build-arg WITH_LIBREOFFICE=1 -t sb-annotator:1.0 .
```

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SB_MAX_UPLOAD_MB` | 200 | Largest PDF accepted |
| `SB_MAX_PAGES` | 60 | Most pages processed per upload |
| `SB_JOB_TTL_HOURS` | 24 | Uploads and results are deleted after this time |
| `SB_WORKERS` | 1 | Jobs processed in parallel (each needs ~1.5 GB RAM for A1 pages) |
| `SB_DATA_DIR` | `/data` | Where jobs are stored (mount a volume here) |

Example: `docker run ... -e SB_MAX_UPLOAD_MB=500 -e SB_WORKERS=2 --memory 6g ...`

### Putting it on a server
* The app has **no login**. Put it behind your company VPN, or behind a reverse proxy that adds authentication and HTTPS.
* nginx example:
  ```nginx
  server {
    listen 443 ssl;
    server_name annotator.example.com;
    client_max_body_size 200m;                 # must be >= SB_MAX_UPLOAD_MB
    location / {
      proxy_pass http://127.0.0.1:8000;
      proxy_set_header Host $host;
      proxy_read_timeout 300s;
      auth_basic "Restricted"; auth_basic_user_file /etc/nginx/.htpasswd;
    }
  }
  ```
* Uploaded drawings are stored under `/data/jobs/<random id>/` and removed after `SB_JOB_TTL_HOURS`.

---

## 2. Run without Docker

```bash
sudo apt install tesseract-ocr poppler-utils      # macOS: brew install tesseract poppler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                                      # http://localhost:8000
```

Command line (no UI):

```bash
python -m sb_core.cli drawing.pdf SB4 --out ./out [--scale 100] [--offset 4] [--no-diagonal-text]
python -m sb_core.cli drawing.pdf SB4 --out ./out --extra all --colors colour=#FF00FF   # every extra finder, custom colour (see 3c)
```

Self-checks: `python tests/test_geometry.py`, `python tests/test_vecmatch.py`, `python tests/test_finders.py`

---

## 3. How it works

1. **Find the label.** CAD-exported PDFs normally have *outlined* text (no text layer), so the label is located visually: OCR (tesseract) finds clean samples of the name you typed, in three text orientations.
2. **Find every copy - three independent detectors.**
   * **Exact vector matching (primary).** Every character is a set of thin stroke paths, so the same label is the *same set of vertices* wherever it appears (shifted, or turned by 90 degrees). Matching those vertices is exact and does not depend on resolution or anti-aliasing. A label that differs in even one glyph (SB3 vs SB1) is rejected.
   * **Pixel template matching and raw OCR hits (cross-checks).** Anything they find that the vector matcher did not is kept, but must also pass the OCR check below.
   * **Diagonal text (exact).** The same stroke template is rotated to any angle and fitted onto the page's stroke vertices, so a diagonal label is matched exactly (score 1.000) like a right-angle one, with no OCR needed. A longer-mark check (e.g. SB91 vs SB9) is done on the strokes after the label.
   * **Diagonal text (pixel fallback).** The older rotated pixel search still runs as a cross-check; each of its finds must be confirmed by OCR.
3. **Reject look-alikes.** Longer marks (SB4A, SB41, SB12 when searching SB1) are rejected by looking for extra characters after the label; a label whose OCR text reads as a different mark is rejected when it came from a cross-check detector.
4. **Find the member.** The thickest black stroke width on the page is taken as the member line weight (override in *Advanced*). Collinear, touching bold segments are merged into one run (any angle). Each label takes the nearest bold run parallel to the text (a non-parallel run only if the label touches it, <= 3 units).
5. **Account for every label.** A label that was found is either annotated or listed as skipped with a reason - nothing is dropped silently. On the PNG preview an **orange box** marks a skipped label that deserves a look (for example "no bold member within 8 units": a schedule-table row, or a label tied to its member by a leader line) and a **grey box** marks a longer mark that was skipped on purpose.
6. **Annotate.** A blue 1.44 pt line of identical length is drawn 4 units to the right of vertical members and 4 units below horizontal ones (diagonals: perpendicular, lower-right side). The ID is placed on the outer side at the spot that overlaps the least existing drawing, with a white halo.
7. **Export.** PDF, PNG, Excel and JSON. IDs run continuously through the whole document (page order, left to right, top to bottom).

Typical speed (one CPU core): about 80 s per A1 page with every cross-check on. The OCR cross-check and the diagonal search can be switched off under *Advanced settings* for a faster run.

## 3c. Extra ways to find the member (dotted lines, different colours)

The standard finder only accepts the **thickest black line**. Many drawings draw steel members differently, e.g. as **purple dashed lines** (every dash is a small filled polygon) or in another line weight. Every label that the standard finder cannot place is now tried with the methods below, in this order. The first one that finds a member wins, and the member is annotated with a **dotted line in that method's colour** (same length and offset as before), so you can see how each one was found.

| Method | What it looks for | Default look |
|---|---|---|
| standard (`bold`) | thickest black line, parallel to the label, within *Max label-to-member distance* | solid blue |
| `colour` | coloured lines, **including dashed lines**: dash polygons of one colour are grouped, and dashes closer than *Join dashes* are joined into one member. *Member colours* = `auto` (any saturated colour except red) or a list such as `#C78FFF;0.26,0.31,0.94` | dotted green `#00A63F` |
| `width` | dark lines whose width is between *Line width from / to* (default 0.9-4 pt), plus any *More exact widths* | dotted orange `#FF7F00` |
| `leader` | a thin black line that starts at the label (up to 4 bent pieces) and ends within *Leader end to member* of a member line | dotted brown `#8B4A0B` |
| `wide` (off by default) | members found by any of the above, but up to *Wider search factor* x farther from the label, with a looser parallel test | dotted cyan `#00B8D9` |
| label without member | the label was found but no method placed it (schedule table, general note ...) - **not measured** | dotted magenta box `#E000E0` |

* Colours and the line style (`dotted`, `dashed`, `solid`) can be changed in *Advanced settings* or with `--colors`, `--dash-style`.
* Switch any method off (or all of them: `--extra none`) to get the old behaviour; the standard finder itself is unchanged.
* Every member has a **Found by** column in Excel and `found_by` in the JSON. The legend page (PDF), the strip under the PNG and the Legend list in the UI show how many members each method found.
* Extra finds are *less certain* than the standard one - always check the dotted ones against the drawing (the `wide` method in particular can pick a line in a nearby detail).
* **Bold width fix:** the auto-detected member width now ignores strokes thicker than `--bold-max-width` (default 4 pt). Before, a title-block or border strip drawn 6-24 pt wide could be taken as "the member width" and no black member was found at all.
* A dashed member line is cut where something else (for example a purple connection triangle) breaks the row of dashes, so one drawn line can be reported as two members.

## 3b. Checking a result yourself

```bash
python tests/audit_run.py <name>_members.json <name>_annotated.pdf [original.pdf]
```

It re-reads the **output files** (not the detection code) and checks that: every label found is annotated or listed as skipped; IDs are unique; no annotated member's OCR text names a different mark; every member has a line (blue, or the colour of its extra method) of identical length at the stated offset in the PDF (and no stray annotation lines if you pass the original PDF); and no two members share one line. Exit code 0 = all passed.

The Pages table in the UI (and `pages[].sources` in the JSON) shows how many labels each detector found. If the pixel or OCR detectors keep adding labels the vector matcher did not find, tell me - that is a signal worth investigating.

Other tests: `python tests/test_vecmatch.py`, `python tests/test_geometry.py`.

## 4. Limits and tips

* Works on **vector** PDFs. Scanned drawings (raster images) have no member lines to measure.
* Member lines must be the heaviest black stroke on the page. If members are coloured or a different weight, set *Member line width* under Advanced settings.
* Members labelled only through a leader line (label far from the member) are not annotated; they are listed as skipped, with the reason, in the Pages table and the JSON, and boxed in orange on the PNG.
* One continuous drawn line is one member, even if it passes over columns.
* Pages with a `/Rotate` entry or a non-zero MediaBox origin may not line up; a warning is shown.
* If nothing is found: check the spelling (the name must match the drawing text), lower the *Match threshold* (0.85), or raise *Max label-to-member distance*.
* Check the `rejected` list in the JSON to see what was skipped and why.


## Diagonal-member fix (this version)
Problem reported: only 3 of the 20 diagonal SB9 members on the roof-plan page were annotated. Causes found and fixed:
1. Diagonal labels were found only by a pixel search (5-degree steps, score cutoff) and then thrown away when OCR could not read the rotated text. They are now matched exactly on the stroke vertices at any angle (`vecmatch.match_rotated`).
2. The label angle (counter-clockwise, y up) was compared with the member angle (clockwise, y down), so "parallel" was wrong for diagonals and the wrong arm of a V could be chosen. Fixed in `members.associate`.
3. The ID text of members running down to the right was placed behind the start of the line, outside the member. Fixed in `annotate.place_text`.
New tests: `test_diagonal_label_picks_its_own_arm_of_a_v`, `test_id_text_stays_on_the_member_for_every_direction` (both fail on the previous code).


## Extra-finder version (this version)
Request: more options to find the steel member, and annotate those finds with a dotted line in a different colour.
* New `sb_core/finders.py` (methods `colour`, `width`, `leader`, `wide`), options in `Options`, CLI, web UI, Excel/JSON (`found_by`), PDF legend page and PNG legend strip.
* `members.detect_bold_width` ignores strokes above 4 pt; `members.associate` got an `along_margin` argument.
* New tests: `tests/test_finders.py`; `tests/audit_run.py` understands the coloured / dotted lines.
