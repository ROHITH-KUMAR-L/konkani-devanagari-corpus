"""Download free e-paper issues from epaper.thegoan.net and OCR them to text.

The site (ReadWhere platform) hosts Bhaangarbhuin (Konkani) plus Marathi and
English papers/magazines. This script:
  1. discovers every newspaper/magazine title on the site and keeps only those
     whose language is in --languages (default: Konkani);
  2. walks each title's archive listing, newest first;
  3. for each issue that is FREE to download (not archived / not paid / no
     login), downloads every page as a PDF through the site's own per-page
     download link. Archived issues are sold by the publisher and are skipped;
  4. extracts text: uses the PDF text layer if it is real Devanagari Unicode,
     otherwise renders the page and OCRs it with Tesseract (mar+hin);
  5. writes one UTF-8/LF text file per page and per issue, plus a metadata CSV.

Runs are resumable: pages already downloaded/OCR'd are skipped.

Requirements: pip install pymupdf pytesseract pillow requests
              Tesseract 5 + mar/hin traineddata (tessdata_best), e.g.
              winget install UB-Mannheim.TesseractOCR
              and put mar.traineddata, hin.traineddata in ~/tessdata

Usage:
    # fetch + OCR all free Konkani issues (Bhaangarbhuin)
    python scripts/fetch_epaper.py fetch
    # only the 10 newest issues, or a date range
    python scripts/fetch_epaper.py fetch --max-issues 10
    python scripts/fetch_epaper.py fetch --since 2026-01-01 --until 2026-06-30
    # OCR PDFs that are already on disk (e.g. the 20 PDFs in data/raw)
    python scripts/fetch_epaper.py ocr --pdf-dir data/raw/02_news_and_media \
        --out data/interim/02_news_and_media/local_pdfs
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import requests
from PIL import Image

BASE = "https://epaper.thegoan.net"
SEED_TITLE = ("BHAANGARBHUIN", "23246")
USER_AGENT = ("Mozilla/5.0 (konkani-devanagari-corpus; academic NLP research; "
              "+https://github.com/ROHITH-KUMAR-L/konkani-devanagari-corpus)")
ISSUE_LINK = re.compile(r"/r/(\d+)")
CONFIG_FIELD = re.compile(r"'(\w+)'\s*:\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[^,\n]+)")
META_FIELDS = ["title", "issue_id", "date", "language", "type", "pages",
               "pages_ocr", "pages_text_layer", "tokens", "status", "source_url"]


# ---------------------------------------------------------------- HTTP

class Client:
    def __init__(self, delay):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def get(self, url, **kw):
        for attempt in range(4):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.session.get(url, timeout=60, **kw)
                if r.status_code == 200:
                    return r
                if r.status_code in (404, 403):
                    return None
            except requests.RequestException as e:
                print(f"    ! {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
        return None


# ---------------------------------------------------------------- site parsing

def reader_config(client, issue_id):
    """Parse the JS config object of an issue's reader page."""
    r = client.get(f"{BASE}/r/{issue_id}")
    if r is None:
        return None
    cfg = {}
    for key, val in CONFIG_FIELD.findall(r.text):
        val = val.strip().strip("'\"")
        cfg.setdefault(key, val)
    if "otherTitles" in cfg:  # JSON blob, parse separately
        m = re.search(r"'otherTitles'\s*:\s*(\{.*?\})\s*,\s*\n", r.text)
        cfg["otherTitles"] = json.loads(m.group(1)) if m else {}
    cfg["_url"] = r.url
    return cfg


def discover_titles(client):
    """All titles listed on the site: {title_id: {name, slug, type}}."""
    issues = list_issues(client, *SEED_TITLE, max_pages=1)
    cfg = reader_config(client, issues[0]) if issues else None
    titles = {}
    if cfg and isinstance(cfg.get("otherTitles"), dict):
        for tid, t in cfg["otherTitles"].items():
            titles[tid] = {"name": t["name"].strip(), "slug": t["slug"], "type": t["type"]}
    titles.setdefault(SEED_TITLE[1], {"name": SEED_TITLE[0], "slug": SEED_TITLE[0],
                                      "type": "newspaper"})
    return titles


def list_issues(client, slug, title_id, max_pages=1000):
    """Yield issue ids from the title's archive listing, newest first."""
    seen = []
    for page in range(1, max_pages + 1):
        r = client.get(f"{BASE}/edition/{slug}/{title_id}", params={"page": page})
        if r is None:
            break
        new = [i for i in dict.fromkeys(ISSUE_LINK.findall(r.text)) if i not in seen]
        if not new:
            break
        seen.extend(new)
    return seen


def page_download_links(client, ptype, issue_id):
    # the endpoint only accepts the fixed range the site's reader uses
    r = client.get(f"{BASE}/reader/download/pagemeta/get/{ptype}/{issue_id}/1-100")
    if r is None:
        return {}
    try:
        data = r.json()
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {int(k): v["download_link"] for k, v in data.items()
            if isinstance(v, dict) and v.get("download_link")}


# ---------------------------------------------------------------- text extraction

def setup_tesseract(args):
    cmd = args.tesseract or shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not Path(cmd).exists() and not shutil.which(cmd):
        sys.exit("Tesseract not found. Install it (winget install UB-Mannheim.TesseractOCR) "
                 "or pass --tesseract.")
    pytesseract.pytesseract.tesseract_cmd = cmd
    # env var avoids quoting problems with --tessdata-dir on Windows paths
    os.environ["TESSDATA_PREFIX"] = args.tessdata
    for lang in args.lang.split("+"):
        if not (Path(args.tessdata) / f"{lang}.traineddata").exists():
            sys.exit(f"{lang}.traineddata not found in {args.tessdata}; download it from "
                     "https://github.com/tesseract-ocr/tessdata_best")


def devanagari_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0, 0
    deva = sum(0x0900 <= ord(c) <= 0x097F for c in letters)
    return deva / len(letters), deva


def normalise(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def pdf_page_to_text(pdf_path, args):
    """Return (text, method) for a single-page PDF (or page 0 of a PDF)."""
    with fitz.open(pdf_path) as doc:
        return page_text(doc[0], args)


def page_text(page, args):
    layer = page.get_text()
    ratio, deva = devanagari_ratio(layer)
    if ratio >= 0.5 and deva >= 200:
        return normalise(layer), "text_layer"
    pix = page.get_pixmap(dpi=args.dpi, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    text = pytesseract.image_to_string(img, lang=args.lang, config=f"--psm {args.psm}")
    return normalise(text), "ocr"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- fetch command

def process_issue(client, title, issue_id, cfg, args, pool):
    ptype = cfg.get("pType", title["type"])
    date = cfg.get("publishDate", "unknown")
    num_pages = int(cfg.get("numPages", "0") or 0)
    tag = f"{date}_{issue_id}"
    pdf_dir = Path(args.pdf_out) / title["slug"] / tag
    txt_dir = Path(args.out) / title["slug"] / tag
    issue_txt = Path(args.out) / title["slug"] / f"{tag}.txt"
    row = {"title": title["name"], "issue_id": issue_id, "date": date,
           "language": cfg.get("language", ""), "type": ptype, "pages": num_pages,
           "source_url": f"{BASE}/r/{issue_id}"}

    if issue_txt.exists() and not args.force:
        text = issue_txt.read_text(encoding="utf-8")
        return {**row, "tokens": len(text.split()), "status": "cached"}

    links = page_download_links(client, ptype, issue_id)
    if not links:
        return {**row, "status": "no_free_download"}

    jobs = []
    for pnum, link in sorted(links.items()):
        pdf_path = pdf_dir / f"page_{pnum:02d}.pdf"
        if not pdf_path.exists():
            r = client.get(BASE + link)
            if r is None or not r.content.startswith(b"%PDF"):
                print(f"    ! page {pnum}: download failed")
                continue
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(r.content)
        jobs.append((pnum, pdf_path))

    def run(job):
        pnum, pdf_path = job
        txt_path = txt_dir / f"page_{pnum:02d}.txt"
        if txt_path.exists() and not args.force:
            return pnum, txt_path.read_text(encoding="utf-8"), "cached"
        text, method = pdf_page_to_text(pdf_path, args)
        write_text(txt_path, text)
        return pnum, text, method

    results = sorted(pool.map(run, jobs))
    write_text(issue_txt, "\n".join(t for _, t, _ in results))
    tokens = sum(len(t.split()) for _, t, _ in results)
    return {**row, "pages_ocr": sum(m == "ocr" for *_, m in results),
            "pages_text_layer": sum(m == "text_layer" for *_, m in results),
            "tokens": tokens, "status": "ok" if len(results) == num_pages else "partial"}


def cmd_fetch(args):
    setup_tesseract(args)
    client = Client(args.delay)
    wanted = {l.lower() for l in args.languages}
    meta_path = Path(args.meta)
    done = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            done = {r["issue_id"] for r in csv.DictReader(f) if r["status"] in ("ok", "cached")}
    new_file = not meta_path.exists()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_f = open(meta_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(meta_f, fieldnames=META_FIELDS)
    if new_file:
        writer.writeheader()

    titles = discover_titles(client)
    if args.titles:
        keep = {t.lower() for t in args.titles}
        titles = {k: v for k, v in titles.items() if v["slug"].lower() in keep or k in keep}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for tid, title in titles.items():
            issues = list_issues(client, title["slug"], tid, max_pages=args.max_listing_pages)
            if not issues:
                print(f"[{title['name']}] no issues listed")
                continue
            first = reader_config(client, issues[0]) or {}
            lang = first.get("language", "?")
            if lang.lower() not in wanted:
                print(f"[{title['name']}] skipped: language {lang}")
                continue
            print(f"[{title['name']}] {lang}, {len(issues)} issues listed")

            fetched = archived_run = 0
            for issue_id in issues:
                if args.max_issues and fetched >= args.max_issues:
                    break
                if issue_id in done and not args.force:
                    continue
                cfg = first if issue_id == issues[0] else reader_config(client, issue_id)
                if not cfg:
                    continue
                date = cfg.get("publishDate", "")
                if args.until and date and date > args.until:
                    continue
                if args.since and date and date < args.since:
                    break  # listing is newest-first
                paid = (cfg.get("archived") == "1" or cfg.get("isPaid") == "true"
                        or cfg.get("needsAuth") == "true")
                if paid:
                    archived_run += 1
                    print(f"  {date} {issue_id}: archived/paid, skipped")
                    if archived_run >= args.stop_after_archived:
                        print(f"  {archived_run} archived issues in a row; stopping this title")
                        break
                    continue
                archived_run = 0
                t0 = time.time()
                row = process_issue(client, title, issue_id, cfg, args, pool)
                writer.writerow(row)
                meta_f.flush()
                fetched += row["status"] in ("ok", "partial", "cached")
                print(f"  {date} {issue_id}: {row['status']}, {row.get('pages', 0)} pages, "
                      f"{row.get('tokens', 0)} tokens ({time.time() - t0:.0f}s)")
    meta_f.close()


# ---------------------------------------------------------------- ocr command

def cmd_ocr(args):
    setup_tesseract(args)
    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdfs = sorted(Path(args.pdf_dir).glob(pattern))
    out = Path(args.out)

    def run(job):
        pdf, pno = job
        with fitz.open(pdf) as doc:
            return pdf, pno, *page_text(doc[pno], args)

    for pdf in pdfs:
        target = out / (pdf.stem + ".txt")
        if target.exists() and not args.force:
            print(f"{pdf.name}: cached")
            continue
        with fitz.open(pdf) as doc:
            jobs = [(pdf, i) for i in range(len(doc))]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run, jobs))
        text = "\n".join(t for *_, t, _ in results)
        write_text(target, text)
        n_ocr = sum(m == "ocr" for *_, m in results)
        print(f"{pdf.name}: {len(results)} pages ({n_ocr} OCR), {len(text.split())} tokens "
              f"({time.time() - t0:.0f}s) -> {target}")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def ocr_opts(p):
        p.add_argument("--lang", default="mar+hin", help="Tesseract languages (default mar+hin)")
        p.add_argument("--dpi", type=int, default=150,
                       help="render resolution; these PDFs use oversized page units, "
                            "so 150 already gives ~3000px wide pages")
        p.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
        p.add_argument("--tessdata", default=os.environ.get("TESSDATA_PREFIX",
                                                            str(Path.home() / "tessdata")))
        p.add_argument("--tesseract", help="path to tesseract executable")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                       help="pages OCR'd in parallel")
        p.add_argument("--force", action="store_true", help="redo pages that already exist")

    f = sub.add_parser("fetch", help="download free issues from the site and OCR them")
    f.add_argument("--languages", nargs="+", default=["Konkani"])
    f.add_argument("--titles", nargs="+", help="limit to these title slugs or ids")
    f.add_argument("--max-issues", type=int, default=0, help="per title; 0 = all free issues")
    f.add_argument("--since", help="YYYY-MM-DD, oldest issue date to fetch")
    f.add_argument("--until", help="YYYY-MM-DD, newest issue date to fetch")
    f.add_argument("--stop-after-archived", type=int, default=10,
                   help="stop a title after this many consecutive archived (paid) issues")
    f.add_argument("--max-listing-pages", type=int, default=1000)
    f.add_argument("--delay", type=float, default=1.0, help="seconds between HTTP requests")
    f.add_argument("--pdf-out", default="data/raw/02_news_and_media/epaper")
    f.add_argument("--out", default="data/interim/02_news_and_media/epaper")
    f.add_argument("--meta", default="metadata/epaper_issues.csv")
    ocr_opts(f)

    o = sub.add_parser("ocr", help="OCR PDFs already on disk")
    o.add_argument("--pdf-dir", default="data/raw/02_news_and_media")
    o.add_argument("--out", default="data/interim/02_news_and_media/local_pdfs")
    o.add_argument("--recursive", action="store_true", help="also search subfolders")
    ocr_opts(o)

    args = ap.parse_args()
    if args.cmd == "fetch":
        cmd_fetch(args)
    else:
        cmd_ocr(args)


if __name__ == "__main__":
    main()
