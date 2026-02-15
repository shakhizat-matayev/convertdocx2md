#!/usr/bin/env python3
"""ssaf_docx_to_md.py

Convert SSAF Well Health Reports (.docx) into GraphRAG-ready Markdown.

Pipeline:
  1) DOCX -> raw Markdown (+ extracted images) via Pandoc (preferred)
  2) Post-process Markdown to:
     - add YAML front matter (well name, report status, iterations)
     - normalize headings
     - wrap images into Plot Blocks (+ plot_metadata placeholders)
     - normalize & tag tables (+ table_metadata)
     - normalize recommendations (+ recommendation_metadata)

This script is designed for batch processing.

Requirements:
  - Python 3.9+
  - Pandoc installed and available on PATH (recommended)

Optional fallback:
  - mammoth (pip install mammoth) if pandoc is unavailable

Example:
  python ssaf_docx_to_md.py input.docx -o output.md --media-dir media

Batch:
  python ssaf_docx_to_md.py ./reports --out-dir ./md --media-root ./md_media
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# -------------------------
# Utilities
# -------------------------

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_pandoc(docx_path: Path, raw_md_path: Path, media_dir: Path) -> None:
    """Run Pandoc to convert docx to markdown and extract embedded media."""
    media_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pandoc",
        str(docx_path),
        "--from=docx",
        "--to=markdown",
        "--wrap=none",
        "--markdown-headings=atx",
        f"--extract-media={str(media_dir)}",
        "-o",
        str(raw_md_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Pandoc conversion failed.\n"
            f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n"
        )


def run_mammoth_fallback(docx_path: Path, raw_md_path: Path, media_dir: Path) -> None:
    """Fallback conversion using mammoth (less accurate for tables)."""
    try:
        import mammoth  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Pandoc is not available and mammoth is not installed. "
            "Install pandoc (recommended) or `pip install mammoth`."
        ) from e

    media_dir.mkdir(parents=True, exist_ok=True)

    # mammoth image handling
    def _save_image(image):
        ext = image.content_type.split("/")[-1]
        name = f"image_{_save_image.counter:04d}.{ext}"
        _save_image.counter += 1
        out = media_dir / name
        with open(out, "wb") as f:
            f.write(image.read())
        return {"src": str(out).replace('\\\\', '/')}

    _save_image.counter = 1

    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_markdown(f, convert_image=mammoth.images.img_element(_save_image))

    md = result.value
    raw_md_path.write_text(md, encoding="utf-8")


# -------------------------
# SSAF-specific parsing & normalization
# -------------------------

@dataclass
class ReportMeta:
    well_name: Optional[str] = None
    report_status: Optional[str] = None
    iterations: Optional[int] = None


WELL_RE = re.compile(r"Well Health Report for:\s*(.+?)\s*$", re.IGNORECASE)
STATUS_RE = re.compile(r"\*\*Status:\*\*\s*(.+?)\s*$", re.IGNORECASE)
ITER_RE = re.compile(r"after\s+(\d+)\s+iteration\(s\)", re.IGNORECASE)


def extract_report_meta(md: str) -> ReportMeta:
    meta = ReportMeta()

    # Well name
    for line in md.splitlines():
        m = WELL_RE.search(line.strip("# "))
        if m:
            meta.well_name = m.group(1).strip()
            break

    # Status + iterations
    for line in md.splitlines():
        m = STATUS_RE.search(line)
        if m:
            status_line = m.group(1).strip()
            meta.report_status = status_line
            m2 = ITER_RE.search(status_line)
            if m2:
                try:
                    meta.iterations = int(m2.group(1))
                except ValueError:
                    pass
            break

    # Normalize status to short label if possible
    if meta.report_status:
        s = meta.report_status.lower()
        if "compliant" in s:
            meta.report_status = "compliant"
        elif "failed" in s or "non" in s and "compliant" in s:
            meta.report_status = "non_compliant"
        else:
            # keep original
            meta.report_status = meta.report_status

    return meta


def add_front_matter(md: str, meta: ReportMeta, source_file: str) -> str:
    # If YAML front matter already exists, keep as-is.
    if md.lstrip().startswith("---"):
        return md

    today = _dt.date.today().isoformat()
    fm = [
        "---",
        "document_type: ssaf_well_health_report",
        f"well_name: {meta.well_name or 'UNKNOWN'}",
        f"report_status: {meta.report_status or 'UNKNOWN'}",
        f"iterations: {meta.iterations if meta.iterations is not None else 'UNKNOWN'}",
        "domain: subsurface_well_surveillance",
        f"source_file: {source_file}",
        f"ingested_on: {today}",
        "---",
        "",
    ]
    return "\n".join(fm) + md


# Heading normalization:
# - Promote main sections to ##
# - Keep numbered subsections like 5.2.3 as ##
# - Analyst subheaders (e.g., Analyst's Synthesis) as ###

NUMBERED_HEADING_RE = re.compile(r"^(#{1,6})\s*(\d+(?:\.\d+)+)\s+(.*)$")


def normalize_headings(md: str) -> str:
    out_lines = []
    for line in md.splitlines():
        m = NUMBERED_HEADING_RE.match(line)
        if m:
            # Force numbered headings to level-2
            num = m.group(2)
            title = m.group(3).strip()
            out_lines.append(f"## {num} {title}")
            continue

        # normalize common top headings
        l = line.strip()
        if l.startswith("### Executive Summary"):
            out_lines.append("## Executive Summary")
        elif l.startswith("### Report Generation"):
            out_lines.append("## Report Generation")
        elif l.startswith("#### Analyst") or l.startswith("##### Analyst"):
            # Ensure analyst commentary is consistent
            out_lines.append("### " + l.lstrip('#').strip())
        else:
            out_lines.append(line)

    # Ensure a single top title
    joined = "\n".join(out_lines)
    meta = extract_report_meta(joined)
    title = f"# SSAF Well Health Report — {meta.well_name or 'UNKNOWN'}"

    # If first non-empty line is not a level-1 title, prepend
    lines2 = joined.splitlines()
    i = 0
    while i < len(lines2) and not lines2[i].strip():
        i += 1
    if i < len(lines2) and not lines2[i].startswith("# "):
        joined = title + "\n\n" + joined
    return joined


# Image patterns:
# Pandoc can output reference-style images: ![][image_xxx]
# Or inline: ![](media/image.png)
IMG_REF_RE = re.compile(r"^!\[\]\[(.+?)\]\s*$")
IMG_INLINE_RE = re.compile(r"^!\[.*?\]\((.+?)\)\s*$")


def classify_plot_type(section_heading: str) -> str:
    s = section_heading.lower()
    if "production performance" in s:
        return "production_vs_forecast"
    if "anomal" in s:
        return "anomaly_dashboard"
    if "stability" in s:
        return "stability"
    if "systemic performance" in s:
        return "systemic_performance"
    return "unknown"


def wrap_images_into_plot_blocks(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []

    current_section = ""
    fig_counter = 1

    def _emit_plot_block(img_line: str):
        nonlocal fig_counter
        plot_id = f"FIG_{fig_counter:03d}"
        fig_counter += 1
        plot_type = classify_plot_type(current_section)

        out.extend([
            "",  # spacing
            f"### Plot {plot_id}",
            "",
            img_line,
            "",
            "**Figure:** (add a short caption if available)",
            "",
            "**Observation:**",
            "",
            "**Interpretation:**",
            "",
            "```plot_metadata",
            f"plot_id: {plot_id}",
            f"plot_type: {plot_type}",
            "well: ${WELL_NAME}",
            "x_axis: UNKNOWN",
            "y_axis: UNKNOWN",
            "linked_anomalies: []",
            "confidence: UNKNOWN",
            "```",
            "",
        ])

    i = 0
    while i < len(lines):
        line = lines[i]
        # Track current section for classification
        if re.match(r"^##\s+", line):
            current_section = line.lstrip('#').strip()

        # Detect image lines
        if IMG_REF_RE.match(line.strip()) or IMG_INLINE_RE.match(line.strip()):
            # If previous line is already a plot wrapper heading, keep as-is.
            prev = out[-1] if out else ""
            if prev.startswith("### Plot FIG_"):
                out.append(line)
            else:
                _emit_plot_block(line.strip())
        else:
            out.append(line)
        i += 1

    # Substitute WELL_NAME placeholder if front matter exists
    meta = extract_report_meta(md)
    well = meta.well_name or "UNKNOWN"
    return "\n".join(out).replace("${WELL_NAME}", well)


# HTML table detection
HTML_TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)


def html_table_to_markdown(html: str) -> str:
    """Minimal HTML table -> Markdown converter without external deps."""
    # extract rows
    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE)
    table: List[List[str]] = []
    for r in rows:
        # header cells
        cells = re.findall(r"<(t[hd])[^>]*>(.*?)</t[hd]>", r, flags=re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        row: List[str] = []
        for _, c in cells:
            # strip html tags
            c2 = re.sub(r"<[^>]+>", "", c)
            c2 = c2.replace("\n", " ").replace("\r", " ")
            c2 = re.sub(r"\s+", " ", c2).strip()
            # unescape typical markdown artifacts
            c2 = c2.replace("\\_", "_")
            row.append(c2)
        table.append(row)

    if not table:
        return html

    # Normalize widths
    max_cols = max(len(r) for r in table)
    for r in table:
        while len(r) < max_cols:
            r.append("")

    header = table[0]
    body = table[1:] if len(table) > 1 else []

    def md_row(r: List[str]) -> str:
        return "| " + " | ".join(r) + " |"

    md_lines = [md_row(header), "| " + " | ".join(["---"] * max_cols) + " |"]
    for r in body:
        md_lines.append(md_row(r))
    return "\n".join(md_lines)


def classify_table(md_table_block: str) -> str:
    t = md_table_block.lower()
    if "shap value" in t and "driver" in t:
        return "performance_drivers"
    if "early life" in t and "full life" in t and "metric" in t:
        return "stability_matrix"
    if "anomaly" in t and ("[x]" in t or "plunge" in t or "breakout" in t):
        return "anomaly_checklist"
    return "generic_table"


def tag_tables(md: str, well_name: str) -> str:
    """Convert HTML tables to Markdown (if present) and append table_metadata blocks."""
    out = md

    # 1) Convert HTML tables
    tables = list(HTML_TABLE_RE.finditer(out))
    # replace from end to start to keep indices valid
    for m in reversed(tables):
        html = m.group(0)
        md_table = html_table_to_markdown(html)
        out = out[:m.start()] + md_table + out[m.end():]

    # 2) Append metadata block after each Markdown table
    lines = out.splitlines()
    final: List[str] = []

    def is_md_table_row(s: str) -> bool:
        return s.strip().startswith("|") and s.strip().endswith("|")

    i = 0
    while i < len(lines):
        line = lines[i]
        if is_md_table_row(line):
            # capture contiguous table
            j = i
            block = []
            while j < len(lines) and is_md_table_row(lines[j]):
                block.append(lines[j])
                j += 1
            block_text = "\n".join(block)
            ttype = classify_table(block_text)
            final.extend(block)
            final.append("")
            final.append("```table_metadata")
            final.append(f"table_type: {ttype}")
            final.append(f"well: {well_name}")
            final.append("source_section: UNKNOWN")
            if ttype == "anomaly_checklist":
                final.append("entity_type: Anomaly")
                final.append("extraction_rule: checked_items_only")
            elif ttype == "stability_matrix":
                final.append("entity_type: PerformanceState")
                final.append("extraction_rule: row_per_metric_per_phase")
            elif ttype == "performance_drivers":
                final.append("entity_type: PerformanceDriver")
                final.append("weight_field: shap_value")
            else:
                final.append("entity_type: UNKNOWN")
            final.append("```")
            final.append("")
            i = j
        else:
            final.append(line)
            i += 1

    return "\n".join(final)


# Recommendation normalization
REC_LINE_RE = re.compile(r"\*\*Recommendation\s*\d+\s*\((.+?)\)\s*:\*\*\s*(.*)$", re.IGNORECASE)


def normalize_recommendations(md: str, well_name: str) -> str:
    lines = md.splitlines()
    out: List[str] = []

    for line in lines:
        m = REC_LINE_RE.search(line)
        if m:
            rec_type = m.group(1).strip()
            rec_text = m.group(2).strip()
            urgency = "immediate" if re.search(r"\bimmediate\b", rec_text, re.IGNORECASE) else "unspecified"

            # naive anomaly mention extraction (extend as needed)
            anomalies = sorted(set(re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*\([^)]*\))?)\b", rec_text)))

            out.extend([
                "## Recommendation",
                "",
                f"**Type:** {rec_type}",
                f"**Urgency:** {urgency}",
                "",
                f"**Recommendation Text:** {rec_text}",
                "",
                "```recommendation_metadata",
                f"well: {well_name}",
                f"recommendation_type: {rec_type}",
                f"urgency: {urgency}",
                f"triggering_anomalies: {anomalies if anomalies else []}",
                "confidence: UNKNOWN",
                "```",
                "",
            ])
        else:
            out.append(line)

    return "\n".join(out)


def finalize(md: str) -> str:
    # Light cleanup
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


# -------------------------
# Main processing function
# -------------------------

def process_docx(docx_path: Path, out_md_path: Path, media_dir: Path, keep_raw: bool = False) -> Tuple[Path, Path]:
    """Convert and normalize a single docx."""
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    raw_md_path = out_md_path.with_suffix(".raw.md")

    if _which("pandoc"):
        run_pandoc(docx_path, raw_md_path, media_dir)
    else:
        run_mammoth_fallback(docx_path, raw_md_path, media_dir)

    raw_md = raw_md_path.read_text(encoding="utf-8", errors="replace")

    meta = extract_report_meta(raw_md)
    md = add_front_matter(raw_md, meta, source_file=docx_path.name)
    md = normalize_headings(md)

    # Wrap images (plot blocks)
    md = wrap_images_into_plot_blocks(md)

    # Tag & convert tables
    well_name = meta.well_name or "UNKNOWN"
    md = tag_tables(md, well_name=well_name)

    # Normalize recommendations
    md = normalize_recommendations(md, well_name=well_name)

    md = finalize(md)
    out_md_path.write_text(md, encoding="utf-8")

    if not keep_raw:
        try:
            raw_md_path.unlink()
        except Exception:
            pass

    return out_md_path, media_dir


def iter_docx_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".docx":
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.docx") if p.is_file()])
    return []


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert SSAF .docx reports into GraphRAG-ready Markdown")
    ap.add_argument("input", help="Path to a .docx file or a folder containing .docx files")
    ap.add_argument("-o", "--output", help="Output .md path (single file mode)")
    ap.add_argument("--out-dir", help="Output directory (batch mode)")
    ap.add_argument("--media-dir", help="Media directory for extracted images (single file mode)")
    ap.add_argument("--media-root", help="Root folder to store media per report (batch mode)")
    ap.add_argument("--keep-raw", action="store_true", help="Keep intermediate .raw.md files")

    args = ap.parse_args(argv)
    in_path = Path(args.input)

    files = iter_docx_files(in_path)
    if not files:
        ap.error("No .docx files found at the provided input path")

    # Single file mode
    if len(files) == 1 and (args.output or not args.out_dir):
        docx = files[0]
        out_md = Path(args.output) if args.output else docx.with_suffix(".md")
        media = Path(args.media_dir) if args.media_dir else out_md.parent / (out_md.stem + "_media")
        process_docx(docx, out_md, media, keep_raw=args.keep_raw)
        print(f"Wrote: {out_md}")
        print(f"Media: {media}")
        return 0

    # Batch mode
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "md"
    media_root = Path(args.media_root) if args.media_root else out_dir / "media"

    out_dir.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)

    for docx in files:
        out_md = out_dir / (docx.stem + ".md")
        media = media_root / docx.stem
        process_docx(docx, out_md, media, keep_raw=args.keep_raw)
        print(f"OK: {docx.name} -> {out_md.name}")

    print(f"Batch complete. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
