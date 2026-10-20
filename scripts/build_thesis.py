#!/usr/bin/env python3
"""Thesis build pipeline (task m1-t12, closed in Month 4): Markdown -> one document.

The thesis is written as one Markdown file per chapter under ``docs/thesis/``
(order and metadata in ``docs/thesis/thesis.yaml``). This script

1. concatenates the chapters into ``docs/thesis/build/thesis.md`` with a title
   block, chapter numbering and a generated table of contents;
2. renders a self-contained ``thesis.html`` (Python ``markdown`` with tables,
   footnotes and TOC extensions) that prints cleanly to PDF from any browser;
3. if ``pandoc`` is on the PATH, also produces ``thesis.docx`` (optionally with
   a reference docx for the faculty template) and, when a LaTeX engine is
   available, ``thesis.pdf``.

Usage::

    python scripts/build_thesis.py                 # md + html (+ docx/pdf if pandoc)
    python scripts/build_thesis.py --reference-docx docs/thesis/faculty_template.docx
    make thesis

Figures are referenced relative to the repo root (``results/figures/...``), so
the build copies them next to the output and rewrites the links.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
THESIS = REPO / "docs" / "thesis"


def load_spec() -> dict:
    return yaml.safe_load((THESIS / "thesis.yaml").read_text(encoding="utf-8"))


def _rewrite_links(text: str, out_dir: Path) -> str:
    """Copy referenced figures into build/figures and point links at them."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def repl(m):
        alt, target = m.group(1), m.group(2)
        src = (REPO / target) if not target.startswith(("http", "/")) else None
        if src is not None and src.exists() and src.suffix.lower() in (".png", ".svg", ".jpg"):
            dst = fig_dir / src.name
            shutil.copy2(src, dst)
            return f"![{alt}](figures/{src.name})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, text)


def build_markdown(spec: dict, out_dir: Path) -> Path:
    parts = [
        f"# {spec['title']}",
        "",
        f"**{spec['author']}** · {spec['degree']} · {spec['institution']}",
        "",
        f"Supervisor: {spec['supervisor']} · Submission: {spec['submission']}",
        "",
        "[TOC]",
        "",
    ]
    for i, ch in enumerate(spec["chapters"]):
        path = THESIS / ch["file"]
        body = (
            path.read_text(encoding="utf-8")
            if path.exists()
            else f"*({ch['file']} not written yet)*\n"
        )
        number = (
            ""
            if ch.get("unnumbered")
            else f"{i}. " if ch.get("number") is None else f"{ch['number']}. "
        )
        # demote headings so chapter titles are level 1 and sections level 2
        body = re.sub(r"^(#{1,5}) ", lambda m: "#" * (len(m.group(1)) + 1) + " ", body, flags=re.M)
        parts += [
            f"# {number}{ch['title']}",
            "",
            _rewrite_links(body, out_dir),
            "",
            "\\newpage",
            "",
        ]
    md = "\n".join(parts)
    out = out_dir / "thesis.md"
    out.write_text(md, encoding="utf-8")
    return out


def build_html(md_path: Path, spec: dict) -> Path:
    import markdown

    text = md_path.read_text(encoding="utf-8").replace(
        "\\newpage", '<div style="page-break-after:always"></div>'
    )
    html = markdown.markdown(
        text,
        extensions=["tables", "footnotes", "toc", "fenced_code", "attr_list"],
        extension_configs={"toc": {"title": "Contents"}},
    )
    css = """
    body{font-family:Georgia,'Times New Roman',serif;max-width:820px;margin:40px auto;line-height:1.5;color:#111;padding:0 20px}
    h1{font-size:1.7em;margin-top:2.2em;border-bottom:1px solid #999;padding-bottom:.2em}
    h2{font-size:1.3em;margin-top:1.6em}h3{font-size:1.1em}
    table{border-collapse:collapse;font-size:.9em;margin:1em 0}th,td{border:1px solid #bbb;padding:4px 8px}
    th{background:#f2f2f2}img{max-width:100%}code{font-size:.9em}
    .toc{background:#fafafa;border:1px solid #ddd;padding:10px 20px}
    @media print{body{margin:0;max-width:none}}
    """
    out = md_path.with_suffix(".html")
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{spec['title']}</title>"
        f"<style>{css}</style></head><body>{html}</body></html>",
        encoding="utf-8",
    )
    return out


def build_pandoc(md_path: Path, reference_docx: str | None) -> list[Path]:
    outs = []
    if shutil.which("pandoc") is None:
        print("pandoc not found: skipping docx/pdf (install pandoc for the Word / PDF targets)")
        return outs
    docx = md_path.with_suffix(".docx")
    cmd = ["pandoc", str(md_path), "-o", str(docx), "--toc", "--number-sections"]
    if reference_docx:
        cmd += ["--reference-doc", reference_docx]
    subprocess.run(cmd, check=True, cwd=md_path.parent)
    outs.append(docx)
    for engine in ("xelatex", "pdflatex", "wkhtmltopdf"):
        if shutil.which(engine):
            pdf = md_path.with_suffix(".pdf")
            subprocess.run(
                ["pandoc", str(md_path), "-o", str(pdf), "--toc", f"--pdf-engine={engine}"],
                check=False,
                cwd=md_path.parent,
            )
            if pdf.exists():
                outs.append(pdf)
            break
    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(THESIS / "build"))
    ap.add_argument("--reference-docx", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    md = build_markdown(spec, out_dir)
    html = build_html(md, spec)
    extra = build_pandoc(md, args.reference_docx)
    print("built:", md, html, *extra, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
