# Thesis source and build (task m1-t12)

Month 1 recorded the decision to write the thesis in Markdown and build the submission formats from one source instead of maintaining a LaTeX template (`docs/logs/month1_log.md`, decision 1). This directory is that template.

* One file per chapter (order, numbering and metadata in `thesis.yaml`). Section drafts produced during the roadmap (`methods_draft.md`, `fairness_section.md`, `safety_section.md`, …) are the chapter files themselves, so nothing is copied twice.
* `python scripts/build_thesis.py` (or `make thesis`) writes `build/thesis.md`, a self-contained `build/thesis.html` (print to PDF from the browser) and, when `pandoc` is installed, `build/thesis.docx` using `--reference-docx` for the faculty Word template and `build/thesis.pdf` when a LaTeX engine is present.
* Figures are referenced by their repository path (`results/figures/…`); the build copies them into `build/figures/` so the output is portable.
* Citations: `[@key]` keys refer to `Literature_Review/GovGridMARL_references.bib`; with pandoc, add `--citeproc --bibliography …` to the command in `build_thesis.py` once the reference list is final (Month 12, m12-t6).

Chapter files not yet written render as a placeholder line so the full document always builds.
