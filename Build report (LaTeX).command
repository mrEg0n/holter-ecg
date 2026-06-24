#!/bin/bash
# Double-click: recompute the data, regenerate figures+tables and compile the LaTeX PDF.
# The PROSE is left untouched: it lives in reports/holter_report.tex (edit it by hand).
# (If macOS blocks it on first launch: right-click -> Open -> Open.)

cd "$(dirname "$0")" || exit 1

echo "▶ 1/3  Recomputing data and figures (host/dashboard.py) ..."
python3 host/dashboard.py || { echo "✗ error in dashboard.py"; read -r; exit 1; }

echo "▶ 2/3  Extracting figures + tables for LaTeX (host/export_latex.py) ..."
python3 host/export_latex.py || { echo "✗ error in export_latex.py"; read -r; exit 1; }

echo "▶ 3/3  Compiling reports/holter_report.tex ..."
cd reports || exit 1
latexmk -pdf -interaction=nonstopmode -halt-on-error holter_report.tex \
  || { echo "✗ LaTeX compilation error (see reports/holter_report.log)"; read -r; exit 1; }
latexmk -c holter_report.tex >/dev/null 2>&1   # clean up auxiliary files

echo "✓ Done. Opening the PDF."
open holter_report.pdf
echo "You can close this window."
