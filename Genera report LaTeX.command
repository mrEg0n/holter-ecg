#!/bin/bash
# Doppio click: ricalcola i dati, rigenera figure+tabelle e compila il PDF LaTeX.
# Il TESTO non viene toccato: sta in reports/holter_report.tex (lo editi a mano).
# (Se al primo avvio macOS lo blocca: tasto destro -> Apri -> Apri.)

cd "$(dirname "$0")" || exit 1

echo "▶ 1/3  Ricalcolo dati e figure (host/dashboard.py) ..."
python3 host/dashboard.py || { echo "✗ errore in dashboard.py"; read -r; exit 1; }

echo "▶ 2/3  Estraggo figure + tabelle per LaTeX (host/export_latex.py) ..."
python3 host/export_latex.py || { echo "✗ errore in export_latex.py"; read -r; exit 1; }

echo "▶ 3/3  Compilo reports/holter_report.tex ..."
cd reports || exit 1
latexmk -pdf -interaction=nonstopmode -halt-on-error holter_report.tex \
  || { echo "✗ errore di compilazione LaTeX (vedi reports/holter_report.log)"; read -r; exit 1; }
latexmk -c holter_report.tex >/dev/null 2>&1   # pulisce i file ausiliari

echo "✓ Fatto. Apro il PDF."
open holter_report.pdf
echo "Puoi chiudere questa finestra."
