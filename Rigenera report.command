#!/bin/bash
# Doppio click su questo file per rigenerare il report (HTML + PDF) e aprirlo.
# (Se al primo avvio macOS lo blocca: tasto destro → Apri → Apri.)

cd "$(dirname "$0")" || exit 1
echo "▶ Rigenero il report dalle registrazioni in logs/ ..."
python3 host/dashboard.py || { echo "✗ Errore nella generazione"; read -r; exit 1; }

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  echo "▶ Creo il PDF ..."
  rm -f reports/holter_dashboard.pdf
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="reports/holter_dashboard.pdf" --virtual-time-budget=25000 \
    "file://$(pwd)/reports/holter_dashboard.html" >/dev/null 2>&1
fi

echo "✓ Fatto. Apro i risultati."
open reports/holter_dashboard.html
[ -f reports/holter_dashboard.pdf ] && open reports/holter_dashboard.pdf
echo "Puoi chiudere questa finestra."
