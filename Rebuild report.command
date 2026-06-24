#!/bin/bash
# Double-click this file to regenerate the report (HTML + PDF) and open it.
# (If macOS blocks it on first launch: right-click → Open → Open.)

cd "$(dirname "$0")" || exit 1
echo "▶ Regenerating the report from the recordings in logs/ ..."
python3 host/dashboard.py || { echo "✗ Generation error"; read -r; exit 1; }

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  echo "▶ Creating the PDF ..."
  rm -f reports/holter_dashboard.pdf
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="reports/holter_dashboard.pdf" --virtual-time-budget=25000 \
    "file://$(pwd)/reports/holter_dashboard.html" >/dev/null 2>&1
fi

echo "✓ Done. Opening the results."
open reports/holter_dashboard.html
[ -f reports/holter_dashboard.pdf ] && open reports/holter_dashboard.pdf
echo "You can close this window."
