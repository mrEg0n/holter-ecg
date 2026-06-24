# legacy/

Earlier, standalone report generators from before the project settled on its
current pipeline. They are **superseded** and not wired into anything, but are
kept here to show how the analysis and reporting evolved.

The current pipeline is:

```
python3 host/dashboard.py      # recompute everything -> reports/holter_dashboard.html (with figures)
python3 host/export_latex.py   # HTML -> reports/figs/*.png + reports/tables.tex
# then compile reports/holter_report.tex
```

What's in here:

- `generate_report.py` — early self-contained HTML report.
- `generate_report_pdf.py` — multi-page A4 PDF report (cover, methodology, HRV,
  Poincaré, AF screening, interpolated-vs-compensated appendix, strip-charts).
- `synthetic_report.py` — cross-session comparison report across N sessions.

Some analyses here (e.g. Poincaré plots, extended HRV) are not reproduced in the
current dashboard. They still run, but expect rough edges — they were the
scaffolding the final report grew out of.
