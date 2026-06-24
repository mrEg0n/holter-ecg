"""
Extracts the "automatic" parts of the report (the 18 figures and the 5 tables)
from  reports/holter_dashboard.html  and writes them as files that the LaTeX
document includes. This way you edit the prose by hand in  reports/holter_report.tex ,
while figures and tables regenerate themselves whenever you redo the recordings:

    python3 host/dashboard.py        # recompute everything -> HTML (with figures)
    python3 host/export_latex.py     # HTML -> reports/figs/*.png + tables.tex

Output:
    reports/figs/NN_slug.png   one per figure, in report order
    reports/tables.tex         \newcommand for each table + macros with the
                               aggregate numbers (\cumPVC, \pauseValley, ...)
"""
import base64
import html
import os
import re
from html.parser import HTMLParser

HTML_PATH = "reports/holter_dashboard.html"
FIG_DIR = "reports/figs"
TABLES_TEX = "reports/tables.tex"


def slugify(alt):
    s = alt.lower()
    s = s.replace("&amp;", "and").replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:40] or "fig"


def extract_figures(doc):
    """Save each <img base64> as a PNG. Returns [(idx, filename, alt), ...]."""
    os.makedirs(FIG_DIR, exist_ok=True)
    # clean up old figures
    for f in os.listdir(FIG_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(FIG_DIR, f))
    pat = re.compile(
        r'<img\s+src="data:image/png;base64,([^"]+)"[^>]*?alt="([^"]*)"',
        re.DOTALL)
    out = []
    for i, m in enumerate(pat.finditer(doc), start=1):
        b64, alt = m.group(1), html.unescape(m.group(2))
        fname = f"{i:02d}_{slugify(alt)}.png"
        with open(os.path.join(FIG_DIR, fname), "wb") as fh:
            fh.write(base64.b64decode(b64))
        out.append((i, fname, alt))
    return out


# ---- table parsing ---------------------------------------------------------
class TableGrabber(HTMLParser):
    """Collects each <table> as a list of rows; each row a list of cells
    (text + header flag). Preserves the order of appearance in the document."""
    def __init__(self):
        super().__init__()
        self.tables = []
        self._cur = None
        self._row = None
        self._cell = None
        self._is_th = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._cur = []
        elif tag == "tr" and self._cur is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._is_th = (tag == "th")

    def handle_endtag(self, tag):
        if tag == "table" and self._cur is not None:
            self.tables.append(self._cur)
            self._cur = None
        elif tag == "tr" and self._row is not None:
            self._cur.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            txt = html.unescape("".join(self._cell)).strip()
            txt = re.sub(r"\s+", " ", txt)
            self._row.append((txt, self._is_th))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}

# unicode -> LaTeX, so the document compiles even with plain pdflatex
UNI = {
    "—": "---", "–": "--", "−": r"$-$", "·": r"$\cdot$",
    "×": r"$\times$", "≈": r"$\approx$", "±": r"$\pm$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≠": r"$\neq$",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "χ": r"$\chi$", "μ": r"$\mu$", "²": r"\textsuperscript{2}",
    "<": r"$<$", ">": r"$>$",
    "°": r"$^\circ$", "…": r"\ldots{}", " ": " ", " ": "~",
    "‘": "`", "’": "'", "“": "``", "”": "''",
}


def tex_escape(s):
    out = []
    for ch in s:
        out.append(TEX_ESC.get(ch, UNI.get(ch, ch)))
    return "".join(out)


def table_to_latex(rows, macro, caption, col_align=None):
    """Generates \newcommand{<macro>}{ ... tabular booktabs ... }. Wide tables
    (>=7 columns) are scaled to \textwidth with \resizebox."""
    ncol = max(len(r) for r in rows)
    if col_align is None:
        # first column left-aligned, the rest right-aligned (numbers)
        col_align = "l" + "r" * (ncol - 1)
    wide = ncol >= 7
    inner = [f"\\begin{{tabular}}{{{col_align}}}", r"\toprule"]
    for ri, row in enumerate(rows):
        cells = [tex_escape(c) for c, _ in row]
        cells += [""] * (ncol - len(cells))
        is_header = len(row) > 0 and all(h for _, h in row)
        if is_header:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        inner.append(" & ".join(cells) + r" \\")
        if ri == 0:
            inner.append(r"\midrule")
    inner.append(r"\bottomrule")
    inner.append(r"\end{tabular}")
    body = "\n".join(inner)

    lines = [f"\\newcommand{{{macro}}}{{%", r"\begin{center}",
             r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}"]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(body)
        lines.append("}")
    else:
        lines.append(r"\footnotesize")
        lines.append(body)
    lines.append(r"\end{center}")
    lines.append("}")
    return "\n".join(lines)


def extract_scalars(doc):
    """Pulls the aggregate numbers from the report (stat-grid + key phrases) and
    returns them as a dict of LaTeX macro -> string value."""
    m = {}
    # stat-grid: <span class="v">VAL</span> ... <span class="l">LABEL</span>
    grid = re.findall(
        r'<span class="v">([^<]+)</span>\s*<span class="l">([^<]+)</span>', doc)
    def find(label_kw, default="?"):
        for v, l in grid:
            if label_kw in html.unescape(l).lower():
                return html.unescape(v).strip()
        return default
    m[r"\nSessions"]   = find("sessions analyzed")
    m[r"\totMinutes"]  = find("total minutes")
    m[r"\totBeats"]    = find("heartbeats classified")
    m[r"\totPVC"]      = find("pvcs detected")
    m[r"\cumBurden"]   = find("cumulative pvc burden")
    m[r"\exclMinutes"] = find("manually excluded")

    def grab(pat, default="?"):
        mm = re.search(pat, doc)
        return html.unescape(mm.group(1)).strip() if mm else default
    m[r"\pauseValley"] = grab(r"empirical valley\s*<b>([\d.]+)&times;")
    m[r"\pctInterp"]   = grab(r"<b>([\d.]+)% interpolated</b>")
    m[r"\pctComp"]     = grab(r"<b>([\d.]+)% compensated</b>")
    m[r"\methodBeats"] = grab(r"on ([\d,]+) classified beats")
    return m


def main():
    with open(HTML_PATH, encoding="utf-8") as f:
        doc = f.read()

    figs = extract_figures(doc)
    print(f"Figures saved in {FIG_DIR}/ :")
    for i, fname, alt in figs:
        print(f"  {fname:42s}  <- {alt}")

    g = TableGrabber()
    g.feed(doc)
    tables = g.tables
    print(f"\nTables found: {len(tables)}")
    for i, t in enumerate(tables):
        head = " | ".join(c for c, _ in t[0]) if t else "(empty)"
        print(f"  [{i}] righe={len(t):2d}  header: {head[:90]}")

    # Map from order index -> (macro, caption). Adjust if the order changes.
    plan = [
        (r"\tableSessions",  "Recorded sessions."),
        (r"\tableOutlier",   "Mean-N correlation per session."),
        (r"\tableCross",     "Cross-session rhythm and burden dynamics."),
        (r"\tableSummary",   "Per-session summary (metrics x sessions)."),
        (r"\tableResp",      "Respiratory phase and PVC coupling."),
    ]
    blocks = []
    for idx, (macro, cap) in enumerate(plan):
        if idx < len(tables):
            blocks.append(table_to_latex(tables[idx], macro, cap))
        else:
            blocks.append(f"\\newcommand{{{macro}}}{{\\emph{{(table missing)}}}}")

    scalars = extract_scalars(doc)
    print("\nAggregate numbers:")
    for k, v in scalars.items():
        print(f"  {k:14s} = {v}")

    os.makedirs("reports", exist_ok=True)
    with open(TABLES_TEX, "w", encoding="utf-8") as f:
        f.write("% Generated by host/export_latex.py - do not edit by hand.\n")
        f.write("% Numbers + tables regenerate from the data; the prose lives in holter_report.tex\n\n")
        f.write("% --- aggregate numbers (snapshot of the current data) ---\n")
        for k, v in scalars.items():
            f.write(f"\\newcommand{{{k}}}{{{tex_escape(v)}}}\n")
        f.write("\n% --- tables ---\n")
        f.write("\n\n".join(blocks))
        f.write("\n")
    print(f"\n✓ Wrote {TABLES_TEX}")
    print(f"  Figures: {len(figs)}  ·  tables: {len(plan)}  ·  number macros: {len(scalars)}")


if __name__ == "__main__":
    main()
