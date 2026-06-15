"""
Estrae da  reports/holter_dashboard.html  i pezzi "automatici" del report
(le 18 figure e le 5 tabelle) e li scrive come file che il documento LaTeX
include. Cosi il testo lo lavori a mano in  reports/holter_report.tex , mentre
figure e tabelle si rigenerano da sole quando rifai le registrazioni:

    python3 host/dashboard.py        # ricalcola tutto -> HTML (con le figure)
    python3 host/export_latex.py     # HTML -> reports/figs/*.png + tables.tex

Output:
    reports/figs/NN_slug.png   una per figura, nell'ordine del report
    reports/tables.tex         \newcommand per ogni tabella + macro coi numeri
                               aggregati (\cumPVC, \pauseValley, ...)
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
    """Salva ogni <img base64> come PNG. Ritorna [(idx, filename, alt), ...]."""
    os.makedirs(FIG_DIR, exist_ok=True)
    # pulizia vecchie figure
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


# ---- parsing tabelle -------------------------------------------------------
class TableGrabber(HTMLParser):
    """Raccoglie ogni <table> come lista di righe; ogni riga lista di celle
    (testo + flag header). Conserva l'ordine di comparsa nel documento."""
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

# unicode -> LaTeX, cosi il documento compila anche con pdflatex puro
UNI = {
    "—": "---", "–": "--", "−": r"$-$", "·": r"$\cdot$",
    "×": r"$\times$", "≈": r"$\approx$", "±": r"$\pm$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≠": r"$\neq$",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "χ": r"$\chi$", "μ": r"$\mu$", "²": r"\textsuperscript{2}",
    "°": r"$^\circ$", "…": r"\ldots{}", " ": " ", " ": "~",
    "‘": "`", "’": "'", "“": "``", "”": "''",
}


def tex_escape(s):
    out = []
    for ch in s:
        out.append(TEX_ESC.get(ch, UNI.get(ch, ch)))
    return "".join(out)


def table_to_latex(rows, macro, caption, col_align=None):
    """Genera \newcommand{<macro>}{ ... tabular booktabs ... }. Le tabelle
    larghe (>=7 colonne) vengono scalate a \textwidth con \resizebox."""
    ncol = max(len(r) for r in rows)
    if col_align is None:
        # prima colonna a sinistra, le altre a destra (numeri)
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
    """Pesca i numeri aggregati dal report (stat-grid + frasi chiave) e li
    restituisce come dict di macro LaTeX -> valore stringa."""
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
    print(f"Figure salvate in {FIG_DIR}/ :")
    for i, fname, alt in figs:
        print(f"  {fname:42s}  <- {alt}")

    g = TableGrabber()
    g.feed(doc)
    tables = g.tables
    print(f"\nTabelle trovate: {len(tables)}")
    for i, t in enumerate(tables):
        head = " | ".join(c for c, _ in t[0]) if t else "(vuota)"
        print(f"  [{i}] righe={len(t):2d}  header: {head[:90]}")

    # Mappa per indice d'ordine -> (macro, caption). Adatta se cambia l'ordine.
    plan = [
        (r"\tableSessions",  "Sessioni registrate."),
        (r"\tableOutlier",   "Correlazione media N per sessione."),
        (r"\tableCross",     "Dinamica del ritmo e del burden tra sessioni."),
        (r"\tableSummary",   "Riassunto per sessione (metriche x sessioni)."),
        (r"\tableResp",      "Fase respiratoria e accoppiamento delle PVC."),
    ]
    blocks = []
    for idx, (macro, cap) in enumerate(plan):
        if idx < len(tables):
            blocks.append(table_to_latex(tables[idx], macro, cap))
        else:
            blocks.append(f"\\newcommand{{{macro}}}{{\\emph{{(tabella mancante)}}}}")

    scalars = extract_scalars(doc)
    print("\nNumeri aggregati:")
    for k, v in scalars.items():
        print(f"  {k:14s} = {v}")

    os.makedirs("reports", exist_ok=True)
    with open(TABLES_TEX, "w", encoding="utf-8") as f:
        f.write("% Generato da host/export_latex.py - NON modificare a mano.\n")
        f.write("% Numeri + tabelle si rigenerano dai dati; il testo sta in holter_report.tex\n\n")
        f.write("% --- numeri aggregati (snapshot dei dati correnti) ---\n")
        for k, v in scalars.items():
            f.write(f"\\newcommand{{{k}}}{{{tex_escape(v)}}}\n")
        f.write("\n% --- tabelle ---\n")
        f.write("\n\n".join(blocks))
        f.write("\n")
    print(f"\n✓ Scritto {TABLES_TEX}")
    print(f"  Figure: {len(figs)}  ·  tabelle: {len(plan)}  ·  macro numeri: {len(scalars)}")


if __name__ == "__main__":
    main()
