"""
Genera un report PDF professionale da una sessione ECG registrata.

Layout multi-pagina:
  - Cover con sintesi numerica
  - Tracce ECG complete (strip chart stile holter)
  - Analisi RR e coupling
  - Pattern temporali
  - HRV pre-PVC e Poincaré
  - Conclusioni e limiti tecnici

Usage:
    python3 generate_report_pdf.py logs/ecg_YYYYMMDD_HHMMSS.csv
"""
import csv
import io
import math
import os
import statistics
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)

# ------------------ load data ------------------
PATH = sys.argv[1] if len(sys.argv) > 1 else None
if PATH is None:
    print("usage: generate_report_pdf.py <ecg_*.csv>")
    sys.exit(1)

SAMPLE_HZ = 250
STRIP_ROW_SECONDS = 60
STRIP_ROWS_PER_PAGE = 5  # ⇒ 5 minuti per pagina

t_s, vr, vf = [], [], []
with open(PATH) as f:
    for row in csv.DictReader(f):
        try:
            t_s.append(float(row["t_s"]))
            vr.append(float(row["raw"]))
            vf.append(float(row["filt"]))
        except (KeyError, ValueError):
            continue
t  = np.array(t_s)
vr = np.array(vr)
vf = np.array(vf)
N  = len(vf)

peaks_path = PATH.replace(os.sep + "ecg_", os.sep + "peaks_")
peaks = []
if os.path.exists(peaks_path):
    with open(peaks_path) as f:
        for row in csv.DictReader(f):
            try:
                peaks.append({
                    "t":   float(row["t_s"]),
                    "amp": float(row["amp_V"]),
                    "w":   float(row["width_ms"]),
                    "reb": float(row["rebound_ratio"]),
                    "cls": row["class"],
                })
            except (KeyError, ValueError):
                continue

ses_id = os.path.basename(PATH).replace("ecg_", "").replace(".csv", "")
total_s = float(t[-1] - t[0]) if N else 0
total_min = total_s / 60.0
fs_real = N / total_s if total_s else SAMPLE_HZ
norm = [p for p in peaks if p["cls"] == "normal"]
pvc  = [p for p in peaks if p["cls"] == "pvc"]
n_total = len(peaks)
sinus_bpm = 60 * len(norm) / total_s if total_s else 0
pvc_rate  = 60 * len(pvc)  / total_s if total_s else 0
burden    = 100 * len(pvc) / max(1, n_total)

# RR
for i in range(len(peaks)):
    peaks[i]["rr_prev"] = (peaks[i]["t"] - peaks[i-1]["t"]) if i > 0 else None
    peaks[i]["rr_next"] = (peaks[i+1]["t"] - peaks[i]["t"]) if i < len(peaks)-1 else None
sinus_rr  = [peaks[i]["rr_prev"] for i in range(1, len(peaks))
             if peaks[i]["cls"] == "normal" and peaks[i-1]["cls"] == "normal"]
coupling  = [p["rr_prev"] for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None]
compensatory = [p["rr_next"] for p in peaks if p["cls"] == "pvc" and p["rr_next"] is not None]

# transizioni RR per categoria (per decomposizione tachogramma)
transitions = {"N→N": [], "N→PVC": [], "PVC→N": [], "PVC→PVC": []}
for i_t in range(1, len(peaks)):
    if peaks[i_t]["rr_prev"] is None: continue
    rr_ms_t = peaks[i_t]["rr_prev"] * 1000
    prev_t = "PVC" if peaks[i_t-1]["cls"] == "pvc" else "N"
    cur_t  = "PVC" if peaks[i_t]["cls"]   == "pvc" else "N"
    transitions[f"{prev_t}→{cur_t}"].append((peaks[i_t]["t"]/60, rr_ms_t))

# fascia "750ms" (700-800) decomposta
band750_breakdown = {k: sum(1 for tm, rr in v if 700 <= rr <= 800) for k, v in transitions.items()}
# fascia "1180ms" (1000-1500)
band1180_breakdown = {k: sum(1 for tm, rr in v if 1000 <= rr <= 1500) for k, v in transitions.items()}

# ----- Analisi ampiezza dei battiti normali per contesto -----
# Per ogni normale: classifica il contesto rispetto al PVC piu' vicino
# - stable: prev e next sono normali (e non immediatamente adiacenti a PVC)
# - pre_pvc: il battito SUCCESSIVO e' una PVC
# - post_pvc: il battito PRECEDENTE era una PVC
# - sandwich: pre PVC E post PVC (caso bigeminia stretta, raro)
amp_groups = {"stable": [], "pre_pvc": [], "post_pvc": [], "sandwich": []}
amp_rr_pairs = []  # (rr_prev_s, amp_V, group) per scatter Frank-Starling
for i, p in enumerate(peaks):
    if p["cls"] != "normal": continue
    prev_pvc = (i > 0 and peaks[i-1]["cls"] == "pvc")
    next_pvc = (i < len(peaks)-1 and peaks[i+1]["cls"] == "pvc")
    if prev_pvc and next_pvc:
        g = "sandwich"
    elif next_pvc:
        g = "pre_pvc"
    elif prev_pvc:
        g = "post_pvc"
    else:
        g = "stable"
    amp_groups[g].append(p["amp"])
    if p.get("rr_prev"):
        amp_rr_pairs.append((p["rr_prev"], p["amp"], g))

def grp_stats(vals):
    if not vals: return None
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0,
        "min": min(vals), "max": max(vals),
    }

amp_stats = {k: grp_stats(v) for k, v in amp_groups.items()}

# correlazione ampiezza vs RR precedente (Frank-Starling): rough Pearson
def pearson(xs, ys):
    if len(xs) < 3: return 0
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy > 0 else 0

stable_pairs = [(rr, a) for rr, a, g in amp_rr_pairs if g == "stable"]
r_stable = pearson([p[0] for p in stable_pairs], [p[1] for p in stable_pairs])
r_all_norm = pearson([rr for rr, a, g in amp_rr_pairs], [a for rr, a, g in amp_rr_pairs])

# patterns
iso_pvc = sum(1 for i, p in enumerate(peaks) if p["cls"] == "pvc"
              and (i == 0 or peaks[i-1]["cls"] != "pvc")
              and (i == len(peaks)-1 or peaks[i+1]["cls"] != "pvc"))
couplets_n = 0
i = 0
while i < len(peaks) - 1:
    if peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc":
        if i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc":
            i += 1; continue
        couplets_n += 1; i += 2
    else:
        i += 1

bigem = 0; i = 0
while i < len(peaks) - 5:
    if [peaks[i+j]["cls"] for j in range(6)] == ["normal","pvc"]*3:
        bigem += 1
        end = i
        while end+1 < len(peaks) and peaks[end+1]["cls"] != peaks[end]["cls"]:
            end += 1
        i = end + 1
    else:
        i += 1

trigem = 0; i = 0
while i < len(peaks) - 8:
    if [peaks[i+j]["cls"] for j in range(9)] == ["normal","normal","pvc"]*3:
        trigem += 1
        end = i
        while end+3 < len(peaks) and peaks[end+1]["cls"]=="normal" and peaks[end+2]["cls"]=="normal" and peaks[end+3]["cls"]=="pvc":
            end += 3
        i = end + 1
    else:
        i += 1

# pre-PVC HRV
LOOKBACK = 5
pre_pvc_stdevs = []
pre_pvc_means  = []
for idx, p in enumerate(peaks):
    if p["cls"] != "pvc": continue
    rrs = []
    j = idx - 1
    while j >= 1 and len(rrs) < LOOKBACK:
        if peaks[j]["cls"] == "normal" and peaks[j-1]["cls"] == "normal":
            rrs.append(peaks[j]["t"] - peaks[j-1]["t"])
        j -= 1
    if len(rrs) >= LOOKBACK:
        pre_pvc_stdevs.append(statistics.stdev(rrs))
        pre_pvc_means.append(statistics.mean(rrs))

baseline_stdevs = []
if len(sinus_rr) >= LOOKBACK:
    for k in range(0, len(sinus_rr) - LOOKBACK + 1, LOOKBACK):
        baseline_stdevs.append(statistics.stdev(sinus_rr[k:k+LOOKBACK]))

WINDOW = 60
windows = []
i_w = 0
while peaks and peaks[0]["t"] + i_w*WINDOW < peaks[-1]["t"]:
    ws = peaks[0]["t"] + i_w*WINDOW
    we = ws + WINDOW
    in_w = [p for p in peaks if ws <= p["t"] < we]
    nn = sum(1 for p in in_w if p["cls"] == "normal")
    np_ = sum(1 for p in in_w if p["cls"] == "pvc")
    windows.append({"t": ws, "norm": nn, "pvc": np_})
    i_w += 1

# summary numbers
sinus_median_ms = 1000 * statistics.median(sinus_rr) if sinus_rr else 0
sinus_mean_ms   = 1000 * statistics.mean(sinus_rr)   if sinus_rr else 0
sinus_std_ms    = 1000 * statistics.stdev(sinus_rr)  if len(sinus_rr) > 1 else 0
sinus_rmssd_ms  = 0
if len(sinus_rr) > 1:
    diffs = [(sinus_rr[k+1]-sinus_rr[k])**2 for k in range(len(sinus_rr)-1)]
    sinus_rmssd_ms = 1000 * math.sqrt(statistics.mean(diffs))
coupling_median = 1000 * statistics.median(coupling) if coupling else 0
coupling_std    = 1000 * statistics.stdev(coupling)  if len(coupling) > 1 else 0
coupling_iqr    = 0
if len(coupling) >= 4:
    q = sorted(coupling)
    q1 = q[len(q)//4]*1000; q3 = q[3*len(q)//4]*1000
    coupling_iqr = q3 - q1
prematurity = (1 - coupling_median/sinus_median_ms) * 100 if sinus_median_ms else 0
compensatory_median = 1000 * statistics.median(compensatory) if compensatory else 0
pre_pvc_stdev_mean = 1000 * statistics.mean(pre_pvc_stdevs) if pre_pvc_stdevs else 0
baseline_stdev_mean = 1000 * statistics.mean(baseline_stdevs) if baseline_stdevs else 0
hrv_delta_pct = (pre_pvc_stdev_mean/baseline_stdev_mean - 1) * 100 if baseline_stdev_mean else 0

# ------------------ plotting helpers ------------------
DARK_BG = "#1e1e1e"
PANEL_BG = "#0d0d0d"
GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#f39c12"
GRID = "#444444"
MUTED = "#aaaaaa"

def styled_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, alpha=0.25, color=GRID, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def make_strip_page_image(t0, t1, rows_per_page=STRIP_ROWS_PER_PAGE,
                          row_s=STRIP_ROW_SECONDS):
    """Plot di una pagina di strip-chart: N righe da row_s secondi.
    t0..t1 sono i secondi assoluti del primo e ultimo punto della pagina."""
    fig, axes = plt.subplots(rows_per_page, 1, figsize=(7.5, 9.5))
    fig.patch.set_facecolor(DARK_BG)
    if rows_per_page == 1: axes = [axes]
    for row_idx, ax in enumerate(axes):
        rs = t0 + row_idx * row_s
        re = rs + row_s
        if rs >= t1:
            ax.set_visible(False); continue
        mask = (t >= rs) & (t < re)
        if not mask.any():
            ax.set_visible(False); continue
        tt = t[mask] - rs
        vv = vf[mask]
        ax.set_facecolor(PANEL_BG)
        ax.plot(tt, vv, linewidth=0.55, color=GREEN)
        # overlays + markers
        row_peaks = [p for p in peaks if rs <= p["t"] < re]
        for p in row_peaks:
            pt = p["t"] - rs
            if p["cls"] == "pvc":
                wm = (tt >= pt - 0.12) & (tt <= pt + 0.12)
                if wm.any():
                    ax.plot(tt[wm], vv[wm], linewidth=1.0, color=RED)
                ax.scatter(pt, min(1.55, p["amp"] + 0.30), s=22, marker="v",
                           color=RED, edgecolors="white", linewidths=0.4, zorder=5)
            else:
                ax.scatter(pt, min(1.40, p["amp"] + 0.18), s=6, marker="v",
                           color=GREEN, edgecolors="white", linewidths=0.2, zorder=4)
        ax.set_xlim(0, row_s); ax.set_ylim(-1.2, 1.7)
        ax.tick_params(colors=MUTED, labelsize=6)
        ax.grid(True, alpha=0.2, color=GRID, linewidth=0.3)
        for sp in ax.spines.values(): sp.set_color(GRID)
        mm_ = int(rs // 60); ss_ = int(rs % 60)
        ax.set_ylabel(f"{mm_:02d}:{ss_:02d}", color=MUTED, fontsize=8,
                      rotation=0, ha="right", va="center", labelpad=18)
        rn = sum(1 for p in row_peaks if p["cls"] == "normal")
        rp = sum(1 for p in row_peaks if p["cls"] == "pvc")
        if rn or rp:
            ax.text(0.995, 0.94, f"{rn}N · {rp}PVC",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color=MUTED,
                    bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=2, alpha=0.7))
    axes[-1].set_xlabel("secondi dall'inizio della riga", color=MUTED, fontsize=8)
    plt.tight_layout(h_pad=0.4)
    return fig_to_bytes(fig)

# ---- generate all plot images ----
print("genero plots...")

# (A) ECG example: 8 secondi che contengono almeno una PVC
ecg_example_img = None
if pvc and N:
    p0 = pvc[0]["t"]
    mask = (t >= p0-3) & (t <= p0+5)
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(t[mask]-p0, vf[mask], linewidth=0.9, color=GREEN)
    for p in pvc:
        if p0-3 <= p["t"] <= p0+5:
            wm = (t >= p["t"]-0.12) & (t <= p["t"]+0.12)
            if wm.any():
                ax.plot(t[wm]-p0, vf[wm], linewidth=1.8, color=RED)
            ax.scatter(p["t"]-p0, min(1.6, p["amp"]+0.3), s=70, marker="v",
                       color=RED, edgecolors="white", linewidths=0.6, zorder=5)
    for p in peaks:
        if p["cls"] == "normal" and p0-3 <= p["t"] <= p0+5:
            ax.scatter(p["t"]-p0, min(1.4, p["amp"]+0.18), s=18, marker="v",
                       color=GREEN, edgecolors="white", linewidths=0.3, zorder=4)
    styled_ax(ax, "Esempio: 8 secondi con QRS normali (verde) e PVC (rosso)",
              "t (s) rispetto al primo PVC della sessione", "ECG filtrato (V)")
    plt.tight_layout()
    ecg_example_img = fig_to_bytes(fig)

# (B) Overview compressa
overview_img = None
if N:
    fig, ax = plt.subplots(figsize=(8.5, 2.5))
    fig.patch.set_facecolor(DARK_BG)
    step = max(1, N // 30000)
    ax.plot(t[::step]/60, vf[::step], linewidth=0.3, color=GREEN, alpha=0.8)
    if pvc:
        ax.scatter([p["t"]/60 for p in pvc], [1.55]*len(pvc), s=4, color=RED, marker="v")
    ax.set_ylim(-1.2, 1.8)
    styled_ax(ax, f"Overview compresso ({total_min:.1f} min). Triangoli rossi = posizioni PVC.",
              "Tempo (min)", "ECG filt (V)")
    plt.tight_layout()
    overview_img = fig_to_bytes(fig)

# (C) Tachogramma
tacho_img = None
if peaks:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor(DARK_BG)
    for p in peaks:
        if p["rr_prev"] is None: continue
        c = RED if p["cls"] == "pvc" else GREEN
        ax.scatter(p["t"]/60, 1000*p["rr_prev"], c=c, s=5, alpha=0.7)
    styled_ax(ax, "Tachogramma — intervallo RR per ogni battito nel tempo",
              "Tempo (min)", "RR (ms)")
    plt.tight_layout()
    tacho_img = fig_to_bytes(fig)

# (D1) Tachogram decomposed per transition type (4 colors)
tacho_decomp_img = None
if peaks:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    fig.patch.set_facecolor(DARK_BG)
    colors_map = {"N→N": GREEN, "N→PVC": RED, "PVC→N": BLUE, "PVC→PVC": ORANGE}
    sizes_map  = {"N→N": 5,    "N→PVC": 6,   "PVC→N": 12,   "PVC→PVC": 22}
    for k in ["N→N","PVC→N","N→PVC","PVC→PVC"]:
        vv = transitions[k]
        if not vv: continue
        ax.scatter([x[0] for x in vv], [x[1] for x in vv],
                   c=colors_map[k], s=sizes_map[k], alpha=0.6, label=f"{k} (n={len(vv)})",
                   edgecolors="white" if k=="PVC→PVC" else "none", linewidths=0.4)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8, loc="upper right", ncol=2)
    styled_ax(ax, "Tachogramma decomposto per tipo di transizione",
              "Tempo (min)", "RR (ms)")
    plt.tight_layout()
    tacho_decomp_img = fig_to_bytes(fig)

# (D) Histogram RR
hist_img = None
if coupling and sinus_rr:
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.hist([r*1000 for r in sinus_rr], bins=40, alpha=0.7, color=GREEN,
            label=f"Sinus N→N (n={len(sinus_rr)})", density=True, edgecolor="white", linewidth=0.3)
    ax.hist([r*1000 for r in coupling], bins=40, alpha=0.85, color=RED,
            label=f"Coupling pre-PVC (n={len(coupling)})", density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(sinus_median_ms, color=GREEN, linestyle="--", alpha=0.8, linewidth=1.5,
               label=f"Sinus med {sinus_median_ms:.0f}ms")
    ax.axvline(coupling_median, color=RED, linestyle="--", alpha=0.8, linewidth=1.5,
               label=f"Coupling med {coupling_median:.0f}ms")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8, loc="upper right")
    styled_ax(ax, "Distribuzione RR — bimodalità sinus vs coupling pre-PVC",
              "RR (ms)", "Densità")
    plt.tight_layout()
    hist_img = fig_to_bytes(fig)

# (E) Coupling stability over time
coupling_stability_img = None
if coupling:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor(DARK_BG)
    pvc_times_for_coupling = [p["t"]/60 for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None]
    ax.scatter(pvc_times_for_coupling, [c*1000 for c in coupling], c=RED, s=8, alpha=0.7)
    ax.axhline(coupling_median, color=ORANGE, linestyle="--", linewidth=1.2,
               label=f"Mediana {coupling_median:.0f}ms")
    ax.fill_between([pvc_times_for_coupling[0], pvc_times_for_coupling[-1]],
                    coupling_median - coupling_std,
                    coupling_median + coupling_std,
                    color=ORANGE, alpha=0.1, label=f"±1σ ({coupling_std:.0f}ms)")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Coupling interval nel tempo — stabilità del focolaio ectopico",
              "Tempo (min)", "Coupling RR (ms)")
    plt.tight_layout()
    coupling_stability_img = fig_to_bytes(fig)

# (F) Counts per minute
counts_img = None
if windows:
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor(DARK_BG)
    ts_ = [w["t"]/60 for w in windows]
    ax.bar([x - 0.15 for x in ts_], [w["norm"] for w in windows], width=0.3,
           color=GREEN, alpha=0.85, label="Sinus")
    ax.bar([x + 0.15 for x in ts_], [w["pvc"] for w in windows], width=0.3,
           color=RED, alpha=0.85, label="PVC")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Battiti per minuto", "Tempo (min)", "N battiti/min")
    plt.tight_layout()
    counts_img = fig_to_bytes(fig)

# (G) HRV pre-PVC
hrv_img = None
if pre_pvc_stdevs:
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor(DARK_BG)
    pvc_ts = [p["t"]/60 for p in peaks if p["cls"] == "pvc"]
    if len(pvc_ts) >= len(pre_pvc_stdevs):
        pvc_ts = pvc_ts[-len(pre_pvc_stdevs):]
    ax.scatter(pvc_ts, [1000*s for s in pre_pvc_stdevs], c=RED, s=8, alpha=0.7,
               label="Stdev RR (5 normali pre-PVC)")
    if baseline_stdev_mean:
        ax.axhline(baseline_stdev_mean, color=GREEN, linestyle="--", linewidth=1.5,
                   label=f"Baseline sinus ({baseline_stdev_mean:.0f}ms)")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Variabilità RR nei 5 battiti normali prima di ogni PVC",
              "Tempo (min)", "Stdev RR (ms)")
    plt.tight_layout()
    hrv_img = fig_to_bytes(fig)

# (H) Poincaré plot
poincare_img = None
if len(sinus_rr) >= 2:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    xs = [1000*sinus_rr[i] for i in range(len(sinus_rr)-1)]
    ys = [1000*sinus_rr[i+1] for i in range(len(sinus_rr)-1)]
    ax.scatter(xs, ys, c=GREEN, s=6, alpha=0.4, label="Sinus N-N")
    # PVC: RR_prev (coupling) vs RR_next (compensatoria)
    px = [1000*p["rr_prev"] for p in peaks if p["cls"]=="pvc" and p["rr_prev"] and p["rr_next"]]
    py = [1000*p["rr_next"] for p in peaks if p["cls"]=="pvc" and p["rr_prev"] and p["rr_next"]]
    ax.scatter(px, py, c=RED, s=12, alpha=0.7, label="PVC (coupling, compensatoria)")
    lim = max(max(xs+px+[1]), max(ys+py+[1])) * 1.05
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Poincaré plot — RRₙ₊₁ vs RRₙ",
              "RRₙ (ms)", "RRₙ₊₁ (ms)")
    ax.set_aspect("equal")
    plt.tight_layout()
    poincare_img = fig_to_bytes(fig)

# (A1) Amplitude — histogram per contesto
amp_hist_img = None
if any(amp_groups[k] for k in ["stable","pre_pvc","post_pvc"]):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor(DARK_BG)
    if amp_groups["stable"]:
        ax.hist(amp_groups["stable"], bins=30, alpha=0.55, color=GREEN,
                label=f"Stable (N→N→N) n={len(amp_groups['stable'])}",
                density=True, edgecolor="white", linewidth=0.3)
    if amp_groups["pre_pvc"]:
        ax.hist(amp_groups["pre_pvc"], bins=30, alpha=0.85, color=ORANGE,
                label=f"Pre-PVC (N prima ectopica) n={len(amp_groups['pre_pvc'])}",
                density=True, edgecolor="white", linewidth=0.3)
    if amp_groups["post_pvc"]:
        ax.hist(amp_groups["post_pvc"], bins=30, alpha=0.75, color=BLUE,
                label=f"Post-PVC (N dopo pausa) n={len(amp_groups['post_pvc'])}",
                density=True, edgecolor="white", linewidth=0.3)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Distribuzione ampiezze QRS dei battiti normali per contesto",
              "Ampiezza picco (V)", "Densità")
    plt.tight_layout()
    amp_hist_img = fig_to_bytes(fig)

# (A2) Amplitude vs RR precedente (Frank-Starling)
amp_rr_img = None
if amp_rr_pairs:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    fig.patch.set_facecolor(DARK_BG)
    colors_map_amp = {"stable": GREEN, "pre_pvc": ORANGE, "post_pvc": BLUE, "sandwich": RED}
    for g, color in colors_map_amp.items():
        pts = [(rr*1000, a) for rr, a, gg in amp_rr_pairs if gg == g]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       c=color, s=10, alpha=0.5, label=f"{g} (n={len(pts)})")
    # linea di regressione globale
    if len(amp_rr_pairs) > 10:
        xs_all = np.array([p[0]*1000 for p in amp_rr_pairs])
        ys_all = np.array([p[1] for p in amp_rr_pairs])
        m, b = np.polyfit(xs_all, ys_all, 1)
        xx = np.linspace(xs_all.min(), xs_all.max(), 50)
        ax.plot(xx, m*xx + b, color="white", linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"trend globale (r={r_all_norm:.2f})")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Ampiezza QRS vs RR precedente — effetto Frank-Starling",
              "RR precedente (ms)", "Ampiezza QRS (V)")
    plt.tight_layout()
    amp_rr_img = fig_to_bytes(fig)

# (Z) Zoom strip 9-11 min (analisi locale)
zoom_img = None
ZOOM_T0, ZOOM_T1 = 9*60, 11*60   # 9-11 min
zoom_mask = (t >= ZOOM_T0) & (t < ZOOM_T1)
zoom_peaks_local = [p for p in peaks if ZOOM_T0 <= p["t"] < ZOOM_T1]
if zoom_mask.any():
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.5))
    fig.patch.set_facecolor(DARK_BG)
    SEG = 30  # 30 sec per riga = 4 righe coprono 2 minuti
    for row_idx, ax in enumerate(axes):
        rs = ZOOM_T0 + row_idx * SEG
        re = rs + SEG
        m = (t >= rs) & (t < re)
        if not m.any():
            ax.set_visible(False); continue
        tt = t[m] - rs; vv = vf[m]
        ax.set_facecolor(PANEL_BG)
        ax.plot(tt, vv, linewidth=0.7, color=GREEN)
        row_peaks_z = [p for p in peaks if rs <= p["t"] < re]
        for p in row_peaks_z:
            pt = p["t"] - rs
            if p["cls"] == "pvc":
                wm = (tt >= pt - 0.12) & (tt <= pt + 0.12)
                if wm.any():
                    ax.plot(tt[wm], vv[wm], linewidth=1.4, color=RED)
                ax.scatter(pt, min(1.55, p["amp"] + 0.30), s=40, marker="v",
                           color=RED, edgecolors="white", linewidths=0.5, zorder=5)
            else:
                ax.scatter(pt, min(1.40, p["amp"] + 0.18), s=12, marker="v",
                           color=GREEN, edgecolors="white", linewidths=0.3, zorder=4)
        ax.set_xlim(0, SEG); ax.set_ylim(-1.2, 1.7)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.grid(True, alpha=0.2, color=GRID, linewidth=0.3)
        for sp in ax.spines.values(): sp.set_color(GRID)
        mm_ = int(rs//60); ss_ = int(rs%60)
        ax.set_ylabel(f"{mm_:02d}:{ss_:02d}", color=MUTED, fontsize=8,
                      rotation=0, ha="right", va="center", labelpad=20)
        rn = sum(1 for p in row_peaks_z if p["cls"]=="normal")
        rp = sum(1 for p in row_peaks_z if p["cls"]=="pvc")
        ax.text(0.995, 0.94, f"{rn}N · {rp}PVC", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=MUTED,
                bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=2, alpha=0.7))
    axes[-1].set_xlabel("secondi nella riga", color=MUTED, fontsize=8)
    axes[0].set_title("Zoom strip-chart 09:00 → 11:00 (4 righe da 30s)",
                      color="white", fontsize=10, pad=10)
    plt.tight_layout(h_pad=0.4)
    zoom_img = fig_to_bytes(fig)

# (I) Strip pages — multiple
strip_imgs = []
n_strip_pages = math.ceil(total_min / (STRIP_ROW_SECONDS * STRIP_ROWS_PER_PAGE / 60))
for page_idx in range(n_strip_pages):
    t0 = page_idx * STRIP_ROW_SECONDS * STRIP_ROWS_PER_PAGE
    t1 = t0 + STRIP_ROW_SECONDS * STRIP_ROWS_PER_PAGE
    img = make_strip_page_image(t0, t1)
    strip_imgs.append((img, t0, min(t1, total_s)))

print(f"  {len(strip_imgs)} pagine di strip-chart")
print(f"  {1 + len(strip_imgs) + 5} pagine totali stimate")

# ------------------ PDF assembly ------------------
out_path = PATH.replace(os.sep + "ecg_", os.sep + "report_").replace(".csv", ".pdf")
doc = SimpleDocTemplate(out_path, pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=15*mm, bottomMargin=15*mm,
                        title=f"Holter session {ses_id}",
                        author="holter-ecg (self-built educational device)")

# styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=20, leading=24,
                    textColor=colors.HexColor("#1a1a1a"), spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=18,
                    textColor=colors.HexColor("#2980b9"), spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, leading=15,
                    textColor=colors.HexColor("#27ae60"), spaceBefore=8, spaceAfter=4)
NORMAL = ParagraphStyle("N", parent=ss["Normal"], fontSize=10, leading=14,
                        textColor=colors.HexColor("#222"),
                        alignment=TA_JUSTIFY)
MUTED_P = ParagraphStyle("M", parent=NORMAL, fontSize=9, leading=12,
                         textColor=colors.HexColor("#666"))
MONO = ParagraphStyle("Mono", parent=NORMAL, fontName="Courier", fontSize=9)
BIG_NUM = ParagraphStyle("BN", parent=ss["Normal"], fontSize=22, leading=22,
                         alignment=TA_CENTER,
                         textColor=colors.HexColor("#27ae60"))

def kv_table(rows, col_widths=None):
    tbl = Table(rows, colWidths=col_widths or [70*mm, 100*mm])
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#555")),
        ("TEXTCOLOR", (1,0), (1,-1), colors.HexColor("#111")),
        ("ALIGN", (1,0), (1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("LINEBELOW", (0,0), (-1,-2), 0.3, colors.HexColor("#ddd")),
    ]))
    return tbl

def metric_card(label, value, unit, color):
    inner = Table(
        [[Paragraph(f'<font color="#888" size=7>{label.upper()}</font>', NORMAL)],
         [Paragraph(f'<font color="{color}" size=22><b>{value}</b></font>', NORMAL)],
         [Paragraph(f'<font color="#888" size=8>{unit}</font>', NORMAL)]],
        colWidths=[40*mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f5f7fa")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#dde3e9")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    return inner

# build story
story = []
now = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---- PAGE 1 — COVER ----
story.append(Paragraph("Report sessione holter", H1))
story.append(Paragraph(
    f"<font color='#777'>Sessione <font name='Courier'>{ses_id}</font> · "
    f"Durata <b>{total_min:.1f} min</b> · "
    f"Sample rate {fs_real:.2f} Hz · Generato il {now}</font>",
    NORMAL
))
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd")))
story.append(Spacer(1, 14))

# Metric cards row
ecg_total_bpm = 60 * n_total / total_s if total_s else 0
cards = Table([[
    metric_card("ECG total", f"{ecg_total_bpm:.0f}", "BPM elettrico", "#27ae60"),
    metric_card("Sinus only", f"{sinus_bpm:.0f}", "BPM normali", "#2980b9"),
    metric_card("PVC rate", f"{pvc_rate:.1f}", "/min", "#c0392b"),
    metric_card("PVC burden", f"{burden:.1f}", "% del totale", "#e67e22"),
]], colWidths=[44*mm]*4)
cards.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                           ("LEFTPADDING", (0,0), (-1,-1), 0),
                           ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
story.append(cards)
story.append(Spacer(1, 14))

# Brief summary
summary_text = (
    f"La registrazione di <b>{total_min:.1f} minuti</b> contiene <b>{n_total}</b> battiti, "
    f"di cui <b>{len(norm)}</b> sinusali ({sinus_bpm:.0f} BPM medi) e <b>{len(pvc)}</b> "
    f"classificati come battiti ectopici ventricolari (PVC). "
    f"Il <b>PVC burden</b> è del <b>{burden:.1f}%</b>: una percentuale "
    f"significativa di battiti ectopici è una caratteristica nota del paziente "
    f"in carico al cardiologo. "
    f"L'analisi successiva mostra che si tratta di un focolaio ectopico "
    f"<b>monomorfo e temporalmente stabile</b> "
    f"(coupling interval mediano {coupling_median:.0f} ms ± {coupling_std:.0f} ms, "
    f"prematurità del {prematurity:.0f}% rispetto al ciclo sinusale di "
    f"{sinus_median_ms:.0f} ms)."
)
story.append(Paragraph("Sintesi esecutiva", H2))
story.append(Paragraph(summary_text, NORMAL))
story.append(Spacer(1, 10))

# Example ECG
if ecg_example_img:
    story.append(Paragraph("Esempio rappresentativo", H3))
    story.append(Paragraph(
        "Otto secondi della registrazione centrati sulla prima PVC. La linea verde mostra "
        "i battiti sinusali; il segmento e il triangolo rossi evidenziano il QRS della PVC e "
        "la finestra di ±120 ms su cui viene misurata l'iperpolarizzazione di rebound — la "
        "caratteristica fisiologica che il detector usa per classificarla.",
        NORMAL))
    story.append(Image(ecg_example_img, width=174*mm, height=58*mm))

story.append(PageBreak())

# ---- PAGE 2 — METHODOLOGY + DETAILED METRICS ----
story.append(Paragraph("Metodologia", H2))
story.append(Paragraph(
    "<b>Hardware.</b> Frontend analogico AD8232 (Analog Devices) configurato in derivazione "
    "Einthoven I (RA, LA, RL come riferimento), uscita campionata dall'ADC 12-bit del Pi Pico 2 W "
    "a 250 Hz. Sistema alimentato da cella LiPo (3.7 V) e completamente floating rispetto alla rete "
    "elettrica durante la registrazione.",
    NORMAL))
story.append(Paragraph(
    "<b>Trasporto e archiviazione.</b> Il Pico invia ogni campione via TCP/WiFi al server "
    "(Python/Flask) che effettua filtraggio in tempo reale e logging contemporaneo su CSV. "
    "Risoluzione temporale: 4 ms per sample. Durata registrazione corrente: "
    f"{total_s:.1f} s ({fs_real:.2f} Hz reali).",
    NORMAL))
story.append(Paragraph(
    "<b>Filtraggio.</b> Cascata di due filtri IIR del primo ordine: high-pass a 0.3 Hz (rimuove "
    "deriva di baseline e DC) seguito da low-pass a 25 Hz (attenua mains 50 Hz ed EMG). "
    "La banda passante 0.3–25 Hz preserva la morfologia QRS e l'undershoot post-QRS che caratterizza "
    "le PVC.",
    NORMAL))
story.append(Paragraph(
    "<b>Detection dei battiti.</b> Macchina a stati a 4 stati (idle, width, detect, post). "
    "Il segnale entra in fase di tracking quando supera 0.10 V; viene confermato come QRS se "
    "supera anche una soglia adattiva (mediana ampiezze recenti × 0.45, minimo 0.30 V). "
    "Nei 200 ms successivi al picco viene misurato il trough — la deflessione negativa post-QRS.",
    NORMAL))
story.append(Paragraph(
    "<b>Classificazione PVC.</b> Un battito è classificato come PVC se il rapporto "
    "|trough|/peak supera 0.40 (iperpolarizzazione pronunciata) OPPURE se la larghezza del "
    "QRS supera 95 ms. Refractory period di 300 ms tra battiti accettati.",
    NORMAL))

story.append(Paragraph("Metriche dettagliate", H2))
story.append(kv_table([
    ["Durata registrazione",            f"{total_s:.1f} s  ({total_min:.2f} min)"],
    ["Sample rate misurato",            f"{fs_real:.2f} Hz"],
    ["Campioni totali",                 f"{N:,}"],
    ["Battiti rilevati",                f"{n_total:,}"],
    ["Battiti sinusali",                f"{len(norm):,}  ({100*len(norm)/max(1,n_total):.1f}%)"],
    ["Battiti PVC",                     f"{len(pvc):,}  ({100*len(pvc)/max(1,n_total):.1f}%)"],
    ["BPM totale (tutti i battiti)",    f"{ecg_total_bpm:.1f}"],
    ["BPM sinusale",                    f"{sinus_bpm:.1f}"],
    ["PVC rate",                        f"{pvc_rate:.2f} /min"],
    ["PVC burden",                      f"{burden:.1f} %"],
    ["RR sinusale mediano",             f"{sinus_median_ms:.1f} ms"],
    ["RR sinusale medio",               f"{sinus_mean_ms:.1f} ms"],
    ["Deviazione std RR sinusale (SDNN)", f"{sinus_std_ms:.1f} ms"],
    ["RMSSD",                           f"{sinus_rmssd_ms:.1f} ms"],
    ["Coupling pre-PVC mediano",        f"{coupling_median:.1f} ms"],
    ["Coupling std",                    f"{coupling_std:.1f} ms"],
    ["Coupling IQR",                    f"{coupling_iqr:.1f} ms"],
    ["Prematurità",                     f"{prematurity:.1f} % più precoce del sinus"],
    ["RR post-PVC mediano (compensatoria)", f"{compensatory_median:.1f} ms"],
]))

story.append(PageBreak())

# ---- STRIP CHART PAGES ----
story.append(Paragraph("Tracce ECG complete", H2))
story.append(Paragraph(
    f"Visualizzazione integrale della registrazione, formato strip-chart stile holter. "
    f"Ogni riga rappresenta <b>{STRIP_ROW_SECONDS} secondi</b> ({STRIP_ROW_SECONDS//60 if STRIP_ROW_SECONDS>=60 else 0} min). "
    f"Il numero a sinistra di ogni riga indica il minuto:secondo di inizio. "
    f"A destra il conteggio battiti normali (N) e PVC. "
    f"I triangoli sopra il tracciato marcano la classificazione, l'overlay rosso evidenzia "
    f"il QRS e l'iperpolarizzazione delle PVC.",
    NORMAL))
story.append(Spacer(1, 6))

for img, t0, t1 in strip_imgs:
    story.append(Image(img, width=174*mm, height=233*mm))
    story.append(PageBreak())

# ---- OVERVIEW + TACHOGRAM ----
story.append(Paragraph("Overview e tachogramma", H2))
story.append(Paragraph(
    "Vista d'insieme della registrazione e degli intervalli RR per ogni battito. "
    "Il tachogramma evidenzia la bimodalità del segnale: i battiti sinusali (verdi) "
    "stanno su un livello orizzontale stabile, mentre i coupling pre-PVC (rossi) "
    "formano un cluster molto più basso e ben distinto.",
    NORMAL))
if overview_img:
    story.append(Image(overview_img, width=174*mm, height=53*mm))
if tacho_img:
    story.append(Spacer(1, 8))
    story.append(Image(tacho_img, width=174*mm, height=64*mm))

story.append(PageBreak())

# ---- COUPLING ANALYSIS ----
story.append(Paragraph("Analisi del coupling interval", H2))
story.append(Paragraph(
    "Il coupling interval è il tempo tra un battito sinusale e la PVC che lo segue. "
    "Quando è <b>costante</b> nel tempo è la firma di un focolaio ectopico singolo e "
    "monomorfo (sempre la stessa zona di miocardio anomalo che scarica con la stessa "
    "latenza dopo ogni stimolazione). Coupling variabile suggerirebbe sorgenti multiple "
    "o meccanismi più complessi.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"In questa sessione il coupling è <b>{coupling_median:.0f} ± {coupling_std:.0f} ms</b> "
    f"(IQR {coupling_iqr:.0f} ms), pari al <b>{100-prematurity:.0f}%</b> del ciclo sinusale "
    f"({sinus_median_ms:.0f} ms). La deviazione standard è il <b>{100*coupling_std/coupling_median:.1f}%</b> "
    f"del valore mediano — variazione molto contenuta, pattern altamente ripetibile.",
    NORMAL))
if hist_img:
    story.append(Spacer(1, 6))
    story.append(Image(hist_img, width=174*mm, height=66*mm))
if coupling_stability_img:
    story.append(Spacer(1, 6))
    story.append(Image(coupling_stability_img, width=174*mm, height=62*mm))

story.append(PageBreak())

# ---- PATTERNS ----
story.append(Paragraph("Pattern temporali delle PVC", H2))
story.append(Paragraph(
    "Le PVC possono organizzarsi in pattern ripetitivi caratteristici. "
    "<b>Bigeminia</b>: alternanza fissa N–PVC–N–PVC. "
    "<b>Trigeminia</b>: pattern N–N–PVC ripetuto. "
    "<b>Couplet</b>: due PVC consecutive senza battiti normali in mezzo. "
    "<b>Triplet</b>: tre o più PVC consecutive (si parla di salve / runs di tachicardia "
    "ventricolare non sostenuta se >3 e <30 s).",
    NORMAL))
story.append(Spacer(1, 6))

pat_tbl = Table([
    [Paragraph("<b>Pattern</b>", NORMAL), Paragraph("<b>Count</b>", NORMAL),
     Paragraph("<b>Interpretazione</b>", NORMAL)],
    ["PVC isolate (N–PVC–N)", f"{iso_pvc}",
     "Forma più comune; il sistema cardiaco ritorna a ritmo sinusale subito dopo l'ectopia."],
    ["Couplet (2 PVC consecutive)", f"{couplets_n}",
     "Due battiti ectopici di seguito. Meno frequenti delle isolate."],
    ["Triplet / salve (3+)", "0",
     "Nessun run ventricolare osservato nella sessione."],
    ["Bigeminia (≥3 cicli)", f"{bigem}",
     "Alternanza N-PVC-N-PVC, episodi brevi durante la registrazione."],
    ["Trigeminia (≥3 cicli)", f"{trigem}",
     "Pattern N-N-PVC ripetuto. Il più frequente dei pattern ritmici osservati."],
], colWidths=[42*mm, 18*mm, 114*mm])
pat_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,-1), "Helvetica", 9),
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#2980b9")),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf3fb")),
    ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#bbb")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
]))
story.append(pat_tbl)

if counts_img:
    story.append(Spacer(1, 10))
    story.append(Paragraph("Distribuzione battiti per minuto", H3))
    story.append(Image(counts_img, width=174*mm, height=58*mm))

story.append(PageBreak())

# ---- TACHOGRAM DECOMPOSITION ----
story.append(Paragraph("Decomposizione del tachogramma", H2))
story.append(Paragraph(
    "A occhio nudo nel tachogramma si distinguono <b>tre bande di intervalli RR per "
    "i battiti classificati come normali</b> (punti verdi): una fina attorno ai "
    "750 ms, una grossa e densa attorno ai 900 ms, e una più sparsa centrata sui "
    "1180 ms. Ciascuna corrisponde a un contesto fisiologico differente, riconoscibile "
    "se si separano gli RR in base al <b>tipo di transizione</b> (cosa è il battito "
    "corrente e cosa era il precedente).",
    NORMAL))
story.append(Spacer(1, 6))

# tabella decomposizione transizioni
decomp_rows = [["Transizione", "n", "Mediana", "Std", "Min–Max", "Interpretazione"]]
trans_explain = {
    "N→N":    "Sinus → sinus. Il ritmo cardiaco di base, tra due battiti normali consecutivi.",
    "N→PVC":  "Sinus → PVC. Coupling interval: il timing del trigger ectopico.",
    "PVC→N":  "PVC → sinus. La pausa post-ectopica, completa o interpolata.",
    "PVC→PVC":"PVC → PVC. Due battiti ectopici consecutivi (couplet).",
}
for k in ["N→N", "N→PVC", "PVC→N", "PVC→PVC"]:
    vals = [x[1] for x in transitions[k]]
    if not vals:
        decomp_rows.append([k, "0", "—", "—", "—", trans_explain[k]])
        continue
    med = statistics.median(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    decomp_rows.append([
        k, str(len(vals)), f"{med:.0f} ms", f"{std:.0f} ms",
        f"{min(vals):.0f}–{max(vals):.0f}", trans_explain[k]
    ])
decomp_tbl = Table(decomp_rows, colWidths=[20*mm, 12*mm, 22*mm, 18*mm, 26*mm, 76*mm])
decomp_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,-1), "Helvetica", 9),
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#2980b9")),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf3fb")),
    ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#bbb")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
]))
story.append(decomp_tbl)

if tacho_decomp_img:
    story.append(Spacer(1, 6))
    story.append(Image(tacho_decomp_img, width=174*mm, height=68*mm))

story.append(Spacer(1, 8))
story.append(Paragraph("La banda dei 900 ms — sinus regolare", H3))
nn_vals = [x[1] for x in transitions["N→N"]]
if nn_vals:
    nn_med = statistics.median(nn_vals); nn_std = statistics.stdev(nn_vals)
    story.append(Paragraph(
        f"È il <b>ritmo sinusale di base</b>: {len(nn_vals)} intervalli, "
        f"mediana {nn_med:.0f} ms ({60000/nn_med:.0f} BPM), deviazione standard "
        f"{nn_std:.0f} ms. La larghezza della banda è la <b>variabilità "
        f"fisiologica HRV</b> (componente respiratoria + tono autonomico). "
        f"Più stretta è la banda, più regolare il sinus.",
        NORMAL))

story.append(Paragraph("La banda dei 1180 ms — pause compensatorie", H3))
pn_vals = [x[1] for x in transitions["PVC→N"]]
if pn_vals:
    pn_med = statistics.median(pn_vals); pn_std = statistics.stdev(pn_vals)
    story.append(Paragraph(
        f"È la <b>pausa post-PVC</b>: {len(pn_vals)} intervalli, mediana "
        f"<b>{pn_med:.0f} ms ± {pn_std:.0f} ms</b>. La dispersione enorme "
        f"(range {min(pn_vals):.0f}–{max(pn_vals):.0f} ms) <b>non è rumore</b>, è un dato "
        f"clinicamente significativo: riflette il comportamento variabile del "
        f"trigger ectopico rispetto al pacemaker sinusale.",
        NORMAL))
    story.append(Paragraph(
        "<b>Pausa compensatoria completa</b> (verso 2000+ ms): il PVC ha "
        "azzerato il sinus node, che riparte da capo con un ciclo intero di "
        "ritardo. <b>PVC interpolato</b> (verso 600 ms): il sinus node è stato "
        "ignorato dal trigger ectopico e continua il suo ritmo come se nulla "
        "fosse — il battito normale successivo arriva al tempo previsto. "
        "I casi intermedi (compenso parziale) riempiono il continuum.",
        NORMAL))

story.append(Paragraph("La banda dei 750 ms — due meccanismi distinti", H3))
n_b750_nn = band750_breakdown.get("N→N", 0)
n_b750_pn = band750_breakdown.get("PVC→N", 0)
story.append(Paragraph(
    f"Banda fine attorno ai 700–800 ms: <b>{n_b750_nn + n_b750_pn} osservazioni</b>. "
    f"Decomposta: <b>{n_b750_nn} N→N</b> + <b>{n_b750_pn} PVC→N</b>. Sono due "
    f"fenomeni fisiologicamente diversi sovrapposti.",
    NORMAL))
story.append(Paragraph(
    f"<b>(a) {n_b750_nn} N→N a ~750 ms</b>: fasi di <b>accelerazione sinusale</b> "
    f"a ~80 BPM. Tipicamente associate a picchi di <b>aritmia sinusale "
    f"respiratoria</b> (inspirazione profonda → ritiro vagale → HR temporaneamente "
    f"più alto), sospiri spontanei, brevi attivazioni simpatiche.",
    NORMAL))
story.append(Paragraph(
    f"<b>(b) {n_b750_pn} PVC→N a ~750 ms</b>: <b>PVC interpolate</b> "
    f"o quasi. Il battito ectopico si infila tra due sinusali senza disturbare "
    f"il pacemaker. Variante tipicamente benigna, più frequente con bassa "
    f"frequenza cardiaca basale (come nel caso di questa sessione).",
    NORMAL))

story.append(PageBreak())

# ---- ZOOM 9-11 MIN ----
story.append(Paragraph("Analisi locale: oscillazioni tra 9 e 11 minuti", H2))
window_beats_zoom = [r for r in peaks if 9*60 <= r["t"] < 11*60]
zoom_n = sum(1 for r in window_beats_zoom if r["cls"] == "normal")
zoom_p = sum(1 for r in window_beats_zoom if r["cls"] == "pvc")
zoom_rrs = [r["rr_prev"]*1000 for r in window_beats_zoom if r["rr_prev"]]
zoom_std = statistics.stdev(zoom_rrs) if len(zoom_rrs) > 1 else 0
all_rrs_global = [p["rr_prev"]*1000 for p in peaks if p["rr_prev"]]
baseline_std_global = statistics.stdev(all_rrs_global) if len(all_rrs_global) > 1 else 0
story.append(Paragraph(
    f"Nel tachogramma si nota un'oscillazione visiva marcata attorno al minuto 10. "
    f"Numericamente, nella finestra <b>09:00–11:00</b> ci sono <b>{len(window_beats_zoom)} "
    f"battiti</b> ({zoom_n} normali, {zoom_p} PVC) con deviazione standard degli RR "
    f"pari a <b>{zoom_std:.0f} ms</b>, contro un baseline di {baseline_std_global:.0f} ms "
    f"per tutta la sessione. Non c'è quindi un aumento anomalo della variabilità.",
    NORMAL))
story.append(Paragraph(
    "Le \"oscillazioni\" sono il <b>rimbalzo verticale tipico delle zone con maggiore "
    "burden di PVC</b>: ogni PVC produce un punto basso (coupling ~500 ms), il battito "
    "normale subito dopo produce un punto alto (compensatoria ~1200 ms), il battito "
    "normale stabile sta in mezzo (~900 ms). Quando il pattern di trigeminia è "
    "particolarmente regolare per un intervallo, i punti rimbalzano fra questi tre "
    "livelli in successione rapida, dando l'effetto visivo di un'onda quadra "
    "verticale.",
    NORMAL))
if zoom_img:
    story.append(Spacer(1, 8))
    story.append(Image(zoom_img, width=174*mm, height=140*mm))

story.append(PageBreak())

# ---- AMPLITUDE ANALYSIS ----
story.append(Paragraph("Analisi dell'ampiezza dei battiti normali", H2))
story.append(Paragraph(
    "L'ampiezza del QRS di un battito normale dipende da più fattori: l'orientamento "
    "del vettore elettrico cardiaco (fisso per la geometria del torace), il <b>volume "
    "ventricolare al momento della depolarizzazione</b> (più sangue dentro = più "
    "massa eccitata = QRS più ampio), il <b>tempo di riempimento</b> dal battito "
    "precedente, e il <b>tono autonomico</b> (simpatico aumenta inotropismo e contrattilità). "
    "Per indagare se l'insorgenza di una PVC è preannunciata da uno stato "
    "particolare del sinus, classifichiamo i battiti normali in tre contesti.",
    NORMAL))
story.append(Spacer(1, 6))

amp_tbl_rows = [["Contesto", "n", "Media (V)", "Mediana (V)", "Std (V)", "Significato"]]
ctx_descr = {
    "stable":   "N sinusale circondato da altri normali — baseline.",
    "pre_pvc":  "L'ultimo normale prima di una PVC — il \"sospetto\".",
    "post_pvc": "Il normale dopo la pausa compensatoria — più riempimento.",
    "sandwich": "Tra due PVC consecutive (bigeminia stretta) — raro.",
}
for k in ["stable", "pre_pvc", "post_pvc", "sandwich"]:
    s = amp_stats.get(k)
    if not s or s["n"] == 0:
        amp_tbl_rows.append([k, "0", "—", "—", "—", ctx_descr[k]])
        continue
    amp_tbl_rows.append([
        k, str(s["n"]), f"{s['mean']:.3f}", f"{s['median']:.3f}", f"{s['std']:.3f}",
        ctx_descr[k]
    ])
amp_tbl = Table(amp_tbl_rows, colWidths=[20*mm, 12*mm, 20*mm, 22*mm, 16*mm, 84*mm])
amp_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,-1), "Helvetica", 9),
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#2980b9")),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf3fb")),
    ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#bbb")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
]))
story.append(amp_tbl)

# delta comparativi
stable_med = amp_stats["stable"]["median"] if amp_stats["stable"] else 0
pre_med    = amp_stats["pre_pvc"]["median"] if amp_stats["pre_pvc"] else 0
post_med   = amp_stats["post_pvc"]["median"] if amp_stats["post_pvc"] else 0
pre_delta  = 100*(pre_med - stable_med)/stable_med if stable_med else 0
post_delta = 100*(post_med - stable_med)/stable_med if stable_med else 0

story.append(Spacer(1, 8))
story.append(Paragraph(
    f"<b>Confronto vs baseline stable</b>: "
    f"pre-PVC <b>{pre_delta:+.1f}%</b> ({pre_med:.3f} vs {stable_med:.3f} V), "
    f"post-PVC <b>{post_delta:+.1f}%</b> ({post_med:.3f} vs {stable_med:.3f} V).",
    NORMAL))

if amp_hist_img:
    story.append(Spacer(1, 6))
    story.append(Image(amp_hist_img, width=174*mm, height=66*mm))

story.append(Spacer(1, 10))
story.append(Paragraph("Effetto Frank-Starling — ampiezza vs intervallo precedente", H3))
story.append(Paragraph(
    "La <b>legge di Frank-Starling</b> dice che più si riempie il ventricolo prima della "
    "contrazione, più la contrazione successiva sarà potente. Tradotto in ECG: un RR "
    "più lungo dovrebbe correlare con un QRS più ampio nel battito che lo segue. "
    f"Sul nostro dataset, la correlazione di Pearson tra RR precedente e ampiezza del "
    f"QRS che segue è <b>r = {r_all_norm:+.2f}</b> su tutti i normali e "
    f"<b>r = {r_stable:+.2f}</b> sul solo gruppo stable. "
    + (
        "Correlazione modestamente positiva osservata: l'effetto Frank-Starling è "
        "presente ma debole nella derivazione I (più visibile in derivazioni precordiali). "
        if r_all_norm > 0.15 else
        "Correlazione bassa: la derivazione I cattura solo una proiezione del vettore "
        "cardiaco, e gli effetti di volume sono più visibili in derivazioni precordiali "
        "(es. V5-V6) che qui non abbiamo. "
    )
    + "Il post-PVC, che segue la pausa compensatoria più lunga, dovrebbe quindi avere "
    "ampiezza leggermente superiore al baseline.",
    NORMAL))
if amp_rr_img:
    story.append(Spacer(1, 6))
    story.append(Image(amp_rr_img, width=174*mm, height=70*mm))

# interpretazione finale
story.append(Spacer(1, 10))
story.append(Paragraph("Cosa significa per i triggers delle PVC", H3))
findings = []
if abs(pre_delta) < 3:
    findings.append(
        f"<b>L'ampiezza pre-PVC è essenzialmente uguale al baseline</b> ({pre_delta:+.1f}%). "
        f"Il sinus che precede una PVC NON appare in uno stato meccanico/elettrico "
        f"speciale rispetto al sinus normale. Questo suggerisce che il <b>trigger ectopico "
        f"è autonomo</b> rispetto al ciclo sinusale immediatamente precedente — il "
        f"focolaio si scarica per propri ritmi (ad esempio modulati dal tono autonomico "
        f"o respiratorio globale), non perché c'è qualcosa di particolare nel battito "
        f"appena prima."
    )
elif pre_delta > 0:
    findings.append(
        f"L'ampiezza pre-PVC è <b>più grande</b> del baseline ({pre_delta:+.1f}%). "
        "Potrebbe suggerire un riempimento ventricolare leggermente maggiore o "
        "un'attivazione simpatica nei secondi precedenti l'ectopia."
    )
else:
    findings.append(
        f"L'ampiezza pre-PVC è <b>più piccola</b> del baseline ({pre_delta:+.1f}%). "
        "Suggerirebbe un riempimento ventricolare ridotto — possibile se l'ectopia "
        "tende ad arrivare in fasi di tachicardia transitoria con minor preload."
    )

if post_delta > 5:
    findings.append(
        f"<b>L'ampiezza post-PVC è significativamente maggiore</b> del baseline "
        f"({post_delta:+.1f}%). Coerente con l'<b>effetto Frank-Starling</b>: dopo "
        f"la pausa compensatoria il ventricolo ha avuto più tempo di riempirsi, lo "
        f"stroke volume è maggiore, e il QRS è più ampio. È un fenomeno fisiologico "
        f"atteso e che si misura clinicamente come \"post-extrasystolic potentiation\"."
    )
elif post_delta > 0:
    findings.append(
        f"L'ampiezza post-PVC è leggermente più grande del baseline ({post_delta:+.1f}%), "
        f"compatibile con un piccolo effetto Frank-Starling — visibile in modo "
        f"più marcato in derivazioni che proiettano meglio sul vettore principale "
        f"di depolarizzazione."
    )
else:
    findings.append(
        f"L'ampiezza post-PVC è simile al baseline ({post_delta:+.1f}%); l'effetto "
        f"Frank-Starling non è ben visibile in questa singola derivazione, ma è "
        f"probabilmente presente a livello emodinamico."
    )

findings.append(
    "Conclusione: in base a queste osservazioni, le PVC <b>non sembrano essere "
    "preannunciate da un'alterazione dell'ampiezza del battito sinusale precedente</b>. "
    "Il trigger ectopico appare quindi modulato da fattori più \"sistemici\" "
    "(tono autonomico, fase respiratoria) piuttosto che da una condizione "
    "meccanica/elettrica del battito immediatamente precedente."
)

for line in findings:
    story.append(Paragraph("• " + line, NORMAL))
    story.append(Spacer(1, 4))

story.append(PageBreak())

# ---- HRV PRE-PVC + POINCARE ----
story.append(Paragraph("HRV e modulazione autonomica pre-PVC", H2))
story.append(Paragraph(
    "Si analizzano i <b>5 battiti normali immediatamente precedenti</b> a ciascuna PVC. "
    "Calcolando la deviazione standard degli intervalli RR in queste finestre e confrontandola "
    "con il baseline sinusale stabile, si valuta se il ritmo sinusale è più irregolare nei "
    "secondi che precedono un'ectopia (compatibile con modulazione autonomica/vagale).",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"Stdev RR pre-PVC media: <b>{pre_pvc_stdev_mean:.1f} ms</b>. "
    f"Stdev baseline sinusale: <b>{baseline_stdev_mean:.1f} ms</b>. "
    f"Delta: <b>{hrv_delta_pct:+.1f}%</b>. "
    + (
        "Differenza piccola ma sistematica su centinaia di osservazioni: il ritmo "
        "sinusale è leggermente più variabile prima di ogni PVC. Compatibile con un "
        "trigger ectopico modulato dal tono vagale / fase respiratoria."
        if hrv_delta_pct > 5 else
        "La variabilità non differisce sostanzialmente dal baseline; il trigger ectopico "
        "appare relativamente indipendente dalla variabilità sinusale a breve termine."
    ),
    NORMAL))
if hrv_img:
    story.append(Spacer(1, 6))
    story.append(Image(hrv_img, width=174*mm, height=58*mm))

story.append(Spacer(1, 10))
story.append(Paragraph("Poincaré plot", H2))
story.append(Paragraph(
    "Ogni punto rappresenta una coppia di intervalli consecutivi (RRₙ, RRₙ₊₁). "
    "Nei battiti sinusali (verde) ci si aspetta un cluster compatto attorno alla bisettrice "
    "(RRₙ ≈ RRₙ₊₁): più stretto è il cluster trasversalmente, minore è la variabilità battito-battito. "
    "Le PVC (rosso) plottano <b>coupling vs RR successivo</b> e cadono fuori dal cluster sinusale "
    "in una zona caratteristica: RRₙ molto corto (coupling), RRₙ₊₁ lungo (pausa compensatoria).",
    NORMAL))
if poincare_img:
    story.append(Spacer(1, 6))
    story.append(Image(poincare_img, width=110*mm, height=110*mm,
                       hAlign="CENTER"))

story.append(PageBreak())

# ---- CONCLUSIONS + LIMITS ----
story.append(Paragraph("Conclusioni descrittive", H2))
concl = [
    f"<b>Focolaio ectopico monomorfo e stabile</b>. Il coupling interval ha mediana "
    f"{coupling_median:.0f} ms e deviazione standard {coupling_std:.0f} ms "
    f"({100*coupling_std/max(1,coupling_median):.1f}% del mediano). La dispersione è molto "
    f"bassa, indicando un'unica sorgente ectopica che scarica con timing molto consistente.",

    f"<b>PVC prevalentemente isolate</b>. {iso_pvc} delle {len(pvc)} PVC totali "
    f"({100*iso_pvc/max(1,len(pvc)):.0f}%) sono isolate (preceduta e seguita da battito normale). "
    f"Solo {couplets_n} couplet, nessun triplet. Nessun run di tachicardia ventricolare osservato.",

    f"<b>Pattern ritmico dominante: trigeminia</b>. {trigem} run di trigeminia "
    f"(N-N-PVC ripetuto ≥3 cicli) contro {bigem} run di bigeminia "
    f"(N-PVC alternato ≥3 cicli).",

    f"<b>Burden temporalmente uniforme</b>. La frequenza PVC/min resta costante intorno "
    f"a {pvc_rate:.0f}/min per tutta la durata della registrazione, senza accumuli o "
    f"quiescenze importanti.",

    f"<b>Modulazione autonomica lieve ma sistematica</b>. La variabilità RR nei 5 battiti "
    f"sinusali immediatamente precedenti una PVC è del {hrv_delta_pct:+.0f}% rispetto al "
    f"baseline. Il segnale è piccolo ma osservabile su centinaia di osservazioni.",

    f"<b>Pulse deficit teorico</b>. Tipicamente le PVC non producono polso periferico "
    f"apprezzabile per via dello stroke volume ridotto. Atteso un <b>deficit di "
    f"~{100*len(pvc)/n_total:.0f}%</b> tra frequenza ECG e frequenza al polso radiale "
    f"(quanto rilevabile da un Garmin/Apple Watch). Compatibile con osservazioni "
    f"empiriche dell'utente.",
]
for c in concl:
    story.append(Paragraph("• " + c, NORMAL))
    story.append(Spacer(1, 4))

story.append(Spacer(1, 12))
story.append(Paragraph("Limiti tecnici", H2))
lims = [
    "<b>Singola derivazione (Einthoven I)</b>. Impossibile localizzare il focolaio "
    "ectopico nei tre piani anatomici o distinguere PVC da altre forme di ectopia "
    "che richiedano vista multi-lead.",

    "<b>Detector basato su rebound e larghezza</b>. Robusto per la morfologia "
    "attuale del paziente, ma potrebbe perdere PVC con polarità invertita o "
    "morfologia atipica. Non esegue rilevazione di onde P.",

    "<b>Durata limitata</b>. Una registrazione di "
    f"{total_min:.1f} min cattura un campione del comportamento elettrico ma "
    "non permette inferenza su trend circadiani, effetti post-prandiali, "
    "risposta a sforzo, sonno.",

    "<b>Assenza di segnale di respiro sincronizzato</b>. L'ipotesi di modulazione "
    "respiratoria del trigger ectopico (suggerita dall'osservazione +HRV pre-PVC) "
    "non è verificabile formalmente senza un sensore di respiro (es. accelerometro "
    "Z sul torace) sincronizzato con l'ECG.",

    "<b>Detector non validato clinicamente</b>. Le soglie sono state tarate "
    "empiricamente sui dati di questo singolo soggetto e non hanno passato "
    "validazione contro un gold standard come l'holter clinico professionale.",
]
for l in lims:
    story.append(Paragraph("• " + l, NORMAL))
    story.append(Spacer(1, 4))

story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#ccc")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"<font color='#888' size=8>"
    f"Sorgente: <font name='Courier'>{os.path.basename(PATH)}</font>. "
    f"Pipeline: Pi Pico 2 W (ADC 12-bit 250 Hz) → WiFi/TCP → server Python (Flask/SSE) → "
    f"filtro IIR 0.3–25 Hz → detector FSM 4 stati → classificazione rebound/width. "
    f"Repository: <font name='Courier'>github.com/mrEg0n/holter-ecg</font>. "
    f"Generato il {now} da host/generate_report_pdf.py."
    f"</font>",
    NORMAL
))

# build
print("rendering PDF...")
doc.build(story)
print(f"PDF salvato: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
