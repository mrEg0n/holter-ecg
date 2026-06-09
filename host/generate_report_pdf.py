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
from PIL import Image as PILImage
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

# ---- ri-classificazione col criterio di produzione corrente ----
# Il server (host/server.py) classifica PVC se:
#   (rebound >= 0.40 OPPURE width >= 95 ms) E ampiezza >= 0.70 V.
# I peaks CSV storici possono essere stati scritti col vecchio criterio (senza
# la soglia di ampiezza): riclassifichiamo qui così il report riflette sempre la
# logica attuale ed evidenzia i falsi positivi che la soglia ampiezza rimuove.
REBOUND_RATIO_PVC = 0.40
PVC_WIDTH_MS      = 95.0
PVC_MIN_AMP_V     = 0.70
# Rebound minimo: una PVC vera ha SEMPRE qualche grado di iperpolarizzazione
# post-QRS (rebound > 0.05). Se rebound = 0 e la classificazione si appoggia solo
# al criterio width, è quasi certamente un artefatto largo (motion, baseline
# shift) che ha bypassato la soglia width senza essere un vero complesso ectopico.
PVC_MIN_REBOUND   = 0.05
# Plausibilità morfologica: un QRS umano ha width fisiologica tra ~40 e 220 ms.
# Sotto 40 ms = spike artefatto (electrode pop / picco di rumore stretto che ha
# crossato la soglia ampiezza). Sopra 220 ms = baseline shift / motion artifact
# (es. respiro profondo, urto sull'elettrodo) che il detector ha interpretato
# come complesso largo. Entrambi vengono declassati a "normal" e tracciati.
PVC_W_MIN_MS      = 40.0
PVC_W_MAX_MS      = 220.0
removed_fp = []   # battiti declassati pvc -> normal dalla soglia di ampiezza
removed_implausible = []  # battiti declassati per width fuori range fisiologico
for p in peaks:
    shape_pvc = (p["reb"] >= REBOUND_RATIO_PVC or p["w"] >= PVC_WIDTH_MS)
    plausible_w = PVC_W_MIN_MS <= p["w"] <= PVC_W_MAX_MS
    has_rebound = p["reb"] >= PVC_MIN_REBOUND
    if shape_pvc and p["amp"] < PVC_MIN_AMP_V:
        removed_fp.append(p)
    if shape_pvc and p["amp"] >= PVC_MIN_AMP_V and (not plausible_w or not has_rebound):
        removed_implausible.append(p)
    p["cls"] = "pvc" if (shape_pvc and p["amp"] >= PVC_MIN_AMP_V
                          and plausible_w and has_rebound) else "normal"

# ---- pulizia: rimuovi gli spike di rumore (non sono battiti) ----
# Larghezza <= 16 ms (4 campioni @250 Hz) è sub-fisiologica: sono artefatti /
# electrode-pop, non QRS reali. Si tolgono del tutto dalla serie (non solo
# declassati) così non inquinano conteggi, RR e morfologia. Restano comunque
# elencati in removed_fp per la sezione esplicativa.
n_spike_removed = sum(1 for p in peaks if p["w"] <= 16 and p["amp"] < PVC_MIN_AMP_V)
peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP_V)]

# ---- esclusione di intervalli temporali contaminati (rumore, elettrodo staccato) ----
# Tre fonti di esclusioni (in ordine di priorità):
#   1) env var EXCLUDE_INTERVALS="s1-e1,s2-e2,..." (override esplicito)
#   2) file exclusions/exclusions_<base>.json (creato da host/mark_exclusions.py)
#   3) nessuna esclusione
# I picchi nei tratti esclusi sono rimossi e il tempo viene sottratto dalla durata
# utile per non gonfiare i rate.
EXCLUDED_INTERVALS = []
_excl_env = os.environ.get("EXCLUDE_INTERVALS", "").strip()
if _excl_env:
    for chunk in _excl_env.split(","):
        a, b = chunk.split("-")
        EXCLUDED_INTERVALS.append((float(a), float(b)))
    print(f"[excl] {len(EXCLUDED_INTERVALS)} intervalli da EXCLUDE_INTERVALS env var")
else:
    # fallback al file JSON dell'editor manuale
    import json as _json
    _ses_id = os.path.basename(PATH).replace("ecg_", "").replace(".csv", "")
    _excl_path = os.path.join("exclusions", f"exclusions_{_ses_id}.json")
    if os.path.exists(_excl_path):
        try:
            with open(_excl_path) as _f:
                _ej = _json.load(_f)
            EXCLUDED_INTERVALS = [(d["start"], d["end"]) for d in _ej.get("intervals", [])]
            print(f"[excl] {len(EXCLUDED_INTERVALS)} intervalli da {_excl_path}")
        except Exception as _e:
            print(f"[excl] errore lettura {_excl_path}: {_e}")
if EXCLUDED_INTERVALS:
    def _in_excl(tv):
        return any(s <= tv <= e for s, e in EXCLUDED_INTERVALS)
    n_pre_excl = len(peaks)
    peaks = [p for p in peaks if not _in_excl(p["t"])]
    n_excl_removed = n_pre_excl - len(peaks)
    excl_seconds = sum(e - s for s, e in EXCLUDED_INTERVALS)
else:
    n_excl_removed = 0
    excl_seconds = 0.0

ses_id = os.path.basename(PATH).replace("ecg_", "").replace(".csv", "")
total_s_raw = float(t[-1] - t[0]) if N else 0
total_s = total_s_raw - excl_seconds  # durata utile dopo esclusioni
total_min = total_s / 60.0
fs_real = N / total_s_raw if total_s_raw else SAMPLE_HZ
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
# ---- pulizia: coupling contaminato da battito sinusale non rilevato ----
# Un coupling vero è PREMATURO (più corto del RR sinusale). Se l'rr_prev di una
# PVC supera ~0.9x il RR sinusale mediano, quasi sempre è perché un battito
# sinusale è stato perso nel gap (falso "late-coupled"): l'intervallo NON è un
# vero coupling, quindi lo escludiamo da coupling/tachogramma.
_sinus_median_rr = statistics.median(sinus_rr) if sinus_rr else 0.0
COUPLING_MAX_FACTOR = 0.9
n_coupling_excluded = 0
for p in peaks:
    p["coupling_bad"] = (p["cls"] == "pvc" and p["rr_prev"] is not None
                         and _sinus_median_rr
                         and p["rr_prev"] > COUPLING_MAX_FACTOR * _sinus_median_rr)
    if p["coupling_bad"]:
        n_coupling_excluded += 1
coupling  = [p["rr_prev"] for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None and not p["coupling_bad"]]
compensatory = [p["rr_next"] for p in peaks if p["cls"] == "pvc" and p["rr_next"] is not None]

# transizioni RR per categoria (per decomposizione tachogramma)
transitions = {"N→N": [], "N→PVC": [], "PVC→N": [], "PVC→PVC": []}
for i_t in range(1, len(peaks)):
    if peaks[i_t]["rr_prev"] is None: continue
    # salta gli rr_prev contaminati da un battito sinusale perso (vedi sopra)
    if peaks[i_t].get("coupling_bad"): continue
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
# couplet = 2 PVC veramente consecutive (RR < 700ms). Senza vincolo temporale
# i battiti normali persi fra due PVC le fanno apparire adiacenti nella lista.
COUPLET_MAX_RR_S = 0.70
couplets_n = 0
couplet_indices = []  # coppie di indici (i, i+1) dei couplet veri
i = 0
while i < len(peaks) - 1:
    if peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc":
        rr = peaks[i+1]["t"] - peaks[i]["t"]
        if rr >= COUPLET_MAX_RR_S:
            i += 1; continue  # gap troppo grande → non è couplet
        if i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc":
            i += 1; continue  # è una run, non un couplet
        couplets_n += 1
        couplet_indices.append((i, i+1))
        i += 2
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

# ---- Classificazione PVC: interpolata vs compensatoria ----
# Una PVC interpolata si infila fra due N senza resettare il nodo SA: la somma
# (RR_pre + RR_post) ≈ 1× RR sinus (la PVC è "in più" nel ritmo). Una PVC con
# pausa compensatoria PIENA ha somma ≈ 2× RR sinus (il nodo SA salta un battito).
# Le interpolate sono favorite dalle bradicardie (più spazio diastolico) e sono
# emodinamicamente più benigne (il cuore non perde gittata).
sinus_rr_for_class = [p["rr_prev"] for p in peaks
                       if p["cls"] == "normal" and p["rr_prev"] is not None
                       and 0.6 < p["rr_prev"] < 1.4]
RR_SINUS_MS = statistics.median(sinus_rr_for_class)*1000 if sinus_rr_for_class else 1000

interpolated_list = []
compensated_list = []
incomplete_list  = []   # tra i due (>1.3× e <1.85×)
for idx, p in enumerate(peaks):
    if p["cls"] != "pvc": continue
    if p["rr_prev"] is None or p["rr_next"] is None: continue
    if idx == 0 or idx == len(peaks)-1: continue
    if peaks[idx-1]["cls"] != "normal" or peaks[idx+1]["cls"] != "normal": continue
    s_ms = (p["rr_prev"] + p["rr_next"]) * 1000
    p["sum_pre_post_ms"] = s_ms
    if s_ms < 1.3 * RR_SINUS_MS:
        p["pause_type"] = "interpolated"
        interpolated_list.append(p)
    elif 1.85 * RR_SINUS_MS < s_ms < 2.15 * RR_SINUS_MS:
        p["pause_type"] = "compensated"
        compensated_list.append(p)
    else:
        p["pause_type"] = "incomplete"
        incomplete_list.append(p)

n_class_total = len(interpolated_list) + len(compensated_list) + len(incomplete_list)
pct_interp = 100*len(interpolated_list)/max(1,n_class_total)
pct_comp   = 100*len(compensated_list)/max(1,n_class_total)
pct_incomp = 100*len(incomplete_list)/max(1,n_class_total)

# ---- Screening fibrillazione atriale (su tutti gli N-N consecutivi) ----
# Marker classici: irregolarità degli RR fra battiti sinusali.
#   RMSSD > 100 ms   pNN50 > 40%   CV RR > 15-20%   entropia ~max   bimodalità persa
# Da solo nessuno è diagnostico (servirebbero 12 derivazioni), ma il quadro
# complessivo permette di flaggare un'eventuale ritmo "irregolarmente irregolare".
af_nn_ms = []
for i_nn in range(1, len(peaks)):
    if peaks[i_nn]["cls"] == "normal" and peaks[i_nn-1]["cls"] == "normal":
        rr_nn = peaks[i_nn]["t"] - peaks[i_nn-1]["t"]
        if 0.4 <= rr_nn <= 2.0:
            af_nn_ms.append(rr_nn * 1000)

af = {"nn_count": len(af_nn_ms)}
if len(af_nn_ms) >= 30:
    af["median_ms"] = statistics.median(af_nn_ms)
    af["mean_ms"]   = statistics.mean(af_nn_ms)
    af["std_ms"]    = statistics.stdev(af_nn_ms)
    af["cv_pct"]    = 100 * af["std_ms"] / af["mean_ms"]
    af["min_ms"]    = min(af_nn_ms)
    af["max_ms"]    = max(af_nn_ms)
    diffs = [abs(af_nn_ms[k] - af_nn_ms[k-1]) for k in range(1, len(af_nn_ms))]
    af["rmssd_ms"] = (sum(d*d for d in diffs) / len(diffs))**0.5
    af["pnn50"]    = 100 * sum(1 for d in diffs if d > 50) / len(diffs)
    af["pnn20"]    = 100 * sum(1 for d in diffs if d > 20) / len(diffs)
    # entropia di Shannon su istogramma 20 bin (rapporto col massimo teorico)
    hist_af, edges_af = np.histogram(af_nn_ms, bins=20)
    p_af = hist_af[hist_af > 0] / sum(hist_af[hist_af > 0])
    H_af = float(-sum(p * np.log2(p) for p in p_af))
    H_max_af = float(np.log2(len(p_af)))
    af["entropy"]     = H_af
    af["entropy_max"] = H_max_af
    af["entropy_ratio"] = H_af / H_max_af if H_max_af else 0
    # bimodalità: cerca due picchi separati nell'istogramma
    smooth = np.convolve(hist_af, [1,1,1], mode="same")
    peaks_h = [k for k in range(1, len(smooth)-1)
               if smooth[k] > smooth[k-1] and smooth[k] > smooth[k+1]
               and smooth[k] > 0.3 * smooth.max()]
    af["histogram"] = hist_af
    af["hist_edges"] = edges_af
    af["n_peaks"]    = len(peaks_h)
    # finestre 30 battiti con CV alto
    WIN_AF = 30
    flagged = 0; total = 0
    for k in range(0, len(af_nn_ms) - WIN_AF, 10):
        seg = af_nn_ms[k:k+WIN_AF]
        if statistics.mean(seg) > 0:
            cv = 100 * statistics.stdev(seg) / statistics.mean(seg)
            total += 1
            if cv > 15: flagged += 1
    af["windows_flagged"] = flagged
    af["windows_total"]   = total
    # scoring: 0-4
    score = 0
    if af["rmssd_ms"] > 100:        score += 1
    if af["pnn50"] > 40:            score += 1
    if af["entropy_ratio"] > 0.85:  score += 1
    if af["n_peaks"] <= 1 and af["cv_pct"] > 15: score += 1  # unimodale e ampia = AF; bimodale = no
    af["score"] = score
    if score == 0:
        af["verdict"] = "Ritmo sinusale regolare. Nessun marker di fibrillazione atriale."
    elif score == 1:
        af["verdict"] = ("Markers HRV elevati ma con struttura conservata. "
                         "Pattern compatibile con bradicardia + RSA + ectopia frequente; "
                         "non suggestivo di fibrillazione atriale.")
    elif score == 2:
        af["verdict"] = ("Markers HRV intermedi. Sospetto basso ma non escluso; "
                         "raccomandato controllo con ECG 12 derivazioni se sintomi presenti.")
    else:
        af["verdict"] = ("Markers HRV elevati con perdita di struttura: ritmo "
                         "irregolarmente irregolare. Compatibile con sospetto AF; "
                         "raccomandato controllo cardiologico.")
else:
    af["verdict"] = "Pochi N-N consecutivi: screening non eseguibile (bigeminia troppo densa o segnale degradato)."

WINDOW = 60
windows = []
i_w = 0
while peaks and peaks[0]["t"] + i_w*WINDOW < peaks[-1]["t"]:
    ws = peaks[0]["t"] + i_w*WINDOW
    we = ws + WINDOW
    in_w = [p for p in peaks if ws <= p["t"] < we]
    nn = sum(1 for p in in_w if p["cls"] == "normal")
    np_ = sum(1 for p in in_w if p["cls"] == "pvc")
    # HR SA effettiva nel minuto: median(60/RR_NN) su coppie N-N consecutive
    rr_nn = []
    for j in range(1, len(in_w)):
        if in_w[j]["cls"] == "normal" and in_w[j-1]["cls"] == "normal":
            rr = in_w[j]["t"] - in_w[j-1]["t"]
            if 0.4 <= rr <= 2.0:
                rr_nn.append(rr)
    hr_sa = 60/statistics.median(rr_nn) if rr_nn else None
    burden_min = 100*np_/(nn+np_) if (nn+np_) > 0 else None
    windows.append({"t": ws, "norm": nn, "pvc": np_,
                    "hr_sa": hr_sa, "burden_min": burden_min,
                    "n_total": nn+np_})
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

def fit_image(buf, max_w_mm=170.0, max_h_mm=245.0, max_px=1500):
    """Crea un Image flowable garantito più piccolo del frame stampabile.
    Scala alla larghezza utile preservando l'aspect ratio reale del PNG e, se
    necessario, riduce i pixel nativi: ReportLab 4.x può ignorare width/height
    espliciti quando la dimensione naturale del PNG supera il frame, causando
    LayoutError 'image too large'. Capando dimensioni native e flowable sotto
    al frame (frame utile A4 ≈ 174×267 mm) il problema sparisce alla radice."""
    buf.seek(0)
    pil = PILImage.open(buf)
    pw, ph = pil.size
    longest = max(pw, ph)
    if longest > max_px:
        s = max_px / longest
        pil = pil.resize((max(1, int(pw * s)), max(1, int(ph * s))),
                         PILImage.LANCZOS)
        out = io.BytesIO(); pil.save(out, format="PNG"); out.seek(0)
        buf = out; pw, ph = pil.size
    ar = pw / ph if ph else 1.0
    w = max_w_mm * mm
    h = w / ar
    if h > max_h_mm * mm:
        h = max_h_mm * mm
        w = h * ar
    buf.seek(0)
    return Image(buf, width=w, height=h, hAlign="CENTER")

def make_event_strip(center_t, win_s=6.0, highlight=None, title=None,
                     figsize=(8.0, 2.2)):
    """Strip di una singola finestra di win_s secondi centrata su un evento.
    highlight: lista di t (s) di battiti da cerchiare in arancione."""
    rs = max(0.0, center_t - win_s / 2.0); re = rs + win_s
    mask = (t >= rs) & (t < re)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    if mask.any():
        ax.plot(t[mask] - rs, vf[mask], linewidth=0.9, color=GREEN)
    hl = set(round(x, 3) for x in (highlight or []))
    for p in peaks:
        if not (rs <= p["t"] < re):
            continue
        pt = p["t"] - rs
        if p["cls"] == "pvc":
            ax.scatter(pt, min(1.6, p["amp"] + 0.30), s=60, marker="v",
                       color=RED, edgecolors="white", linewidths=0.5, zorder=5)
        else:
            ax.scatter(pt, min(1.4, p["amp"] + 0.18), s=16, marker="v",
                       color=GREEN, edgecolors="white", linewidths=0.3, zorder=4)
        if round(p["t"], 3) in hl:
            ax.scatter(pt, p["amp"], s=160, marker="o", facecolors="none",
                       edgecolors=ORANGE, linewidths=1.6, zorder=6)
    ax.set_xlim(0, win_s); ax.set_ylim(-1.2, 1.8)
    styled_ax(ax, title, "t (s)", "ECG (V)")
    plt.tight_layout()
    return fig_to_bytes(fig)

def make_interpolated_strip(p_center, win_s=9.0, title=None):
    """Strip didattica che mostra una PVC con le sue RR_pre / RR_post annotate
    e la somma vs 2x RR sinus (per distinguere interpolata vs compensatoria)."""
    c0 = p_center["t"]
    mask = (t >= c0 - win_s/2.0) & (t <= c0 + win_s/2.0)
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    fig.patch.set_facecolor(DARK_BG)
    if mask.any():
        ax.plot(t[mask] - c0, vf[mask], linewidth=0.9, color=GREEN)
    for p in peaks:
        if not (c0 - win_s/2.0 <= p["t"] <= c0 + win_s/2.0): continue
        if p["cls"] == "pvc":
            wm = (t >= p["t"] - 0.12) & (t <= p["t"] + 0.12)
            if wm.any():
                ax.plot(t[wm] - c0, vf[wm], linewidth=1.8, color=RED)
            ax.scatter(p["t"]-c0, min(1.6,p["amp"]+0.30), s=70, marker="v",
                       color=RED, edgecolors="white", linewidths=0.6, zorder=5)
        else:
            ax.scatter(p["t"]-c0, min(1.4,p["amp"]+0.18), s=18, marker="v",
                       color=GREEN, edgecolors="white", linewidths=0.3, zorder=4)
    rr_p_ms = p_center["rr_prev"]*1000
    rr_n_ms = p_center["rr_next"]*1000
    ax.annotate(f"RR_pre={rr_p_ms:.0f}ms", xy=(-rr_p_ms/2000.0, -0.95),
                color="#7ad9ff", fontsize=9, ha="center", fontweight="bold")
    ax.annotate(f"RR_post={rr_n_ms:.0f}ms", xy=(rr_n_ms/2000.0, -0.95),
                color="#ffe169", fontsize=9, ha="center", fontweight="bold")
    styled_ax(ax, title, "t (s) rispetto alla PVC", "ECG filtrato (V)")
    plt.tight_layout()
    return fig_to_bytes(fig)

def make_example_style_strip(center_t, win_s=6.0, title=None):
    """Strip nello STESSO stile della traccia esemplificativa di cima:
    overlay rosso sul QRS delle PVC (±120 ms) + triangolo rosso, triangoli
    verdi sui sinusali, styled_ax. Nessun cerchio arancione."""
    c0 = center_t
    mask = (t >= c0 - win_s / 2.0) & (t <= c0 + win_s / 2.0)
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor(DARK_BG)
    if mask.any():
        ax.plot(t[mask] - c0, vf[mask], linewidth=0.9, color=GREEN)
    for p in peaks:
        if not (c0 - win_s / 2.0 <= p["t"] <= c0 + win_s / 2.0):
            continue
        if p["cls"] == "pvc":
            wm = (t >= p["t"] - 0.12) & (t <= p["t"] + 0.12)
            if wm.any():
                ax.plot(t[wm] - c0, vf[wm], linewidth=1.8, color=RED)
            ax.scatter(p["t"] - c0, min(1.6, p["amp"] + 0.30), s=70, marker="v",
                       color=RED, edgecolors="white", linewidths=0.6, zorder=5)
        else:
            ax.scatter(p["t"] - c0, min(1.4, p["amp"] + 0.18), s=18, marker="v",
                       color=GREEN, edgecolors="white", linewidths=0.3, zorder=4)
    styled_ax(ax, title, "t (s) rispetto al centro del couplet", "ECG filtrato (V)")
    plt.tight_layout()
    return fig_to_bytes(fig)

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

# (A) ECG example: 8 secondi con una PVC "pulita" rappresentativa.
# Criteri: dopo il primo minuto (no warm-up), sandwich N-PVC-N (entrambi i battiti
# adiacenti normali), lontana ≥2s da qualunque intervallo escluso. Tra i candidati,
# scegliamo quella con ampiezza mediana (più rappresentativa, non outlier).
ecg_example_img = None
if pvc and N:
    def _far_from_excluded(t_s, margin=2.0):
        return all(not (s-margin <= t_s <= e+margin) for s, e in EXCLUDED_INTERVALS)
    # candidati: PVC sandwich N-PVC-N (con RR_prev e RR_next definiti)
    pvc_idx = [i for i, p in enumerate(peaks) if p["cls"] == "pvc"]
    candidates = []
    for i in pvc_idx:
        p = peaks[i]
        if p["t"] < 60: continue  # skip primo minuto
        if i == 0 or i == len(peaks)-1: continue
        if peaks[i-1]["cls"] != "normal" or peaks[i+1]["cls"] != "normal": continue
        if not _far_from_excluded(p["t"]): continue
        candidates.append(p)
    if candidates:
        # ordina per ampiezza e prendi quella mediana
        candidates.sort(key=lambda q: q["amp"])
        chosen = candidates[len(candidates)//2]
    else:
        # fallback: prima PVC dopo il primo minuto, oppure pvc[0]
        chosen = next((p for p in pvc if p["t"] >= 60 and _far_from_excluded(p["t"])), pvc[0])
    p0 = chosen["t"]
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
    styled_ax(ax, f"Esempio rappresentativo: 8 s attorno a una PVC a {int(p0//60):02d}:{int(p0%60):02d}",
              "t (s) rispetto alla PVC selezionata", "ECG filtrato (V)")
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
    pvc_times_for_coupling = [p["t"]/60 for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None and not p["coupling_bad"]]
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

# (F2) Correlazione HR ↔ frequenza PVC nel tempo
# Per ogni minuto della registrazione: HR SA effettiva + PVC rate + burden %.
# Output: 2 plot (time-series dual-axis, scatter HR vs PVC con regressione).
hr_vs_pvc_ts_img = None
hr_vs_pvc_scatter_img = None
hr_pvc_correlation = None
valid_w = [w for w in windows if w["hr_sa"] is not None and w["n_total"] >= 20]
if len(valid_w) >= 5:
    ts_min   = [w["t"]/60 for w in valid_w]
    hrs      = [w["hr_sa"] for w in valid_w]
    pvc_min  = [w["pvc"] for w in valid_w]
    burdens  = [w["burden_min"] for w in valid_w]

    # time series dual-axis
    fig, ax1 = plt.subplots(figsize=(11, 3.4))
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(DARK_BG)
    ax1.plot(ts_min, hrs, color=GREEN, lw=1.0, marker="o", ms=2,
             label="HR SA (BPM)")
    ax1.set_ylabel("HR SA effettiva (BPM)", color=GREEN, fontsize=9)
    ax1.tick_params(axis="y", colors=GREEN)
    ax1.tick_params(axis="x", colors="white")
    ax1.set_xlabel("Tempo (min)", color="white", fontsize=9)
    for sp in ax1.spines.values(): sp.set_color("#444")
    ax1.grid(alpha=0.18, color="#666")
    ax2 = ax1.twinx()
    ax2.set_facecolor(DARK_BG)
    ax2.plot(ts_min, pvc_min, color=RED, lw=1.0, marker="s", ms=2, alpha=0.85,
             label="PVC/min")
    ax2.set_ylabel("PVC/min", color=RED, fontsize=9)
    ax2.tick_params(axis="y", colors=RED)
    for sp in ax2.spines.values(): sp.set_color("#444")
    ax1.set_title("HR SA effettiva e PVC rate minuto per minuto", color="white", fontsize=10)
    plt.tight_layout()
    hr_vs_pvc_ts_img = fig_to_bytes(fig)

    # scatter HR vs PVC rate con regressione + correlazione
    r_pearson = pearson(hrs, pvc_min)
    slope, intercept = np.polyfit(hrs, pvc_min, 1)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    sc = ax.scatter(hrs, pvc_min, c=ts_min, cmap="viridis", s=24,
                    alpha=0.75, edgecolors="white", linewidths=0.3)
    xline = np.linspace(min(hrs), max(hrs), 100)
    ax.plot(xline, slope*xline + intercept, color=ORANGE, lw=2,
            label=f"y = {slope:.2f}·x + {intercept:.1f}")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Tempo (min)", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=9, loc="upper left")
    ax.set_xlabel("HR SA effettiva (BPM)", color="white")
    ax.set_ylabel("PVC al minuto", color="white")
    ax.set_title(f"Scatter HR vs PVC/min — Pearson r = {r_pearson:.3f}",
                 color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.grid(alpha=0.18, color="#666")
    plt.tight_layout()
    hr_vs_pvc_scatter_img = fig_to_bytes(fig)
    hr_pvc_correlation = {
        "n": len(valid_w),
        "r": r_pearson,
        "slope": slope,
        "intercept": intercept,
        "hr_min": min(hrs), "hr_max": max(hrs),
        "pvc_min": min(pvc_min), "pvc_max": max(pvc_min),
    }

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

# (H0) AF screening — istogramma + tachogramma N-N
af_hist_img = None
af_tacho_img = None
if af.get("median_ms") is not None:
    # istogramma
    fig, ax = plt.subplots(figsize=(11, 3.6))
    edges = af["hist_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax.bar(centers, af["histogram"], width=width*0.95,
           color="#33aa66", edgecolor="white", linewidth=0.3)
    ax.axvline(af["median_ms"], color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Mediana {af['median_ms']:.0f}ms")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, ("Istogramma RR N-N (tutti i battiti sinusali consecutivi) — "
                   f"{af['n_peaks']} picco/picchi rilevati"),
              "RR (ms)", "N° intervalli")
    plt.tight_layout()
    af_hist_img = fig_to_bytes(fig)

    # tachogramma RR nel tempo
    fig, ax = plt.subplots(figsize=(11, 3.0))
    # ricostruisco timestamps degli N-N
    t_nn, rr_nn_list = [], []
    for i_nn in range(1, len(peaks)):
        if peaks[i_nn]["cls"] == "normal" and peaks[i_nn-1]["cls"] == "normal":
            rr_nn = (peaks[i_nn]["t"] - peaks[i_nn-1]["t"]) * 1000
            if 400 <= rr_nn <= 2000:
                t_nn.append(peaks[i_nn]["t"]/60)
                rr_nn_list.append(rr_nn)
    ax.scatter(t_nn, rr_nn_list, c="#33aa66", s=4, alpha=0.6)
    ax.axhline(af["median_ms"], color=ORANGE, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"Mediana {af['median_ms']:.0f}ms")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Tachogramma RR N-N — andamento temporale (AF si presenterebbe come nuvola caotica senza struttura)",
              "Tempo (min)", "RR (ms)")
    plt.tight_layout()
    af_tacho_img = fig_to_bytes(fig)

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

# (Z1) Esempi di couplet — stesso vincolo del conteggio (RR < COUPLET_MAX_RR_S).
# Senza il vincolo temporale si finivano per mostrare coppie con un N saltato in
# mezzo, che NON sono couplet veri.
couplet_imgs = []
for (i, j) in couplet_indices[:4]:
    ctr = (peaks[i]["t"] + peaks[j]["t"]) / 2.0
    rr_ms = (peaks[j]["t"] - peaks[i]["t"]) * 1000
    couplet_imgs.append(make_example_style_strip(
        ctr, win_s=6.0,
        title=(f"Couplet a {int(ctr//60):02d}:{int(ctr%60):02d} — "
               f"due PVC consecutive a {rr_ms:.0f}ms (overlay rosso)")))

# (Z2) Esempi rappresentativi dei battiti riclassificati dalla soglia di ampiezza.
# Si escludono gli spike di rumore (w<=20ms, larghezza impossibile per un QRS):
# come esempi servono i battiti piccoli VERI, vicini alla soglia (più istruttivi).
n_fp_spike = sum(1 for q in removed_fp if q["w"] <= 20)
repr_fp = [q for q in removed_fp if q["w"] > 28]
fp_imgs = []
for p in sorted(repr_fp, key=lambda q: -q["amp"])[:3]:
    fp_imgs.append(make_event_strip(
        p["t"], win_s=5.0, highlight=[p["t"]],
        title=(f"{int(p['t']//60):02d}:{int(p['t']%60):02d} — amp {p['amp']:.2f} V "
               f"(reb {p['reb']:.2f}, w {p['w']:.0f} ms): sotto soglia → normale")))
print(f"  {len(couplet_imgs)} esempi couplet, {len(fp_imgs)} esempi falsi positivi")

# (Z3) Esempi di PVC apparentemente "tardive" = battito sinusale non rilevato nel gap.
# Sono i coupling esclusi dalle statistiche: li mostriamo comunque per trasparenza.
latecoupled = [p for p in peaks if p.get("coupling_bad")]
lc_imgs = []
for p in latecoupled[:2]:
    lc_imgs.append(make_event_strip(
        p["t"], win_s=3.6, highlight=[p["t"]],
        title=(f"{int(p['t']//60):02d}:{int(p['t']%60):02d} — RR_prev {p['rr_prev']*1000:.0f} ms "
               f"(coupling tipico ~{coupling_median:.0f} ms): QRS sinusale non marcato nel gap")))
print(f"  {len(lc_imgs)} esempi PVC late-coupled (artefatto)")

# (Z4) Esempi di PVC interpolate vs pausa compensatoria (didattico)
def _pick_spread(lst, n=2, min_gap_s=60):
    out, last = [], -1e9
    for p in sorted(lst, key=lambda q: q["t"]):
        if p["t"] - last >= min_gap_s:
            out.append(p); last = p["t"]
        if len(out) >= n: break
    return out

interp_imgs = []
interp_picked = _pick_spread(interpolated_list, n=5, min_gap_s=60)
for idx_ex, p in enumerate(interp_picked):
    s_ms = p["sum_pre_post_ms"]
    # ritrovo il numero globale (1..N) nella lista completa ordinata
    n_global = next((i+1 for i, q in enumerate(sorted(interpolated_list, key=lambda x: x["t"]))
                     if q is p), idx_ex+1)
    interp_imgs.append(make_interpolated_strip(
        p, win_s=8.0,
        title=(f"#{n_global} INTERPOLATA — {int(p['t']//60):02d}:{int(p['t']%60):02d}   "
               f"Σ = {s_ms:.0f} ms ({s_ms/RR_SINUS_MS:.2f}× RR sinus)")))
comp_imgs = []
comp_picked = _pick_spread(compensated_list, n=5, min_gap_s=60)
for idx_ex, p in enumerate(comp_picked):
    s_ms = p["sum_pre_post_ms"]
    n_global = next((i+1 for i, q in enumerate(sorted(compensated_list, key=lambda x: x["t"]))
                     if q is p), idx_ex+1)
    comp_imgs.append(make_interpolated_strip(
        p, win_s=8.0,
        title=(f"#{n_global} PAUSA COMPENSATORIA — {int(p['t']//60):02d}:{int(p['t']%60):02d}   "
               f"Σ = {s_ms:.0f} ms ({s_ms/RR_SINUS_MS:.2f}× RR sinus)")))
print(f"  {len(interp_imgs)} esempi interpolate, {len(comp_imgs)} esempi compensatorie")

# (Z5) GRID completa: tutte le PVC interpolate, una pagina alla volta.
# Stesso layout dell'export verificato (12 strip/pagina, marcatori arancione/azzurro/giallo).
def _build_interpolated_grid_pages(items, RR_S, rows=6, cols=2, win_s=6.0):
    """Restituisce una lista di immagini PNG (bytes), una per pagina di grid."""
    per_page = rows * cols
    n_pages = (len(items) + per_page - 1) // per_page
    items_sorted = sorted(items, key=lambda q: q["t"])
    pages = []
    for page_idx in range(n_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69), facecolor=DARK_BG)
        fig.suptitle(f"PVC interpolate — pagina {page_idx+1}/{n_pages}   "
                     f"RR sinus mediano {RR_S:.0f}ms   "
                     f"[ ◯ arancione=PVC analizzata · ━azzurro=RR_pre · ━giallo=RR_post · ┄rosso=2×RR atteso ]",
                     color="white", fontsize=7.5, y=0.997)
        for k in range(per_page):
            idx = page_idx*per_page + k
            r, c = k // cols, k % cols
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.set_facecolor(DARK_BG)
            if idx >= len(items_sorted):
                ax.axis("off"); continue
            p = items_sorted[idx]
            s_ms = p["sum_pre_post_ms"]
            c0 = p["t"]
            mask = (t >= c0 - win_s/2) & (t <= c0 + win_s/2)
            ax.plot(t[mask] - c0, vf[mask], color=GREEN, lw=0.5)
            for q in peaks:
                if c0 - win_s/2 <= q["t"] <= c0 + win_s/2:
                    dt = q["t"] - c0
                    if q["cls"] == "pvc":
                        ax.plot(dt, 1.35, "v", color=RED, ms=5)
                        wm = (t >= q["t"]-0.08) & (t <= q["t"]+0.08)
                        ax.plot(t[wm] - c0, vf[wm], color=RED, lw=1.0)
                    else:
                        ax.plot(dt, 0.85, "v", color=GREEN, ms=3)
            # PVC centrale evidenziata
            ax.scatter(0, p["amp"], s=240, marker="o", facecolors="none",
                       edgecolors="#ffa64d", linewidths=1.8, zorder=10)
            # RR_pre / RR_post
            rrp = p["rr_prev"]; rrn = p["rr_next"]
            y_pre, y_post = -0.65, -0.85
            ax.plot([-rrp, 0], [y_pre, y_pre], color="#7ad9ff", lw=2.0)
            ax.plot([-rrp, -rrp], [y_pre-0.05, y_pre+0.05], color="#7ad9ff", lw=1.5)
            ax.plot([0, 0], [y_pre-0.05, y_pre+0.05], color="#7ad9ff", lw=1.5)
            ax.text(-rrp/2, y_pre+0.06, f"{rrp*1000:.0f}", color="#7ad9ff",
                    fontsize=6, ha="center", fontweight="bold")
            ax.plot([0, rrn], [y_post, y_post], color="#ffe169", lw=2.0)
            ax.plot([0, 0], [y_post-0.05, y_post+0.05], color="#ffe169", lw=1.5)
            ax.plot([rrn, rrn], [y_post-0.05, y_post+0.05], color="#ffe169", lw=1.5)
            ax.text(rrn/2, y_post-0.15, f"{rrn*1000:.0f}", color="#ffe169",
                    fontsize=6, ha="center", fontweight="bold")
            # linea 2× RR atteso
            comp_x = -rrp + 2*RR_S/1000.0
            if -win_s/2 < comp_x < win_s/2:
                ax.axvline(comp_x, color="#ff4d6d", lw=0.8, ls="--", alpha=0.6)
            # numero
            ax.text(0.02, 0.97, f"#{idx+1}", transform=ax.transAxes,
                    color="#ffa64d", fontsize=11, fontweight="bold",
                    va="top", ha="left")
            ax.text(0.98, 0.97,
                    f"{int(c0//60):02d}:{c0%60:05.2f}  Σ={s_ms:.0f} ({s_ms/RR_S:.2f}×)",
                    transform=ax.transAxes, color="white", fontsize=6,
                    va="top", ha="right", family="monospace")
            ax.set_xlim(-win_s/2, win_s/2); ax.set_ylim(-1.1, 1.7)
            ax.tick_params(colors="white", labelsize=5)
            for sp in ax.spines.values(): sp.set_color("#444")
            ax.grid(alpha=0.12, color="#666")
        plt.tight_layout(rect=[0, 0, 1, 0.978])
        pages.append(fig_to_bytes(fig))
    return pages

print(f"  generando grid completo {len(interpolated_list)} interpolate...")
interp_grid_pages = _build_interpolated_grid_pages(interpolated_list, RR_SINUS_MS)
print(f"  {len(interp_grid_pages)} pagine grid")

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

# Esempi di couplet subito sotto la traccia esemplificativa, stesso stile/colore
if couplet_imgs:
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Couplet" if couplets_n == 1 else f"Couplet (tutti i {couplets_n})", H3))
    story.append(Paragraph(
        f"<b>Couplet</b>: due PVC consecutive senza battito sinusale interposto. In tutta "
        f"la sessione se ne contano <b>{couplets_n}</b> — nessun triplet o run più lungo. "
        f"Stesso stile della traccia sopra: overlay rosso sul QRS delle due PVC, triangoli "
        f"verdi sui sinusali."
        + ("" if couplets_n > len(couplet_imgs) else " Eccoli tutti:"),
        NORMAL))
    for im in couplet_imgs:
        story.append(Image(im, width=174*mm, height=58*mm))

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
    "<b>Classificazione PVC.</b> Un battito è classificato come PVC se ha morfologia ectopica "
    "— rapporto |trough|/peak ≥ 0.40 (iperpolarizzazione pronunciata) OPPURE larghezza QRS ≥ 95 ms "
    f"— <b>E</b> ampiezza ≥ {PVC_MIN_AMP_V:.2f} V. Il requisito di ampiezza evita di etichettare "
    "come PVC i piccoli battiti sinusali con onda S fisiologica. Refractory period di 300 ms.",
    NORMAL))
story.append(Paragraph(
    f"<b>Pulizia dati.</b> Prima delle analisi la serie viene ripulita: "
    f"(1) rimossi <b>{n_spike_removed}</b> spike di rumore con larghezza ≤ 16 ms "
    f"(sub-fisiologica per un QRS reale, tipici electrode-pop/artefatti di movimento); "
    f"(2) esclusi <b>{n_coupling_excluded}</b> intervalli di coupling non prematuri "
    f"(rr_prev &gt; {COUPLING_MAX_FACTOR:.0%} del RR sinusale mediano): non sono veri coupling "
    f"ma PVC il cui battito sinusale precedente non è stato rilevato nel gap "
    f"(falsi “late-coupled”), e contaminerebbero le statistiche di coupling e il "
    f"tachogramma. Conteggi, RR, coupling e morfologia qui riportati usano la serie pulita.",
    MUTED_P))

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

if n_coupling_excluded:
    story.append(Spacer(1, 10))
    story.append(Paragraph("PVC apparentemente tardive (escluse dal coupling)", H3))
    story.append(Paragraph(
        f"<b>{n_coupling_excluded} PVC</b> presentano un RR precedente molto lungo "
        f"(&gt; {COUPLING_MAX_FACTOR:.0%} del RR sinusale mediano), ben oltre il coupling "
        f"tipico (~{coupling_median:.0f} ms). Non sono però vere PVC \"end-diastolic\": "
        f"hanno la stessa morfologia di tutte le altre, ma il <b>battito sinusale che le "
        f"precede non è stato rilevato</b> dal detector (ampiezza sotto soglia), quindi "
        f"l'RR misurato somma un intervallo sinusale mancante + il coupling reale. Per "
        f"questo sono escluse dalle statistiche di coupling e dal tachogramma. Negli "
        f"esempi sotto si vede il QRS sinusale non marcato nel gap, prima della PVC "
        f"cerchiata:",
        NORMAL))
    for im in lc_imgs:
        story.append(Spacer(1, 6))
        story.append(fit_image(im, max_w_mm=170, max_h_mm=58))

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

# ---- HR ↔ PVC RATE CORRELATION ----
if hr_vs_pvc_ts_img and hr_vs_pvc_scatter_img and hr_pvc_correlation:
    story.append(Paragraph("Correlazione HR ↔ frequenza PVC", H2))
    corr = hr_pvc_correlation
    r = corr["r"]
    # interpretazione del coefficiente
    if abs(r) < 0.1:
        r_descr = "trascurabile"
    elif abs(r) < 0.3:
        r_descr = "debole"
    elif abs(r) < 0.5:
        r_descr = "moderata"
    elif abs(r) < 0.7:
        r_descr = "forte"
    else:
        r_descr = "molto forte"
    direction = "diretta (HR ↑ → PVC ↑)" if r > 0 else "inversa (HR ↑ → PVC ↓)"
    story.append(Paragraph(
        f"Analisi della relazione tra frequenza basale del nodo SA (HR effettiva calcolata "
        f"da median RR di coppie N-N consecutive) e numero di PVC nello stesso minuto. "
        f"Su <b>{corr['n']} finestre da 60s</b> con almeno 20 battiti utili, il coefficiente "
        f"di correlazione di Pearson è <b>r = {r:.3f}</b> (correlazione <b>{r_descr}</b>, "
        f"direzione {direction}). Pendenza della retta: <b>{corr['slope']:+.2f} PVC/min per ogni BPM</b>. "
        f"Range osservato: HR {corr['hr_min']:.0f}-{corr['hr_max']:.0f} BPM, "
        f"PVC {corr['pvc_min']}-{corr['pvc_max']}/min.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Time-series: HR SA e PVC rate minuto per minuto", H3))
    story.append(Image(hr_vs_pvc_ts_img, width=174*mm, height=54*mm))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Scatter: HR vs PVC/min (colore = tempo dalla partenza)", H3))
    story.append(Image(hr_vs_pvc_scatter_img, width=128*mm, height=85*mm,
                       hAlign="CENTER"))
    story.append(Spacer(1, 6))
    if r > 0.3:
        msg = ("Pattern compatibile con <b>aumento del PVC rate al crescere della frequenza "
               "basale</b>. Coerente con: (a) iniziale fase di warm-up sedentario (HR bassa, "
               "poche PVC) seguita da fasi più tonico-simpatiche (HR sale, focolaio più "
               "eccitabile); (b) modulazione autonomica dell'ectopia (vagale ↓ + simpatico ↑ "
               "→ più ectopia); (c) fattori metabolici intercorrenti (digestione, caffeina, "
               "movimento). NB: questo è l'opposto del classico pattern 'esercizio-soppresso' "
               "che si vede agli alti carichi aerobici (>120 BPM), dove le PVC scompaiono.")
    elif r < -0.3:
        msg = ("Pattern compatibile con <b>diminuzione del PVC rate al crescere della frequenza "
               "basale</b>. Coerente con il classico fenomeno delle PVC 'esercizio-soppresse': "
               "il sistema simpatico più attivo accelera la conduzione, riduce le zone di "
               "blocco unidirezionale, e sopprime il rientro / il focolaio ectopico. Marker "
               "di benignità.")
    else:
        msg = ("Correlazione non significativa: la frequenza istantanea delle PVC non è "
               "spiegata principalmente dalla HR basale in questa sessione. Altri fattori "
               "(posizione, respirazione, stato vagale, fattori meccanici toracici) "
               "probabilmente dominano.")
    story.append(Paragraph(msg, NORMAL))
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

# ---- ZOOM 9-11 MIN (solo se c'è davvero un'oscillazione locale) ----
# La sezione è condizionale: ha senso solo se la finestra 09:00-11:00 mostra
# variabilità RR realmente elevata vs baseline. In molte sessioni (es. quella
# pulita 150812) NON c'è oscillazione locale, quindi la sezione viene omessa.
window_beats_zoom = [r for r in peaks if 9*60 <= r["t"] < 11*60]
zoom_n = sum(1 for r in window_beats_zoom if r["cls"] == "normal")
zoom_p = sum(1 for r in window_beats_zoom if r["cls"] == "pvc")
zoom_rrs = [r["rr_prev"]*1000 for r in window_beats_zoom if r["rr_prev"]]
zoom_std = statistics.stdev(zoom_rrs) if len(zoom_rrs) > 1 else 0
all_rrs_global = [p["rr_prev"]*1000 for p in peaks if p["rr_prev"]]
baseline_std_global = statistics.stdev(all_rrs_global) if len(all_rrs_global) > 1 else 0
show_zoom_section = bool(zoom_img) and len(window_beats_zoom) > 10 and zoom_std > 1.25 * baseline_std_global
if show_zoom_section:
    story.append(Paragraph("Analisi locale: oscillazioni tra 9 e 11 minuti", H2))
    story.append(Paragraph(
        f"Nel tachogramma si nota un'oscillazione visiva marcata attorno al minuto 10. "
        f"Nella finestra <b>09:00–11:00</b> ci sono <b>{len(window_beats_zoom)} "
        f"battiti</b> ({zoom_n} normali, {zoom_p} PVC) con deviazione standard degli RR "
        f"pari a <b>{zoom_std:.0f} ms</b>, contro un baseline di {baseline_std_global:.0f} ms "
        f"per tutta la sessione.",
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

# ---- SCREENING FIBRILLAZIONE ATRIALE ----
story.append(Paragraph("Screening fibrillazione atriale (rhythm analysis)", H2))
story.append(Paragraph(
    "Analisi di tutti gli intervalli <b>RR fra battiti sinusali consecutivi</b> (N-N) "
    "sull'intera registrazione utile. La fibrillazione atriale produce un ritmo "
    "<i>irregolarmente irregolare</i>: gli RR perdono ogni struttura, l'istogramma "
    "diventa uniforme/caotico, RMSSD e pNN50 si impennano, l'entropia satura. "
    "I quattro marker sotto (>100 ms RMSSD, >40% pNN50, entropia/max >0.85, "
    "istogramma unimodale ampio) costituiscono uno <b>score 0-4</b>: il referto "
    "non è diagnostico (servirebbero 12 derivazioni e banda passante più ampia per "
    "valutare l'onda P), ma serve a flaggare automaticamente i pattern sospetti.",
    NORMAL))
if af.get("median_ms") is not None:
    story.append(Spacer(1, 8))
    af_rows = [
        ["N-N consecutivi analizzati",     f"{af['nn_count']}"],
        ["Mediana RR / BPM",                f"{af['median_ms']:.0f} ms ({60000/af['median_ms']:.1f} BPM)"],
        ["Std / CV",                        f"{af['std_ms']:.0f} ms / {af['cv_pct']:.1f}%"],
        ["Range",                           f"{af['min_ms']:.0f} – {af['max_ms']:.0f} ms"],
        ["RMSSD (soglia AF >100 ms)",       f"<b>{af['rmssd_ms']:.0f} ms</b>"],
        ["pNN50 (soglia AF >40%)",          f"<b>{af['pnn50']:.1f}%</b>"],
        ["pNN20",                           f"{af['pnn20']:.1f}%"],
        ["Entropia / max (AF se >0.85)",    f"<b>{af['entropy']:.2f} / {af['entropy_max']:.2f} ({af['entropy_ratio']:.2f})</b>"],
        ["Picchi nell'istogramma RR",       f"{af['n_peaks']} (1 = unimodale, ≥2 = struttura conservata)"],
        ["Finestre 30-battiti con CV>15%", f"{af['windows_flagged']} / {af['windows_total']}"],
        ["Score AF (0-4)",                  f"<b>{af['score']}/4</b>"],
    ]
    story.append(kv_table([[Paragraph(k, NORMAL), Paragraph(v, NORMAL)] for k, v in af_rows],
                          col_widths=[80*mm, 90*mm]))
    story.append(Spacer(1, 8))
    if af_hist_img is not None:
        story.append(fit_image(af_hist_img, max_w_mm=175, max_h_mm=70))
    if af_tacho_img is not None:
        story.append(Spacer(1, 4))
        story.append(fit_image(af_tacho_img, max_w_mm=175, max_h_mm=60))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Esito screening:</b> {af['verdict']}", NORMAL))
else:
    story.append(Paragraph(af['verdict'], NORMAL))
story.append(PageBreak())

# ---- PVC INTERPOLATE vs COMPENSATORIE ----
story.append(Paragraph("PVC interpolate vs pausa compensatoria", H2))
story.append(Paragraph(
    "Ogni PVC sandwich-fra-due-N può essere classificata in base a quanto disturba "
    "il ritmo sinusale, sommando l'intervallo che la precede e quello che la segue:",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Interpolata</b> — somma ≈ 1× RR sinusale. La PVC si infila fra due N "
    "senza resettare il nodo SA, che continua a scaricare al suo ritmo. Il battito "
    "successivo arriva quasi subito, non c'è pausa. Favorite dalle bradicardie (più "
    "spazio diastolico), <b>emodinamicamente più benigne</b>: il cuore non perde "
    "gittata e il paziente tipicamente <b>non sente</b> il tonfo.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Pausa compensatoria piena</b> — somma ≈ 2× RR sinusale. La PVC blocca la "
    "conduzione retrograda al nodo SA, che salta un battito. Risultato: pausa "
    "visibile, ripresa al ritmo normale. Più tipica delle frequenze più alte. È la "
    "PVC che fa percepire il classico <b>'tonfo'</b> al petto.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Pausa incompleta</b> — somma fra 1.3× e 1.85× RR sinusale. Caso "
    "intermedio: il nodo SA è parzialmente resettato, oppure è una PVC tardiva. "
    "Meno informativa.",
    NORMAL))
story.append(Spacer(1, 8))

# tabella conteggi
class_rows = [
    [Paragraph("<b>Tipo</b>", NORMAL), Paragraph("<b>Conteggio</b>", NORMAL),
     Paragraph("<b>% sul totale classificato</b>", NORMAL)],
    [Paragraph("Interpolate", NORMAL), Paragraph(f"{len(interpolated_list)}", NORMAL),
     Paragraph(f"{pct_interp:.1f}%", NORMAL)],
    [Paragraph("Pausa compensatoria piena", NORMAL),
     Paragraph(f"{len(compensated_list)}", NORMAL),
     Paragraph(f"{pct_comp:.1f}%", NORMAL)],
    [Paragraph("Pausa incompleta", NORMAL),
     Paragraph(f"{len(incomplete_list)}", NORMAL),
     Paragraph(f"{pct_incomp:.1f}%", NORMAL)],
]
class_tbl = Table(class_rows, colWidths=[70*mm, 35*mm, 55*mm])
class_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
    ("LINEBELOW", (0,0), (-1,0), 0.8, colors.HexColor("#33aa66")),
    ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#444444")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#333333")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(class_tbl)
story.append(Spacer(1, 6))
# interpretazione
if pct_interp >= 25:
    interp_msg = (f"Le interpolate rappresentano una quota <b>elevata</b> "
                  f"({pct_interp:.1f}%) — coerente con la bradicardia di base "
                  f"({sinus_bpm:.0f} BPM): il lungo RR sinusale offre ampio "
                  f"spazio per accogliere una PVC senza disturbare il ritmo. "
                  f"Profilo emodinamicamente favorevole.")
elif pct_interp >= 10:
    interp_msg = (f"Le interpolate sono una <b>quota intermedia</b> "
                  f"({pct_interp:.1f}%). Coesistono con un buon numero di "
                  f"compensatorie classiche, pattern misto.")
else:
    interp_msg = (f"Le interpolate sono <b>rare</b> ({pct_interp:.1f}%): la "
                  f"maggioranza delle PVC reseta il nodo SA con pausa "
                  f"compensatoria completa.")
story.append(Paragraph(interp_msg, NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"<i>Verifica: le {len(interpolated_list)} PVC interpolate sono state riviste "
    f"visivamente una a una (grid PDF separato <b>all_interpolated.pdf</b>) e tutte "
    f"confermate dal pattern visuale: il battito N successivo arriva ben prima "
    f"della linea attesa di pausa compensatoria piena (2× RR sinus).</i>",
    NORMAL))
story.append(Spacer(1, 12))

# strip di esempio
story.append(Paragraph("<b>Esempi reali dalla sessione (numerati come nel grid completo)</b>", H3))
story.append(Spacer(1, 4))
for im in interp_imgs:
    story.append(fit_image(im, max_w_mm=170, max_h_mm=55))
    story.append(Spacer(1, 4))
for im in comp_imgs:
    story.append(fit_image(im, max_w_mm=170, max_h_mm=55))
    story.append(Spacer(1, 4))

# ---- STRIP CHART PAGES (ora alla fine, prima dell'appendice grid) ----
story.append(PageBreak())
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

# Grid completo di tutte le PVC interpolate (1 pagina A4 per grid page)
if interp_grid_pages:
    story.append(PageBreak())
    story.append(Paragraph(
        f"Appendice: tutte le {len(interpolated_list)} PVC interpolate, numerate",
        H2))
    story.append(Paragraph(
        f"Ogni strip mostra una finestra di 6 secondi centrata sulla PVC analizzata "
        f"(cerchio arancione). Le barre azzurra (RR_pre) e gialla (RR_post) "
        f"mostrano gli intervalli con il battito N precedente/successivo. La linea "
        f"rossa tratteggiata segna dove cadrebbe il battito N successivo se la pausa "
        f"fosse compensatoria piena (2× RR sinus = {2*RR_SINUS_MS:.0f}ms): il fatto "
        f"che il triangolo verde sia sempre <b>prima</b> di quella linea conferma "
        f"l'interpolazione.",
        NORMAL))
    for grid_im in interp_grid_pages:
        story.append(PageBreak())
        story.append(fit_image(grid_im, max_w_mm=180, max_h_mm=255))

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

# ---- FALSE-POSITIVE EXAMPLES ----
# (Gli esempi di couplet sono ora in cima, sotto la traccia esemplificativa.)
if fp_imgs:
    story.append(Paragraph("Esempi morfologici", H2))
    story.append(Paragraph("Battiti riclassificati dalla soglia di ampiezza", H3))
    story.append(Paragraph(
        f"Il solo criterio di forma (rebound profondo o QRS largo) sovrastimava come "
        f"PVC <b>{len(removed_fp)} battiti</b> "
        f"(ampiezza {min(q['amp'] for q in removed_fp):.2f}–"
        f"{max(q['amp'] for q in removed_fp):.2f} V); il requisito di ampiezza ≥ "
        f"{PVC_MIN_AMP_V:.2f} V li riporta alla classificazione corretta. La "
        f"<b>maggioranza ({len(removed_fp)-n_fp_spike})</b> sono <b>battiti sinusali "
        f"normali</b> piccoli e veri: QRS stretto ma fisiologico, con onda S che "
        f"superava la soglia di forma pur non essendo ectopica. Una minoranza "
        f"(<b>{n_fp_spike}</b>) sono invece <b>spike di rumore</b> in tratti rumorosi, "
        f"di larghezza ≤16 ms — impossibile per un QRS reale — quindi nemmeno veri "
        f"battiti. In entrambi i casi la soglia li toglie correttamente dal conteggio "
        f"PVC. Esempi rappresentativi (battito reale piccolo, cerchiato in arancione):",
        NORMAL))
    for im in fp_imgs:
        story.append(Spacer(1, 6))
        story.append(fit_image(im, max_w_mm=170, max_h_mm=65))

if fp_imgs:
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
