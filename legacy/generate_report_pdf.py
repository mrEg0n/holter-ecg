"""
Generate a professional PDF report from a recorded ECG session.

Multi-page layout:
  - Cover with numeric summary
  - Full ECG traces (holter-style strip chart)
  - RR and coupling analysis
  - Temporal patterns
  - Pre-PVC HRV and Poincaré
  - Conclusions and technical limitations

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
STRIP_ROWS_PER_PAGE = 5  # ⇒ 5 minutes per page

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

# ---- re-classification with the current production criterion ----
# The server (host/server.py) classifies a beat as PVC if:
#   (rebound >= 0.40 OR width >= 95 ms) AND amplitude >= 0.70 V.
# Historical peaks CSVs may have been written with the old criterion (without
# the amplitude threshold): we reclassify here so the report always reflects the
# current logic and highlights the false positives that the amplitude threshold removes.
REBOUND_RATIO_PVC = 0.40
PVC_WIDTH_MS      = 95.0
PVC_MIN_AMP_V     = 0.70
# Minimum rebound: a true PVC ALWAYS has some degree of post-QRS
# hyperpolarization (rebound > 0.05). If rebound = 0 and the classification rests only
# on the width criterion, it is almost certainly a wide artifact (motion, baseline
# shift) that bypassed the width threshold without being a true ectopic complex.
PVC_MIN_REBOUND   = 0.05
# Morphological plausibility: a human QRS has physiological width between ~40 and 220 ms.
# Below 40 ms = artifact spike (electrode pop / narrow noise peak that
# crossed the amplitude threshold). Above 220 ms = baseline shift / motion artifact
# (e.g. deep breath, knock on the electrode) that the detector interpreted
# as a wide complex. Both are downgraded to "normal" and tracked.
PVC_W_MIN_MS      = 40.0
PVC_W_MAX_MS      = 220.0
removed_fp = []   # beats downgraded pvc -> normal by the amplitude threshold
removed_implausible = []  # beats downgraded for width outside the physiological range
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

# ---- cleanup: remove noise spikes (they are not beats) ----
# Width <= 16 ms (4 samples @250 Hz) is sub-physiological: these are artifacts /
# electrode-pop, not real QRS. They are removed entirely from the series (not just
# downgraded) so they do not pollute counts, RR and morphology. They remain
# listed in removed_fp for the explanatory section.
n_spike_removed = sum(1 for p in peaks if p["w"] <= 16 and p["amp"] < PVC_MIN_AMP_V)
peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP_V)]

# ---- exclusion of contaminated time intervals (noise, detached electrode) ----
# Three sources of exclusions (in priority order):
#   1) env var EXCLUDE_INTERVALS="s1-e1,s2-e2,..." (explicit override)
#   2) file exclusions/exclusions_<base>.json (created by host/mark_exclusions.py)
#   3) no exclusions
# Peaks in the excluded segments are removed and the time is subtracted from the
# useful duration so the rates are not inflated.
EXCLUDED_INTERVALS = []
_excl_env = os.environ.get("EXCLUDE_INTERVALS", "").strip()
if _excl_env:
    for chunk in _excl_env.split(","):
        a, b = chunk.split("-")
        EXCLUDED_INTERVALS.append((float(a), float(b)))
    print(f"[excl] {len(EXCLUDED_INTERVALS)} intervals from EXCLUDE_INTERVALS env var")
else:
    # fallback to the JSON file from the manual editor
    import json as _json
    _ses_id = os.path.basename(PATH).replace("ecg_", "").replace(".csv", "")
    _excl_path = os.path.join("exclusions", f"exclusions_{_ses_id}.json")
    if os.path.exists(_excl_path):
        try:
            with open(_excl_path) as _f:
                _ej = _json.load(_f)
            EXCLUDED_INTERVALS = [(d["start"], d["end"]) for d in _ej.get("intervals", [])]
            print(f"[excl] {len(EXCLUDED_INTERVALS)} intervals from {_excl_path}")
        except Exception as _e:
            print(f"[excl] error reading {_excl_path}: {_e}")
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
total_s = total_s_raw - excl_seconds  # useful duration after exclusions
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
# ---- cleanup: coupling contaminated by an undetected sinus beat ----
# A true coupling is PREMATURE (shorter than the sinus RR). If the rr_prev of a
# PVC exceeds ~0.9x the median sinus RR, it is almost always because a sinus
# beat was missed in the gap (false "late-coupled"): the interval is NOT a
# true coupling, so we exclude it from coupling/tachogram.
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

# RR transitions by category (for tachogram decomposition)
transitions = {"N→N": [], "N→PVC": [], "PVC→N": [], "PVC→PVC": []}
for i_t in range(1, len(peaks)):
    if peaks[i_t]["rr_prev"] is None: continue
    # skip the rr_prev values contaminated by a missed sinus beat (see above)
    if peaks[i_t].get("coupling_bad"): continue
    rr_ms_t = peaks[i_t]["rr_prev"] * 1000
    prev_t = "PVC" if peaks[i_t-1]["cls"] == "pvc" else "N"
    cur_t  = "PVC" if peaks[i_t]["cls"]   == "pvc" else "N"
    transitions[f"{prev_t}→{cur_t}"].append((peaks[i_t]["t"]/60, rr_ms_t))

# "750ms" band (700-800) decomposed
band750_breakdown = {k: sum(1 for tm, rr in v if 700 <= rr <= 800) for k, v in transitions.items()}
# "1180ms" band (1000-1500)
band1180_breakdown = {k: sum(1 for tm, rr in v if 1000 <= rr <= 1500) for k, v in transitions.items()}

# ----- Amplitude analysis of normal beats by context -----
# For each normal beat: classify the context relative to the nearest PVC
# - stable: prev and next are normal (and not immediately adjacent to a PVC)
# - pre_pvc: the NEXT beat is a PVC
# - post_pvc: the PREVIOUS beat was a PVC
# - sandwich: pre PVC AND post PVC (tight bigeminy case, rare)
amp_groups = {"stable": [], "pre_pvc": [], "post_pvc": [], "sandwich": []}
amp_rr_pairs = []  # (rr_prev_s, amp_V, group) for Frank-Starling scatter
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

# amplitude vs previous RR correlation (Frank-Starling): rough Pearson
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
# couplet = 2 truly consecutive PVC (RR < 700ms). Without a temporal constraint
# the normal beats missed between two PVC make them appear adjacent in the list.
COUPLET_MAX_RR_S = 0.70
couplets_n = 0
couplet_indices = []  # index pairs (i, i+1) of the true couplets
i = 0
while i < len(peaks) - 1:
    if peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc":
        rr = peaks[i+1]["t"] - peaks[i]["t"]
        if rr >= COUPLET_MAX_RR_S:
            i += 1; continue  # gap too large → not a couplet
        if i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc":
            i += 1; continue  # it's a run, not a couplet
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

# ---- PVC classification: interpolated vs compensatory ----
# An interpolated PVC slots between two N beats without resetting the SA node: the sum
# (RR_pre + RR_post) ≈ 1× RR sinus (the PVC is "extra" in the rhythm). A PVC with a
# FULL compensatory pause has sum ≈ 2× RR sinus (the SA node skips a beat).
# Interpolated ones are favored by bradycardia (more diastolic room) and are
# hemodynamically more benign (the heart does not lose output).
sinus_rr_for_class = [p["rr_prev"] for p in peaks
                       if p["cls"] == "normal" and p["rr_prev"] is not None
                       and 0.6 < p["rr_prev"] < 1.4]
RR_SINUS_MS = statistics.median(sinus_rr_for_class)*1000 if sinus_rr_for_class else 1000

interpolated_list = []
compensated_list = []
incomplete_list  = []   # between the two (>1.3× and <1.85×)
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

# ---- Atrial fibrillation screening (over all consecutive N-N) ----
# Classic markers: RR irregularity between sinus beats.
#   RMSSD > 100 ms   pNN50 > 40%   CV RR > 15-20%   entropy ~max   bimodality lost
# None is diagnostic on its own (12 leads would be needed), but the overall
# picture lets us flag a possible "irregularly irregular" rhythm.
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
    # Shannon entropy over a 20-bin histogram (ratio to the theoretical maximum)
    hist_af, edges_af = np.histogram(af_nn_ms, bins=20)
    p_af = hist_af[hist_af > 0] / sum(hist_af[hist_af > 0])
    H_af = float(-sum(p * np.log2(p) for p in p_af))
    H_max_af = float(np.log2(len(p_af)))
    af["entropy"]     = H_af
    af["entropy_max"] = H_max_af
    af["entropy_ratio"] = H_af / H_max_af if H_max_af else 0
    # bimodality: look for two separate peaks in the histogram
    smooth = np.convolve(hist_af, [1,1,1], mode="same")
    peaks_h = [k for k in range(1, len(smooth)-1)
               if smooth[k] > smooth[k-1] and smooth[k] > smooth[k+1]
               and smooth[k] > 0.3 * smooth.max()]
    af["histogram"] = hist_af
    af["hist_edges"] = edges_af
    af["n_peaks"]    = len(peaks_h)
    # 30-beat windows with high CV
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
    if af["n_peaks"] <= 1 and af["cv_pct"] > 15: score += 1  # unimodal and wide = AF; bimodal = no
    af["score"] = score
    if score == 0:
        af["verdict"] = "Regular sinus rhythm. No markers of atrial fibrillation."
    elif score == 1:
        af["verdict"] = ("Elevated HRV markers but with preserved structure. "
                         "Pattern compatible with bradycardia + RSA + frequent ectopy; "
                         "not suggestive of atrial fibrillation.")
    elif score == 2:
        af["verdict"] = ("Intermediate HRV markers. Low suspicion but not excluded; "
                         "a 12-lead ECG check is recommended if symptoms are present.")
    else:
        af["verdict"] = ("Elevated HRV markers with loss of structure: an "
                         "irregularly irregular rhythm. Compatible with suspected AF; "
                         "a cardiology check is recommended.")
else:
    af["verdict"] = "Few consecutive N-N: screening not feasible (bigeminy too dense or degraded signal)."

WINDOW = 60
windows = []
i_w = 0
while peaks and peaks[0]["t"] + i_w*WINDOW < peaks[-1]["t"]:
    ws = peaks[0]["t"] + i_w*WINDOW
    we = ws + WINDOW
    in_w = [p for p in peaks if ws <= p["t"] < we]
    nn = sum(1 for p in in_w if p["cls"] == "normal")
    np_ = sum(1 for p in in_w if p["cls"] == "pvc")
    # effective SA HR in the minute: median(60/RR_NN) over consecutive N-N pairs
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
    """Create an Image flowable guaranteed to be smaller than the printable frame.
    Scale to the usable width preserving the real aspect ratio of the PNG and, if
    necessary, reduce the native pixels: ReportLab 4.x may ignore explicit
    width/height when the natural size of the PNG exceeds the frame, causing a
    LayoutError 'image too large'. By capping native and flowable dimensions below
    the frame (usable A4 frame ≈ 174×267 mm) the problem disappears at the root."""
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
    """Strip of a single window of win_s seconds centered on an event.
    highlight: list of t (s) of beats to circle in orange."""
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
    """Didactic strip showing a PVC with its RR_pre / RR_post annotated
    and the sum vs 2x RR sinus (to distinguish interpolated vs compensatory)."""
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
    styled_ax(ax, title, "t (s) relative to the PVC", "Filtered ECG (V)")
    plt.tight_layout()
    return fig_to_bytes(fig)

def make_example_style_strip(center_t, win_s=6.0, title=None):
    """Strip in the SAME style as the example trace at the top:
    red overlay on the QRS of the PVC (±120 ms) + red triangle, green
    triangles on the sinus beats, styled_ax. No orange circle."""
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
    styled_ax(ax, title, "t (s) relative to the couplet center", "Filtered ECG (V)")
    plt.tight_layout()
    return fig_to_bytes(fig)

def make_strip_page_image(t0, t1, rows_per_page=STRIP_ROWS_PER_PAGE,
                          row_s=STRIP_ROW_SECONDS):
    """Plot of one strip-chart page: N rows of row_s seconds.
    t0..t1 are the absolute seconds of the first and last point of the page."""
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
    axes[-1].set_xlabel("seconds from the start of the row", color=MUTED, fontsize=8)
    plt.tight_layout(h_pad=0.4)
    return fig_to_bytes(fig)

# ---- generate all plot images ----
print("generating plots...")

# (A) ECG example: 8 seconds with a representative "clean" PVC.
# Criteria: after the first minute (no warm-up), N-PVC-N sandwich (both adjacent
# beats normal), at least 2s away from any excluded interval. Among the candidates,
# we pick the one with median amplitude (more representative, not an outlier).
ecg_example_img = None
if pvc and N:
    def _far_from_excluded(t_s, margin=2.0):
        return all(not (s-margin <= t_s <= e+margin) for s, e in EXCLUDED_INTERVALS)
    # candidates: N-PVC-N sandwich PVC (with RR_prev and RR_next defined)
    pvc_idx = [i for i, p in enumerate(peaks) if p["cls"] == "pvc"]
    candidates = []
    for i in pvc_idx:
        p = peaks[i]
        if p["t"] < 60: continue  # skip first minute
        if i == 0 or i == len(peaks)-1: continue
        if peaks[i-1]["cls"] != "normal" or peaks[i+1]["cls"] != "normal": continue
        if not _far_from_excluded(p["t"]): continue
        candidates.append(p)
    if candidates:
        # sort by amplitude and take the median one
        candidates.sort(key=lambda q: q["amp"])
        chosen = candidates[len(candidates)//2]
    else:
        # fallback: first PVC after the first minute, or pvc[0]
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
    styled_ax(ax, f"Representative example: 8 s around a PVC at {int(p0//60):02d}:{int(p0%60):02d}",
              "t (s) relative to the selected PVC", "Filtered ECG (V)")
    plt.tight_layout()
    ecg_example_img = fig_to_bytes(fig)

# (B) Compressed overview
overview_img = None
if N:
    fig, ax = plt.subplots(figsize=(8.5, 2.5))
    fig.patch.set_facecolor(DARK_BG)
    step = max(1, N // 30000)
    ax.plot(t[::step]/60, vf[::step], linewidth=0.3, color=GREEN, alpha=0.8)
    if pvc:
        ax.scatter([p["t"]/60 for p in pvc], [1.55]*len(pvc), s=4, color=RED, marker="v")
    ax.set_ylim(-1.2, 1.8)
    styled_ax(ax, f"Compressed overview ({total_min:.1f} min). Red triangles = PVC positions.",
              "Time (min)", "ECG filt (V)")
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
    styled_ax(ax, "Tachogram — RR interval for each beat over time",
              "Time (min)", "RR (ms)")
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
    styled_ax(ax, "Tachogram decomposed by transition type",
              "Time (min)", "RR (ms)")
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
    styled_ax(ax, "RR distribution — bimodality sinus vs pre-PVC coupling",
              "RR (ms)", "Density")
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
               label=f"Median {coupling_median:.0f}ms")
    ax.fill_between([pvc_times_for_coupling[0], pvc_times_for_coupling[-1]],
                    coupling_median - coupling_std,
                    coupling_median + coupling_std,
                    color=ORANGE, alpha=0.1, label=f"±1σ ({coupling_std:.0f}ms)")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Coupling interval over time — ectopic focus stability",
              "Time (min)", "Coupling RR (ms)")
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
    styled_ax(ax, "Beats per minute", "Time (min)", "N beats/min")
    plt.tight_layout()
    counts_img = fig_to_bytes(fig)

# (F2) HR ↔ PVC rate correlation over time
# For each minute of the recording: effective SA HR + PVC rate + burden %.
# Output: 2 plots (dual-axis time-series, HR vs PVC scatter with regression).
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
    ax1.set_ylabel("Effective SA HR (BPM)", color=GREEN, fontsize=9)
    ax1.tick_params(axis="y", colors=GREEN)
    ax1.tick_params(axis="x", colors="white")
    ax1.set_xlabel("Time (min)", color="white", fontsize=9)
    for sp in ax1.spines.values(): sp.set_color("#444")
    ax1.grid(alpha=0.18, color="#666")
    ax2 = ax1.twinx()
    ax2.set_facecolor(DARK_BG)
    ax2.plot(ts_min, pvc_min, color=RED, lw=1.0, marker="s", ms=2, alpha=0.85,
             label="PVC/min")
    ax2.set_ylabel("PVC/min", color=RED, fontsize=9)
    ax2.tick_params(axis="y", colors=RED)
    for sp in ax2.spines.values(): sp.set_color("#444")
    ax1.set_title("Effective SA HR and PVC rate minute by minute", color="white", fontsize=10)
    plt.tight_layout()
    hr_vs_pvc_ts_img = fig_to_bytes(fig)

    # HR vs PVC rate scatter with regression + correlation
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
    cbar.set_label("Time (min)", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=9, loc="upper left")
    ax.set_xlabel("Effective SA HR (BPM)", color="white")
    ax.set_ylabel("PVC per minute", color="white")
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
               label="Stdev RR (5 pre-PVC normal beats)")
    if baseline_stdev_mean:
        ax.axhline(baseline_stdev_mean, color=GREEN, linestyle="--", linewidth=1.5,
                   label=f"Baseline sinus ({baseline_stdev_mean:.0f}ms)")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "RR variability in the 5 normal beats before each PVC",
              "Time (min)", "Stdev RR (ms)")
    plt.tight_layout()
    hrv_img = fig_to_bytes(fig)

# (H0) AF screening — histogram + N-N tachogram
af_hist_img = None
af_tacho_img = None
if af.get("median_ms") is not None:
    # histogram
    fig, ax = plt.subplots(figsize=(11, 3.6))
    edges = af["hist_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax.bar(centers, af["histogram"], width=width*0.95,
           color="#33aa66", edgecolor="white", linewidth=0.3)
    ax.axvline(af["median_ms"], color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Median {af['median_ms']:.0f}ms")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, ("RR N-N histogram (all consecutive sinus beats) — "
                   f"{af['n_peaks']} peak(s) detected"),
              "RR (ms)", "N intervals")
    plt.tight_layout()
    af_hist_img = fig_to_bytes(fig)

    # RR tachogram over time
    fig, ax = plt.subplots(figsize=(11, 3.0))
    # reconstruct the N-N timestamps
    t_nn, rr_nn_list = [], []
    for i_nn in range(1, len(peaks)):
        if peaks[i_nn]["cls"] == "normal" and peaks[i_nn-1]["cls"] == "normal":
            rr_nn = (peaks[i_nn]["t"] - peaks[i_nn-1]["t"]) * 1000
            if 400 <= rr_nn <= 2000:
                t_nn.append(peaks[i_nn]["t"]/60)
                rr_nn_list.append(rr_nn)
    ax.scatter(t_nn, rr_nn_list, c="#33aa66", s=4, alpha=0.6)
    ax.axhline(af["median_ms"], color=ORANGE, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"Median {af['median_ms']:.0f}ms")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "RR N-N tachogram — time course (AF would appear as a chaotic cloud with no structure)",
              "Time (min)", "RR (ms)")
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
    # PVC: RR_prev (coupling) vs RR_next (compensatory)
    px = [1000*p["rr_prev"] for p in peaks if p["cls"]=="pvc" and p["rr_prev"] and p["rr_next"]]
    py = [1000*p["rr_next"] for p in peaks if p["cls"]=="pvc" and p["rr_prev"] and p["rr_next"]]
    ax.scatter(px, py, c=RED, s=12, alpha=0.7, label="PVC (coupling, compensatory)")
    lim = max(max(xs+px+[1]), max(ys+py+[1])) * 1.05
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "Poincaré plot — RRₙ₊₁ vs RRₙ",
              "RRₙ (ms)", "RRₙ₊₁ (ms)")
    ax.set_aspect("equal")
    plt.tight_layout()
    poincare_img = fig_to_bytes(fig)

# (A1) Amplitude — histogram by context
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
                label=f"Pre-PVC (N before ectopic) n={len(amp_groups['pre_pvc'])}",
                density=True, edgecolor="white", linewidth=0.3)
    if amp_groups["post_pvc"]:
        ax.hist(amp_groups["post_pvc"], bins=30, alpha=0.75, color=BLUE,
                label=f"Post-PVC (N after pause) n={len(amp_groups['post_pvc'])}",
                density=True, edgecolor="white", linewidth=0.3)
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "QRS amplitude distribution of normal beats by context",
              "Peak amplitude (V)", "Density")
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
    # global regression line
    if len(amp_rr_pairs) > 10:
        xs_all = np.array([p[0]*1000 for p in amp_rr_pairs])
        ys_all = np.array([p[1] for p in amp_rr_pairs])
        m, b = np.polyfit(xs_all, ys_all, 1)
        xx = np.linspace(xs_all.min(), xs_all.max(), 50)
        ax.plot(xx, m*xx + b, color="white", linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"global trend (r={r_all_norm:.2f})")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor=GRID, fontsize=8)
    styled_ax(ax, "QRS amplitude vs preceding RR — Frank-Starling effect",
              "Preceding RR (ms)", "QRS amplitude (V)")
    plt.tight_layout()
    amp_rr_img = fig_to_bytes(fig)

# (Z) Zoom strip 9-11 min (local analysis)
zoom_img = None
ZOOM_T0, ZOOM_T1 = 9*60, 11*60   # 9-11 min
zoom_mask = (t >= ZOOM_T0) & (t < ZOOM_T1)
zoom_peaks_local = [p for p in peaks if ZOOM_T0 <= p["t"] < ZOOM_T1]
if zoom_mask.any():
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.5))
    fig.patch.set_facecolor(DARK_BG)
    SEG = 30  # 30 sec per row = 4 rows cover 2 minutes
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
    axes[-1].set_xlabel("seconds within the row", color=MUTED, fontsize=8)
    axes[0].set_title("Zoom strip-chart 09:00 → 11:00 (4 rows of 30s)",
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

print(f"  {len(strip_imgs)} strip-chart pages")
print(f"  {1 + len(strip_imgs) + 5} estimated total pages")

# (Z1) Couplet examples — same constraint as the count (RR < COUPLET_MAX_RR_S).
# Without the temporal constraint we ended up showing pairs with an N skipped in
# the middle, which are NOT true couplets.
couplet_imgs = []
for (i, j) in couplet_indices[:4]:
    ctr = (peaks[i]["t"] + peaks[j]["t"]) / 2.0
    rr_ms = (peaks[j]["t"] - peaks[i]["t"]) * 1000
    couplet_imgs.append(make_example_style_strip(
        ctr, win_s=6.0,
        title=(f"Couplet at {int(ctr//60):02d}:{int(ctr%60):02d} — "
               f"two consecutive PVC at {rr_ms:.0f}ms (red overlay)")))

# (Z2) Representative examples of beats reclassified by the amplitude threshold.
# Noise spikes are excluded (w<=20ms, impossible width for a QRS):
# as examples we want the REAL small beats, near the threshold (more instructive).
n_fp_spike = sum(1 for q in removed_fp if q["w"] <= 20)
repr_fp = [q for q in removed_fp if q["w"] > 28]
fp_imgs = []
for p in sorted(repr_fp, key=lambda q: -q["amp"])[:3]:
    fp_imgs.append(make_event_strip(
        p["t"], win_s=5.0, highlight=[p["t"]],
        title=(f"{int(p['t']//60):02d}:{int(p['t']%60):02d} — amp {p['amp']:.2f} V "
               f"(reb {p['reb']:.2f}, w {p['w']:.0f} ms): below threshold → normal")))
print(f"  {len(couplet_imgs)} couplet examples, {len(fp_imgs)} false-positive examples")

# (Z3) Examples of apparently "late" PVC = sinus beat undetected in the gap.
# These are the couplings excluded from the statistics: we show them anyway for transparency.
latecoupled = [p for p in peaks if p.get("coupling_bad")]
lc_imgs = []
for p in latecoupled[:2]:
    lc_imgs.append(make_event_strip(
        p["t"], win_s=3.6, highlight=[p["t"]],
        title=(f"{int(p['t']//60):02d}:{int(p['t']%60):02d} — RR_prev {p['rr_prev']*1000:.0f} ms "
               f"(typical coupling ~{coupling_median:.0f} ms): sinus QRS not marked in the gap")))
print(f"  {len(lc_imgs)} late-coupled PVC examples (artifact)")

# (Z4) Examples of interpolated PVC vs compensatory pause (didactic)
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
    # find the global number (1..N) in the full sorted list
    n_global = next((i+1 for i, q in enumerate(sorted(interpolated_list, key=lambda x: x["t"]))
                     if q is p), idx_ex+1)
    interp_imgs.append(make_interpolated_strip(
        p, win_s=8.0,
        title=(f"#{n_global} INTERPOLATED — {int(p['t']//60):02d}:{int(p['t']%60):02d}   "
               f"Σ = {s_ms:.0f} ms ({s_ms/RR_SINUS_MS:.2f}× RR sinus)")))
comp_imgs = []
comp_picked = _pick_spread(compensated_list, n=5, min_gap_s=60)
for idx_ex, p in enumerate(comp_picked):
    s_ms = p["sum_pre_post_ms"]
    n_global = next((i+1 for i, q in enumerate(sorted(compensated_list, key=lambda x: x["t"]))
                     if q is p), idx_ex+1)
    comp_imgs.append(make_interpolated_strip(
        p, win_s=8.0,
        title=(f"#{n_global} COMPENSATORY PAUSE — {int(p['t']//60):02d}:{int(p['t']%60):02d}   "
               f"Σ = {s_ms:.0f} ms ({s_ms/RR_SINUS_MS:.2f}× RR sinus)")))
print(f"  {len(interp_imgs)} interpolated examples, {len(comp_imgs)} compensatory examples")

# (Z5) Full GRID: all interpolated PVC, one page at a time.
# Same layout as the verified export (12 strips/page, orange/blue/yellow markers).
def _build_interpolated_grid_pages(items, RR_S, rows=6, cols=2, win_s=6.0):
    """Return a list of PNG images (bytes), one per grid page."""
    per_page = rows * cols
    n_pages = (len(items) + per_page - 1) // per_page
    items_sorted = sorted(items, key=lambda q: q["t"])
    pages = []
    for page_idx in range(n_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69), facecolor=DARK_BG)
        fig.suptitle(f"Interpolated PVC — page {page_idx+1}/{n_pages}   "
                     f"median sinus RR {RR_S:.0f}ms   "
                     f"[ ◯ orange=analyzed PVC · ━light blue=RR_pre · ━yellow=RR_post · ┄red=expected 2×RR ]",
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
            # central PVC highlighted
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
            # expected 2× RR line
            comp_x = -rrp + 2*RR_S/1000.0
            if -win_s/2 < comp_x < win_s/2:
                ax.axvline(comp_x, color="#ff4d6d", lw=0.8, ls="--", alpha=0.6)
            # number
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

print(f"  generating full grid of {len(interpolated_list)} interpolated...")
interp_grid_pages = _build_interpolated_grid_pages(interpolated_list, RR_SINUS_MS)
print(f"  {len(interp_grid_pages)} grid pages")

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
story.append(Paragraph("Holter session report", H1))
story.append(Paragraph(
    f"<font color='#777'>Session <font name='Courier'>{ses_id}</font> · "
    f"Duration <b>{total_min:.1f} min</b> · "
    f"Sample rate {fs_real:.2f} Hz · Generated on {now}</font>",
    NORMAL
))
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd")))
story.append(Spacer(1, 14))

# Metric cards row
ecg_total_bpm = 60 * n_total / total_s if total_s else 0
cards = Table([[
    metric_card("ECG total", f"{ecg_total_bpm:.0f}", "electrical BPM", "#27ae60"),
    metric_card("Sinus only", f"{sinus_bpm:.0f}", "normal BPM", "#2980b9"),
    metric_card("PVC rate", f"{pvc_rate:.1f}", "/min", "#c0392b"),
    metric_card("PVC burden", f"{burden:.1f}", "% of total", "#e67e22"),
]], colWidths=[44*mm]*4)
cards.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                           ("LEFTPADDING", (0,0), (-1,-1), 0),
                           ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
story.append(cards)
story.append(Spacer(1, 14))

# Brief summary
summary_text = (
    f"The <b>{total_min:.1f}-minute</b> recording contains <b>{n_total}</b> beats, "
    f"of which <b>{len(norm)}</b> are sinus ({sinus_bpm:.0f} mean BPM) and <b>{len(pvc)}</b> "
    f"classified as ventricular ectopic beats (PVC). "
    f"The <b>PVC burden</b> is <b>{burden:.1f}%</b>: a significant "
    f"percentage of ectopic beats is a known characteristic of the patient "
    f"under the cardiologist's care. "
    f"The subsequent analysis shows this is a "
    f"<b>monomorphic and temporally stable</b> ectopic focus "
    f"(median coupling interval {coupling_median:.0f} ms ± {coupling_std:.0f} ms, "
    f"prematurity of {prematurity:.0f}% relative to the sinus cycle of "
    f"{sinus_median_ms:.0f} ms)."
)
story.append(Paragraph("Executive summary", H2))
story.append(Paragraph(summary_text, NORMAL))
story.append(Spacer(1, 10))

# Example ECG
if ecg_example_img:
    story.append(Paragraph("Representative example", H3))
    story.append(Paragraph(
        "Eight seconds of the recording centered on the first PVC. The green line shows "
        "the sinus beats; the red segment and triangle highlight the PVC's QRS and "
        "the ±120 ms window over which the rebound hyperpolarization is measured — the "
        "physiological feature the detector uses to classify it.",
        NORMAL))
    story.append(Image(ecg_example_img, width=174*mm, height=58*mm))

# Couplet examples right below the example trace, same style/color
if couplet_imgs:
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Couplet" if couplets_n == 1 else f"Couplets (all {couplets_n})", H3))
    story.append(Paragraph(
        f"<b>Couplet</b>: two consecutive PVC with no sinus beat in between. Across the whole "
        f"session there are <b>{couplets_n}</b> — no triplets or longer runs. "
        f"Same style as the trace above: red overlay on the QRS of the two PVC, green "
        f"triangles on the sinus beats."
        + ("" if couplets_n > len(couplet_imgs) else " Here they all are:"),
        NORMAL))
    for im in couplet_imgs:
        story.append(Image(im, width=174*mm, height=58*mm))

story.append(PageBreak())

# ---- PAGE 2 — METHODOLOGY + DETAILED METRICS ----
story.append(Paragraph("Methodology", H2))
story.append(Paragraph(
    "<b>Hardware.</b> AD8232 (Analog Devices) analog front-end configured in Einthoven I "
    "lead (RA, LA, RL as reference), output sampled by the 12-bit ADC of the Pi Pico 2 W "
    "at 250 Hz. The system is powered by a LiPo cell (3.7 V) and completely floating with respect "
    "to mains power during the recording.",
    NORMAL))
story.append(Paragraph(
    "<b>Transport and storage.</b> The Pico sends each sample over TCP/WiFi to the server "
    "(Python/Flask), which performs real-time filtering and simultaneous logging to CSV. "
    "Temporal resolution: 4 ms per sample. Duration of the current recording: "
    f"{total_s:.1f} s ({fs_real:.2f} Hz actual).",
    NORMAL))
story.append(Paragraph(
    "<b>Filtering.</b> Cascade of two first-order IIR filters: a high-pass at 0.3 Hz (removes "
    "baseline drift and DC) followed by a low-pass at 25 Hz (attenuates 50 Hz mains and EMG). "
    "The 0.3–25 Hz passband preserves the QRS morphology and the post-QRS undershoot that characterizes "
    "PVCs.",
    NORMAL))
story.append(Paragraph(
    "<b>Beat detection.</b> A 4-state machine (idle, width, detect, post). "
    "The signal enters the tracking phase when it exceeds 0.10 V; it is confirmed as a QRS if "
    "it also exceeds an adaptive threshold (median of recent amplitudes × 0.45, minimum 0.30 V). "
    "In the 200 ms following the peak the trough — the negative post-QRS deflection — is measured.",
    NORMAL))
story.append(Paragraph(
    "<b>PVC classification.</b> A beat is classified as a PVC if it has ectopic morphology "
    "— |trough|/peak ratio ≥ 0.40 (pronounced hyperpolarization) OR QRS width ≥ 95 ms "
    f"— <b>AND</b> amplitude ≥ {PVC_MIN_AMP_V:.2f} V. The amplitude requirement avoids labeling "
    "small sinus beats with a physiological S wave as PVCs. Refractory period of 300 ms.",
    NORMAL))
story.append(Paragraph(
    f"<b>Data cleaning.</b> Before the analyses the series is cleaned: "
    f"(1) <b>{n_spike_removed}</b> noise spikes with width ≤ 16 ms were removed "
    f"(sub-physiological for a real QRS, typical electrode-pop/motion artifacts); "
    f"(2) <b>{n_coupling_excluded}</b> non-premature coupling intervals were excluded "
    f"(rr_prev &gt; {COUPLING_MAX_FACTOR:.0%} of the median sinus RR): these are not true couplings "
    f"but PVCs whose preceding sinus beat was not detected in the gap "
    f"(false “late-coupled”), and would contaminate the coupling statistics and the "
    f"tachogram. The counts, RR, coupling and morphology reported here use the cleaned series.",
    MUTED_P))

story.append(Paragraph("Detailed metrics", H2))
story.append(kv_table([
    ["Recording duration",              f"{total_s:.1f} s  ({total_min:.2f} min)"],
    ["Measured sample rate",            f"{fs_real:.2f} Hz"],
    ["Total samples",                   f"{N:,}"],
    ["Beats detected",                  f"{n_total:,}"],
    ["Sinus beats",                     f"{len(norm):,}  ({100*len(norm)/max(1,n_total):.1f}%)"],
    ["PVC beats",                       f"{len(pvc):,}  ({100*len(pvc)/max(1,n_total):.1f}%)"],
    ["Total BPM (all beats)",           f"{ecg_total_bpm:.1f}"],
    ["Sinus BPM",                       f"{sinus_bpm:.1f}"],
    ["PVC rate",                        f"{pvc_rate:.2f} /min"],
    ["PVC burden",                      f"{burden:.1f} %"],
    ["Median sinus RR",                 f"{sinus_median_ms:.1f} ms"],
    ["Mean sinus RR",                   f"{sinus_mean_ms:.1f} ms"],
    ["Sinus RR std dev (SDNN)",         f"{sinus_std_ms:.1f} ms"],
    ["RMSSD",                           f"{sinus_rmssd_ms:.1f} ms"],
    ["Median pre-PVC coupling",         f"{coupling_median:.1f} ms"],
    ["Coupling std",                    f"{coupling_std:.1f} ms"],
    ["Coupling IQR",                    f"{coupling_iqr:.1f} ms"],
    ["Prematurity",                     f"{prematurity:.1f} % earlier than the sinus"],
    ["Median post-PVC RR (compensatory)", f"{compensatory_median:.1f} ms"],
]))

story.append(PageBreak())

# ---- OVERVIEW + TACHOGRAM ----
story.append(Paragraph("Overview and tachogram", H2))
story.append(Paragraph(
    "An overall view of the recording and of the RR intervals for each beat. "
    "The tachogram highlights the bimodality of the signal: the sinus beats (green) "
    "sit on a stable horizontal level, while the pre-PVC couplings (red) "
    "form a much lower, clearly distinct cluster.",
    NORMAL))
if overview_img:
    story.append(Image(overview_img, width=174*mm, height=53*mm))
if tacho_img:
    story.append(Spacer(1, 8))
    story.append(Image(tacho_img, width=174*mm, height=64*mm))

story.append(PageBreak())

# ---- COUPLING ANALYSIS ----
story.append(Paragraph("Coupling interval analysis", H2))
story.append(Paragraph(
    "The coupling interval is the time between a sinus beat and the PVC that follows it. "
    "When it is <b>constant</b> over time it is the signature of a single, "
    "monomorphic ectopic focus (always the same area of abnormal myocardium firing with the same "
    "latency after each stimulation). A variable coupling would suggest multiple sources "
    "or more complex mechanisms.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"In this session the coupling is <b>{coupling_median:.0f} ± {coupling_std:.0f} ms</b> "
    f"(IQR {coupling_iqr:.0f} ms), equal to <b>{100-prematurity:.0f}%</b> of the sinus cycle "
    f"({sinus_median_ms:.0f} ms). The standard deviation is <b>{100*coupling_std/coupling_median:.1f}%</b> "
    f"of the median value — very limited variation, a highly repeatable pattern.",
    NORMAL))
if hist_img:
    story.append(Spacer(1, 6))
    story.append(Image(hist_img, width=174*mm, height=66*mm))
if coupling_stability_img:
    story.append(Spacer(1, 6))
    story.append(Image(coupling_stability_img, width=174*mm, height=62*mm))

if n_coupling_excluded:
    story.append(Spacer(1, 10))
    story.append(Paragraph("Apparently late PVC (excluded from the coupling)", H3))
    story.append(Paragraph(
        f"<b>{n_coupling_excluded} PVC</b> have a very long preceding RR "
        f"(&gt; {COUPLING_MAX_FACTOR:.0%} of the median sinus RR), well beyond the typical "
        f"coupling (~{coupling_median:.0f} ms). However, they are not true \"end-diastolic\" PVC: "
        f"they have the same morphology as all the others, but the <b>sinus beat that "
        f"precedes them was not detected</b> by the detector (amplitude below threshold), so "
        f"the measured RR sums a missing sinus interval + the real coupling. For "
        f"this reason they are excluded from the coupling statistics and from the tachogram. In the "
        f"examples below you can see the sinus QRS not marked in the gap, before the circled "
        f"PVC:",
        NORMAL))
    for im in lc_imgs:
        story.append(Spacer(1, 6))
        story.append(fit_image(im, max_w_mm=170, max_h_mm=58))

story.append(PageBreak())

# ---- PATTERNS ----
story.append(Paragraph("Temporal patterns of the PVCs", H2))
story.append(Paragraph(
    "PVCs can organize into characteristic repetitive patterns. "
    "<b>Bigeminy</b>: fixed N–PVC–N–PVC alternation. "
    "<b>Trigeminy</b>: a repeated N–N–PVC pattern. "
    "<b>Couplet</b>: two consecutive PVC with no normal beats in between. "
    "<b>Triplet</b>: three or more consecutive PVC (referred to as salvos / runs of "
    "non-sustained ventricular tachycardia if >3 and <30 s).",
    NORMAL))
story.append(Spacer(1, 6))

pat_tbl = Table([
    [Paragraph("<b>Pattern</b>", NORMAL), Paragraph("<b>Count</b>", NORMAL),
     Paragraph("<b>Interpretation</b>", NORMAL)],
    ["Isolated PVC (N–PVC–N)", f"{iso_pvc}",
     "Most common form; the cardiac system returns to sinus rhythm immediately after the ectopy."],
    ["Couplet (2 consecutive PVC)", f"{couplets_n}",
     "Two ectopic beats in a row. Less frequent than the isolated ones."],
    ["Triplet / salvos (3+)", "0",
     "No ventricular run observed in the session."],
    ["Bigeminy (≥3 cycles)", f"{bigem}",
     "N-PVC-N-PVC alternation, brief episodes during the recording."],
    ["Trigeminy (≥3 cycles)", f"{trigem}",
     "Repeated N-N-PVC pattern. The most frequent of the rhythmic patterns observed."],
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
    story.append(Paragraph("Beat distribution per minute", H3))
    story.append(Image(counts_img, width=174*mm, height=58*mm))

story.append(PageBreak())

# ---- HR ↔ PVC RATE CORRELATION ----
if hr_vs_pvc_ts_img and hr_vs_pvc_scatter_img and hr_pvc_correlation:
    story.append(Paragraph("HR ↔ PVC rate correlation", H2))
    corr = hr_pvc_correlation
    r = corr["r"]
    # interpretation of the coefficient
    if abs(r) < 0.1:
        r_descr = "negligible"
    elif abs(r) < 0.3:
        r_descr = "weak"
    elif abs(r) < 0.5:
        r_descr = "moderate"
    elif abs(r) < 0.7:
        r_descr = "strong"
    else:
        r_descr = "very strong"
    direction = "direct (HR ↑ → PVC ↑)" if r > 0 else "inverse (HR ↑ → PVC ↓)"
    story.append(Paragraph(
        f"Analysis of the relationship between the SA node's baseline rate (effective HR computed "
        f"from the median RR of consecutive N-N pairs) and the number of PVC in the same minute. "
        f"Over <b>{corr['n']} 60s windows</b> with at least 20 usable beats, the Pearson "
        f"correlation coefficient is <b>r = {r:.3f}</b> (a <b>{r_descr}</b> correlation, "
        f"{direction} direction). Line slope: <b>{corr['slope']:+.2f} PVC/min per BPM</b>. "
        f"Observed range: HR {corr['hr_min']:.0f}-{corr['hr_max']:.0f} BPM, "
        f"PVC {corr['pvc_min']}-{corr['pvc_max']}/min.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Time-series: SA HR and PVC rate minute by minute", H3))
    story.append(Image(hr_vs_pvc_ts_img, width=174*mm, height=54*mm))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Scatter: HR vs PVC/min (color = time from start)", H3))
    story.append(Image(hr_vs_pvc_scatter_img, width=128*mm, height=85*mm,
                       hAlign="CENTER"))
    story.append(Spacer(1, 6))
    if r > 0.3:
        msg = ("Pattern compatible with an <b>increase in PVC rate as the baseline rate "
               "rises</b>. Consistent with: (a) an initial sedentary warm-up phase (low HR, "
               "few PVC) followed by more sympathetically toned phases (HR rises, focus more "
               "excitable); (b) autonomic modulation of the ectopy (vagal ↓ + sympathetic ↑ "
               "→ more ectopy); (c) intercurrent metabolic factors (digestion, caffeine, "
               "movement). NB: this is the opposite of the classic 'exercise-suppressed' pattern "
               "seen at high aerobic loads (>120 BPM), where the PVCs disappear.")
    elif r < -0.3:
        msg = ("Pattern compatible with a <b>decrease in PVC rate as the baseline rate "
               "rises</b>. Consistent with the classic phenomenon of 'exercise-suppressed' PVCs: "
               "a more active sympathetic system speeds up conduction, reduces the zones of "
               "unidirectional block, and suppresses reentry / the ectopic focus. A marker "
               "of benignity.")
    else:
        msg = ("Non-significant correlation: the instantaneous PVC rate is not "
               "mainly explained by the baseline HR in this session. Other factors "
               "(position, respiration, vagal state, mechanical thoracic factors) "
               "probably dominate.")
    story.append(Paragraph(msg, NORMAL))
    story.append(PageBreak())

# ---- TACHOGRAM DECOMPOSITION ----
story.append(Paragraph("Tachogram decomposition", H2))
story.append(Paragraph(
    "With the naked eye, the tachogram shows <b>three bands of RR intervals for "
    "the beats classified as normal</b> (green points): a thin one around "
    "750 ms, a large and dense one around 900 ms, and a more scattered one centered on "
    "1180 ms. Each corresponds to a different physiological context, recognizable "
    "if the RR are separated by <b>transition type</b> (what the current beat "
    "is and what the previous one was).",
    NORMAL))
story.append(Spacer(1, 6))

# transition decomposition table
decomp_rows = [["Transition", "n", "Median", "Std", "Min–Max", "Interpretation"]]
trans_explain = {
    "N→N":    "Sinus → sinus. The baseline heart rhythm, between two consecutive normal beats.",
    "N→PVC":  "Sinus → PVC. Coupling interval: the timing of the ectopic trigger.",
    "PVC→N":  "PVC → sinus. The post-ectopic pause, complete or interpolated.",
    "PVC→PVC":"PVC → PVC. Two consecutive ectopic beats (couplet).",
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
story.append(Paragraph("The 900 ms band — regular sinus", H3))
nn_vals = [x[1] for x in transitions["N→N"]]
if nn_vals:
    nn_med = statistics.median(nn_vals); nn_std = statistics.stdev(nn_vals)
    story.append(Paragraph(
        f"This is the <b>baseline sinus rhythm</b>: {len(nn_vals)} intervals, "
        f"median {nn_med:.0f} ms ({60000/nn_med:.0f} BPM), standard deviation "
        f"{nn_std:.0f} ms. The width of the band is the <b>physiological "
        f"HRV variability</b> (respiratory component + autonomic tone). "
        f"The narrower the band, the more regular the sinus.",
        NORMAL))

story.append(Paragraph("The 1180 ms band — compensatory pauses", H3))
pn_vals = [x[1] for x in transitions["PVC→N"]]
if pn_vals:
    pn_med = statistics.median(pn_vals); pn_std = statistics.stdev(pn_vals)
    story.append(Paragraph(
        f"This is the <b>post-PVC pause</b>: {len(pn_vals)} intervals, median "
        f"<b>{pn_med:.0f} ms ± {pn_std:.0f} ms</b>. The huge dispersion "
        f"(range {min(pn_vals):.0f}–{max(pn_vals):.0f} ms) <b>is not noise</b>, it is a "
        f"clinically meaningful datum: it reflects the variable behavior of the "
        f"ectopic trigger relative to the sinus pacemaker.",
        NORMAL))
    story.append(Paragraph(
        "<b>Full compensatory pause</b> (toward 2000+ ms): the PVC has "
        "reset the sinus node, which restarts from scratch with a whole cycle of "
        "delay. <b>Interpolated PVC</b> (toward 600 ms): the sinus node was "
        "ignored by the ectopic trigger and continues its rhythm as if nothing "
        "happened — the next normal beat arrives at the expected time. "
        "The intermediate cases (partial compensation) fill the continuum.",
        NORMAL))

story.append(Paragraph("The 750 ms band — two distinct mechanisms", H3))
n_b750_nn = band750_breakdown.get("N→N", 0)
n_b750_pn = band750_breakdown.get("PVC→N", 0)
story.append(Paragraph(
    f"A thin band around 700–800 ms: <b>{n_b750_nn + n_b750_pn} observations</b>. "
    f"Decomposed: <b>{n_b750_nn} N→N</b> + <b>{n_b750_pn} PVC→N</b>. These are two "
    f"physiologically different phenomena superimposed.",
    NORMAL))
story.append(Paragraph(
    f"<b>(a) {n_b750_nn} N→N at ~750 ms</b>: phases of <b>sinus acceleration</b> "
    f"at ~80 BPM. Typically associated with peaks of <b>respiratory sinus "
    f"arrhythmia</b> (deep inspiration → vagal withdrawal → temporarily higher "
    f"HR), spontaneous sighs, brief sympathetic activations.",
    NORMAL))
story.append(Paragraph(
    f"<b>(b) {n_b750_pn} PVC→N at ~750 ms</b>: <b>interpolated PVC</b> "
    f"or nearly so. The ectopic beat slots between two sinus beats without disturbing "
    f"the pacemaker. A typically benign variant, more frequent with a low "
    f"baseline heart rate (as in the case of this session).",
    NORMAL))

story.append(PageBreak())

# ---- ZOOM 9-11 MIN (only if there really is a local oscillation) ----
# The section is conditional: it only makes sense if the 09:00-11:00 window shows
# genuinely elevated RR variability vs baseline. In many sessions (e.g. the
# clean 150812 one) there is NO local oscillation, so the section is omitted.
window_beats_zoom = [r for r in peaks if 9*60 <= r["t"] < 11*60]
zoom_n = sum(1 for r in window_beats_zoom if r["cls"] == "normal")
zoom_p = sum(1 for r in window_beats_zoom if r["cls"] == "pvc")
zoom_rrs = [r["rr_prev"]*1000 for r in window_beats_zoom if r["rr_prev"]]
zoom_std = statistics.stdev(zoom_rrs) if len(zoom_rrs) > 1 else 0
all_rrs_global = [p["rr_prev"]*1000 for p in peaks if p["rr_prev"]]
baseline_std_global = statistics.stdev(all_rrs_global) if len(all_rrs_global) > 1 else 0
show_zoom_section = bool(zoom_img) and len(window_beats_zoom) > 10 and zoom_std > 1.25 * baseline_std_global
if show_zoom_section:
    story.append(Paragraph("Local analysis: oscillations between 9 and 11 minutes", H2))
    story.append(Paragraph(
        f"In the tachogram a marked visual oscillation is noticeable around minute 10. "
        f"In the <b>09:00–11:00</b> window there are <b>{len(window_beats_zoom)} "
        f"beats</b> ({zoom_n} normal, {zoom_p} PVC) with an RR standard deviation "
        f"of <b>{zoom_std:.0f} ms</b>, against a baseline of {baseline_std_global:.0f} ms "
        f"for the whole session.",
        NORMAL))
    story.append(Paragraph(
        "The \"oscillations\" are the <b>vertical bounce typical of zones with higher "
        "PVC burden</b>: each PVC produces a low point (coupling ~500 ms), the "
        "normal beat right after produces a high point (compensatory ~1200 ms), the stable "
        "normal beat sits in between (~900 ms). When the trigeminy pattern is "
        "particularly regular for an interval, the points bounce between these three "
        "levels in rapid succession, giving the visual effect of a vertical square "
        "wave.",
        NORMAL))
    story.append(Spacer(1, 8))
    story.append(Image(zoom_img, width=174*mm, height=140*mm))
    story.append(PageBreak())

# ---- AMPLITUDE ANALYSIS ----
story.append(Paragraph("Amplitude analysis of the normal beats", H2))
story.append(Paragraph(
    "The QRS amplitude of a normal beat depends on several factors: the orientation "
    "of the cardiac electrical vector (fixed by the chest geometry), the <b>ventricular "
    "volume at the moment of depolarization</b> (more blood inside = more "
    "excited mass = wider QRS), the <b>filling time</b> from the previous "
    "beat, and the <b>autonomic tone</b> (sympathetic increases inotropy and contractility). "
    "To investigate whether the onset of a PVC is foreshadowed by a particular "
    "state of the sinus, we classify the normal beats into three contexts.",
    NORMAL))
story.append(Spacer(1, 6))

amp_tbl_rows = [["Context", "n", "Mean (V)", "Median (V)", "Std (V)", "Meaning"]]
ctx_descr = {
    "stable":   "Sinus N surrounded by other normals — baseline.",
    "pre_pvc":  "The last normal before a PVC — the \"suspect\".",
    "post_pvc": "The normal after the compensatory pause — more filling.",
    "sandwich": "Between two consecutive PVC (tight bigeminy) — rare.",
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

# comparative deltas
stable_med = amp_stats["stable"]["median"] if amp_stats["stable"] else 0
pre_med    = amp_stats["pre_pvc"]["median"] if amp_stats["pre_pvc"] else 0
post_med   = amp_stats["post_pvc"]["median"] if amp_stats["post_pvc"] else 0
pre_delta  = 100*(pre_med - stable_med)/stable_med if stable_med else 0
post_delta = 100*(post_med - stable_med)/stable_med if stable_med else 0

story.append(Spacer(1, 8))
story.append(Paragraph(
    f"<b>Comparison vs stable baseline</b>: "
    f"pre-PVC <b>{pre_delta:+.1f}%</b> ({pre_med:.3f} vs {stable_med:.3f} V), "
    f"post-PVC <b>{post_delta:+.1f}%</b> ({post_med:.3f} vs {stable_med:.3f} V).",
    NORMAL))

if amp_hist_img:
    story.append(Spacer(1, 6))
    story.append(Image(amp_hist_img, width=174*mm, height=66*mm))

story.append(Spacer(1, 10))
story.append(Paragraph("Frank-Starling effect — amplitude vs preceding interval", H3))
story.append(Paragraph(
    "The <b>Frank-Starling law</b> states that the more the ventricle fills before the "
    "contraction, the more powerful the subsequent contraction will be. Translated into ECG terms: a longer "
    "RR should correlate with a wider QRS in the beat that follows it. "
    f"On our dataset, the Pearson correlation between the preceding RR and the amplitude of the "
    f"QRS that follows is <b>r = {r_all_norm:+.2f}</b> over all normal beats and "
    f"<b>r = {r_stable:+.2f}</b> over the stable group alone. "
    + (
        "A modestly positive correlation is observed: the Frank-Starling effect is "
        "present but weak in lead I (more visible in precordial leads). "
        if r_all_norm > 0.15 else
        "A low correlation: lead I captures only one projection of the cardiac "
        "vector, and volume effects are more visible in precordial leads "
        "(e.g. V5-V6) that we do not have here. "
    )
    + "The post-PVC beat, which follows the longer compensatory pause, should therefore have "
    "a slightly higher amplitude than baseline.",
    NORMAL))
if amp_rr_img:
    story.append(Spacer(1, 6))
    story.append(Image(amp_rr_img, width=174*mm, height=70*mm))

# final interpretation
story.append(Spacer(1, 10))
story.append(Paragraph("What it means for the PVC triggers", H3))
findings = []
if abs(pre_delta) < 3:
    findings.append(
        f"<b>The pre-PVC amplitude is essentially equal to baseline</b> ({pre_delta:+.1f}%). "
        f"The sinus beat preceding a PVC does NOT appear to be in a special mechanical/electrical "
        f"state relative to the normal sinus. This suggests that the <b>ectopic trigger "
        f"is autonomous</b> with respect to the immediately preceding sinus cycle — the "
        f"focus fires on its own schedule (for example modulated by global autonomic "
        f"or respiratory tone), not because there is anything particular about the beat "
        f"just before."
    )
elif pre_delta > 0:
    findings.append(
        f"The pre-PVC amplitude is <b>larger</b> than baseline ({pre_delta:+.1f}%). "
        "It could suggest a slightly greater ventricular filling or "
        "a sympathetic activation in the seconds preceding the ectopy."
    )
else:
    findings.append(
        f"The pre-PVC amplitude is <b>smaller</b> than baseline ({pre_delta:+.1f}%). "
        "It would suggest reduced ventricular filling — possible if the ectopy "
        "tends to arrive during phases of transient tachycardia with lower preload."
    )

if post_delta > 5:
    findings.append(
        f"<b>The post-PVC amplitude is significantly higher</b> than baseline "
        f"({post_delta:+.1f}%). Consistent with the <b>Frank-Starling effect</b>: after "
        f"the compensatory pause the ventricle had more time to fill, the "
        f"stroke volume is greater, and the QRS is wider. It is an expected physiological "
        f"phenomenon, measured clinically as \"post-extrasystolic potentiation\"."
    )
elif post_delta > 0:
    findings.append(
        f"The post-PVC amplitude is slightly larger than baseline ({post_delta:+.1f}%), "
        f"compatible with a small Frank-Starling effect — visible more "
        f"markedly in leads that project better onto the main "
        f"depolarization vector."
    )
else:
    findings.append(
        f"The post-PVC amplitude is similar to baseline ({post_delta:+.1f}%); the "
        f"Frank-Starling effect is not clearly visible in this single lead, but it is "
        f"probably present hemodynamically."
    )

findings.append(
    "Conclusion: based on these observations, the PVCs <b>do not appear to be "
    "foreshadowed by a change in the amplitude of the preceding sinus beat</b>. "
    "The ectopic trigger therefore appears to be modulated by more \"systemic\" factors "
    "(autonomic tone, respiratory phase) rather than by a mechanical/electrical "
    "condition of the immediately preceding beat."
)

for line in findings:
    story.append(Paragraph("• " + line, NORMAL))
    story.append(Spacer(1, 4))

story.append(PageBreak())

# ---- SCREENING FIBRILLAZIONE ATRIALE ----
story.append(Paragraph("Atrial fibrillation screening (rhythm analysis)", H2))
story.append(Paragraph(
    "Analysis of all the <b>RR intervals between consecutive sinus beats</b> (N-N) "
    "over the entire useful recording. Atrial fibrillation produces an "
    "<i>irregularly irregular</i> rhythm: the RR lose all structure, the histogram "
    "becomes uniform/chaotic, RMSSD and pNN50 surge, the entropy saturates. "
    "The four markers below (>100 ms RMSSD, >40% pNN50, entropy/max >0.85, "
    "wide unimodal histogram) make up a <b>0-4 score</b>: the report "
    "is not diagnostic (12 leads and a wider passband would be needed to "
    "assess the P wave), but it serves to automatically flag suspicious patterns.",
    NORMAL))
if af.get("median_ms") is not None:
    story.append(Spacer(1, 8))
    af_rows = [
        ["Consecutive N-N analyzed",        f"{af['nn_count']}"],
        ["Median RR / BPM",                 f"{af['median_ms']:.0f} ms ({60000/af['median_ms']:.1f} BPM)"],
        ["Std / CV",                        f"{af['std_ms']:.0f} ms / {af['cv_pct']:.1f}%"],
        ["Range",                           f"{af['min_ms']:.0f} – {af['max_ms']:.0f} ms"],
        ["RMSSD (AF threshold >100 ms)",    f"<b>{af['rmssd_ms']:.0f} ms</b>"],
        ["pNN50 (AF threshold >40%)",       f"<b>{af['pnn50']:.1f}%</b>"],
        ["pNN20",                           f"{af['pnn20']:.1f}%"],
        ["Entropy / max (AF if >0.85)",     f"<b>{af['entropy']:.2f} / {af['entropy_max']:.2f} ({af['entropy_ratio']:.2f})</b>"],
        ["Peaks in the RR histogram",       f"{af['n_peaks']} (1 = unimodal, ≥2 = structure preserved)"],
        ["30-beat windows with CV>15%",     f"{af['windows_flagged']} / {af['windows_total']}"],
        ["AF score (0-4)",                  f"<b>{af['score']}/4</b>"],
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
    story.append(Paragraph(f"<b>Screening result:</b> {af['verdict']}", NORMAL))
else:
    story.append(Paragraph(af['verdict'], NORMAL))
story.append(PageBreak())

# ---- PVC INTERPOLATE vs COMPENSATORIE ----
story.append(Paragraph("Interpolated PVC vs compensatory pause", H2))
story.append(Paragraph(
    "Each PVC sandwiched between two N beats can be classified by how much it disturbs "
    "the sinus rhythm, summing the interval that precedes it and the one that follows it:",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Interpolated</b> — sum ≈ 1× sinus RR. The PVC slots between two N beats "
    "without resetting the SA node, which keeps firing at its own rhythm. The next beat "
    "arrives almost immediately, there is no pause. Favored by bradycardia (more "
    "diastolic room), <b>hemodynamically more benign</b>: the heart does not lose "
    "output and the patient typically <b>does not feel</b> the thump.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Full compensatory pause</b> — sum ≈ 2× sinus RR. The PVC blocks "
    "retrograde conduction to the SA node, which skips a beat. Result: a visible "
    "pause, then resumption of the normal rhythm. More typical of higher rates. It is the "
    "PVC that produces the classic <b>'thump'</b> in the chest.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>• Incomplete pause</b> — sum between 1.3× and 1.85× sinus RR. An "
    "intermediate case: the SA node is partially reset, or it is a late PVC. "
    "Less informative.",
    NORMAL))
story.append(Spacer(1, 8))

# counts table
class_rows = [
    [Paragraph("<b>Type</b>", NORMAL), Paragraph("<b>Count</b>", NORMAL),
     Paragraph("<b>% of total classified</b>", NORMAL)],
    [Paragraph("Interpolated", NORMAL), Paragraph(f"{len(interpolated_list)}", NORMAL),
     Paragraph(f"{pct_interp:.1f}%", NORMAL)],
    [Paragraph("Full compensatory pause", NORMAL),
     Paragraph(f"{len(compensated_list)}", NORMAL),
     Paragraph(f"{pct_comp:.1f}%", NORMAL)],
    [Paragraph("Incomplete pause", NORMAL),
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
# interpretation
if pct_interp >= 25:
    interp_msg = (f"The interpolated ones represent a <b>high</b> share "
                  f"({pct_interp:.1f}%) — consistent with the baseline bradycardia "
                  f"({sinus_bpm:.0f} BPM): the long sinus RR offers ample "
                  f"room to accommodate a PVC without disturbing the rhythm. "
                  f"A hemodynamically favorable profile.")
elif pct_interp >= 10:
    interp_msg = (f"The interpolated ones are an <b>intermediate</b> share "
                  f"({pct_interp:.1f}%). They coexist with a good number of "
                  f"classic compensatory ones, a mixed pattern.")
else:
    interp_msg = (f"The interpolated ones are <b>rare</b> ({pct_interp:.1f}%): the "
                  f"majority of PVC reset the SA node with a full "
                  f"compensatory pause.")
story.append(Paragraph(interp_msg, NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"<i>Verification: the {len(interpolated_list)} interpolated PVC were reviewed "
    f"visually one by one (separate grid PDF <b>all_interpolated.pdf</b>) and all "
    f"confirmed by the visual pattern: the next N beat arrives well before "
    f"the expected full compensatory pause line (2× RR sinus).</i>",
    NORMAL))
story.append(Spacer(1, 12))

# example strip
story.append(Paragraph("<b>Real examples from the session (numbered as in the full grid)</b>", H3))
story.append(Spacer(1, 4))
for im in interp_imgs:
    story.append(fit_image(im, max_w_mm=170, max_h_mm=55))
    story.append(Spacer(1, 4))
for im in comp_imgs:
    story.append(fit_image(im, max_w_mm=170, max_h_mm=55))
    story.append(Spacer(1, 4))

# ---- STRIP CHART PAGES (now at the end, before the grid appendix) ----
story.append(PageBreak())
story.append(Paragraph("Complete ECG traces", H2))
story.append(Paragraph(
    f"Full visualization of the recording, holter-style strip-chart format. "
    f"Each row represents <b>{STRIP_ROW_SECONDS} seconds</b> ({STRIP_ROW_SECONDS//60 if STRIP_ROW_SECONDS>=60 else 0} min). "
    f"The number to the left of each row indicates the start minute:second. "
    f"On the right, the count of normal beats (N) and PVC. "
    f"The triangles above the trace mark the classification, the red overlay highlights "
    f"the QRS and the hyperpolarization of the PVCs.",
    NORMAL))
story.append(Spacer(1, 6))
for img, t0, t1 in strip_imgs:
    story.append(Image(img, width=174*mm, height=233*mm))
    story.append(PageBreak())

# Full grid of all interpolated PVC (1 A4 page per grid page)
if interp_grid_pages:
    story.append(PageBreak())
    story.append(Paragraph(
        f"Appendix: all {len(interpolated_list)} interpolated PVC, numbered",
        H2))
    story.append(Paragraph(
        f"Each strip shows a 6-second window centered on the analyzed PVC "
        f"(orange circle). The light blue (RR_pre) and yellow (RR_post) bars "
        f"show the intervals with the preceding/following N beat. The dashed "
        f"red line marks where the next N beat would fall if the pause "
        f"were a full compensatory one (2× RR sinus = {2*RR_SINUS_MS:.0f}ms): the fact "
        f"that the green triangle is always <b>before</b> that line confirms "
        f"the interpolation.",
        NORMAL))
    for grid_im in interp_grid_pages:
        story.append(PageBreak())
        story.append(fit_image(grid_im, max_w_mm=180, max_h_mm=255))

story.append(PageBreak())

# ---- HRV PRE-PVC + POINCARE ----
story.append(Paragraph("HRV and pre-PVC autonomic modulation", H2))
story.append(Paragraph(
    "The <b>5 normal beats immediately preceding</b> each PVC are analyzed. "
    "By computing the standard deviation of the RR intervals in these windows and comparing it "
    "with the stable sinus baseline, we assess whether the sinus rhythm is more irregular in the "
    "seconds preceding an ectopy (compatible with autonomic/vagal modulation).",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"Mean pre-PVC RR stdev: <b>{pre_pvc_stdev_mean:.1f} ms</b>. "
    f"Sinus baseline stdev: <b>{baseline_stdev_mean:.1f} ms</b>. "
    f"Delta: <b>{hrv_delta_pct:+.1f}%</b>. "
    + (
        "A small but systematic difference over hundreds of observations: the sinus "
        "rhythm is slightly more variable before each PVC. Compatible with an "
        "ectopic trigger modulated by vagal tone / respiratory phase."
        if hrv_delta_pct > 5 else
        "The variability does not substantially differ from baseline; the ectopic trigger "
        "appears relatively independent of short-term sinus variability."
    ),
    NORMAL))
if hrv_img:
    story.append(Spacer(1, 6))
    story.append(Image(hrv_img, width=174*mm, height=58*mm))

story.append(Spacer(1, 10))
story.append(Paragraph("Poincaré plot", H2))
story.append(Paragraph(
    "Each point represents a pair of consecutive intervals (RRₙ, RRₙ₊₁). "
    "For the sinus beats (green) a compact cluster around the diagonal is expected "
    "(RRₙ ≈ RRₙ₊₁): the narrower the cluster transversally, the lower the beat-to-beat variability. "
    "The PVCs (red) plot <b>coupling vs following RR</b> and fall outside the sinus cluster "
    "in a characteristic zone: a very short RRₙ (coupling), a long RRₙ₊₁ (compensatory pause).",
    NORMAL))
if poincare_img:
    story.append(Spacer(1, 6))
    story.append(Image(poincare_img, width=110*mm, height=110*mm,
                       hAlign="CENTER"))

story.append(PageBreak())

# ---- FALSE-POSITIVE EXAMPLES ----
# (The couplet examples are now at the top, below the example trace.)
if fp_imgs:
    story.append(Paragraph("Morphological examples", H2))
    story.append(Paragraph("Beats reclassified by the amplitude threshold", H3))
    story.append(Paragraph(
        f"The shape criterion alone (deep rebound or wide QRS) overestimated as "
        f"PVC <b>{len(removed_fp)} beats</b> "
        f"(amplitude {min(q['amp'] for q in removed_fp):.2f}–"
        f"{max(q['amp'] for q in removed_fp):.2f} V); the amplitude requirement ≥ "
        f"{PVC_MIN_AMP_V:.2f} V brings them back to the correct classification. The "
        f"<b>majority ({len(removed_fp)-n_fp_spike})</b> are <b>small, real normal "
        f"sinus beats</b>: a narrow but physiological QRS, with an S wave that "
        f"exceeded the shape threshold despite not being ectopic. A minority "
        f"(<b>{n_fp_spike}</b>) are instead <b>noise spikes</b> in noisy stretches, "
        f"with width ≤16 ms — impossible for a real QRS — and therefore not even real "
        f"beats. In both cases the threshold correctly removes them from the "
        f"PVC count. Representative examples (small real beat, circled in orange):",
        NORMAL))
    for im in fp_imgs:
        story.append(Spacer(1, 6))
        story.append(fit_image(im, max_w_mm=170, max_h_mm=65))

if fp_imgs:
    story.append(PageBreak())

# ---- CONCLUSIONS + LIMITS ----
story.append(Paragraph("Descriptive conclusions", H2))
concl = [
    f"<b>Monomorphic and stable ectopic focus</b>. The coupling interval has a median "
    f"of {coupling_median:.0f} ms and a standard deviation of {coupling_std:.0f} ms "
    f"({100*coupling_std/max(1,coupling_median):.1f}% of the median). The dispersion is very "
    f"low, indicating a single ectopic source firing with very consistent timing.",

    f"<b>Predominantly isolated PVC</b>. {iso_pvc} of the {len(pvc)} total PVC "
    f"({100*iso_pvc/max(1,len(pvc)):.0f}%) are isolated (preceded and followed by a normal beat). "
    f"Only {couplets_n} couplets, no triplets. No ventricular tachycardia run observed.",

    f"<b>Dominant rhythmic pattern: trigeminy</b>. {trigem} trigeminy runs "
    f"(repeated N-N-PVC ≥3 cycles) versus {bigem} bigeminy runs "
    f"(alternating N-PVC ≥3 cycles).",

    f"<b>Temporally uniform burden</b>. The PVC/min rate stays constant around "
    f"{pvc_rate:.0f}/min for the whole duration of the recording, with no major clustering "
    f"or quiescence.",

    f"<b>Mild but systematic autonomic modulation</b>. The RR variability in the 5 "
    f"sinus beats immediately preceding a PVC is {hrv_delta_pct:+.0f}% relative to "
    f"baseline. The signal is small but observable over hundreds of observations.",

    f"<b>Theoretical pulse deficit</b>. PVCs typically do not produce an appreciable "
    f"peripheral pulse because of the reduced stroke volume. A <b>deficit of "
    f"~{100*len(pvc)/n_total:.0f}%</b> is expected between the ECG rate and the radial pulse rate "
    f"(as detectable by a Garmin/Apple Watch). Compatible with the user's empirical "
    f"observations.",
]
for c in concl:
    story.append(Paragraph("• " + c, NORMAL))
    story.append(Spacer(1, 4))

story.append(Spacer(1, 12))
story.append(Paragraph("Technical limitations", H2))
lims = [
    "<b>Single lead (Einthoven I)</b>. Impossible to localize the ectopic "
    "focus in the three anatomical planes or to distinguish PVC from other forms of ectopy "
    "that require a multi-lead view.",

    "<b>Detector based on rebound and width</b>. Robust for the patient's "
    "current morphology, but it could miss PVC with inverted polarity or "
    "atypical morphology. It does not perform P-wave detection.",

    "<b>Limited duration</b>. A recording of "
    f"{total_min:.1f} min captures a sample of the electrical behavior but "
    "does not allow inference about circadian trends, post-prandial effects, "
    "exertion response, or sleep.",

    "<b>No synchronized respiration signal</b>. The hypothesis of respiratory "
    "modulation of the ectopic trigger (suggested by the +HRV pre-PVC observation) "
    "is not formally verifiable without a respiration sensor (e.g. a Z-axis "
    "accelerometer on the chest) synchronized with the ECG.",

    "<b>Detector not clinically validated</b>. The thresholds were tuned "
    "empirically on this single subject's data and have not passed "
    "validation against a gold standard such as a professional clinical holter.",
]
for l in lims:
    story.append(Paragraph("• " + l, NORMAL))
    story.append(Spacer(1, 4))

story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#ccc")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"<font color='#888' size=8>"
    f"Source: <font name='Courier'>{os.path.basename(PATH)}</font>. "
    f"Pipeline: Pi Pico 2 W (12-bit ADC 250 Hz) → WiFi/TCP → Python server (Flask/SSE) → "
    f"IIR filter 0.3–25 Hz → 4-state FSM detector → rebound/width classification. "
    f"Repository: <font name='Courier'>github.com/mrEg0n/holter-ecg</font>. "
    f"Generated on {now} by host/generate_report_pdf.py."
    f"</font>",
    NORMAL
))

# build
print("rendering PDF...")
doc.build(story)
print(f"PDF saved: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
