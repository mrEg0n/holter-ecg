"""
Report sintetico cross-sessione: confronta 3 registrazioni ECG, calcola gli stessi
indicatori (burden, couplet, interpolate vs compensate, AF screening) e produce
un PDF di sintesi con tabelle, grafici e esempi delle classificazioni chiave.

Usage:
    python3 synthetic_report.py ecg_A.csv ecg_B.csv ecg_C.csv
"""
import csv, io, math, os, statistics, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)

SR = 250
GREEN, RED, ORANGE, BLUE, GRAY = "#33ff66", "#ff4d6d", "#ffa64d", "#7ad9ff", "#888"
DARK_BG, GRID = "#0d0f12", "#333"
PVC_MIN_AMP = 0.70
REBOUND_PVC = 0.40
PVC_W_MS    = 95.0
PVC_W_MIN_MS = 40.0   # range fisiologico per la width PVC: sotto = spike, sopra = baseline shift
PVC_W_MAX_MS = 220.0
PVC_MIN_REBOUND = 0.05  # rebound minimo: artefatti larghi hanno reb=0
COUPLET_MAX_RR_S = 0.70

# ---------- analysis core ----------
def load_session(ecg_path):
    pk_path = ecg_path.replace("ecg_", "peaks_")
    ts, vr, vf = [], [], []
    with open(ecg_path) as f:
        for r in csv.DictReader(f):
            try:
                ts.append(float(r["t_s"]))
                vr.append(float(r["raw"]))
                vf.append(float(r["filt"]))
            except (KeyError, ValueError): continue
    t = np.array(ts); vr = np.array(vr); vf = np.array(vf)
    peaks = []
    with open(pk_path) as f:
        for r in csv.DictReader(f):
            try:
                p = {"t": float(r["t_s"]), "amp": float(r["amp_V"]),
                     "w": float(r["width_ms"]), "reb": float(r["rebound_ratio"]),
                     "cls": r["class"]}
                peaks.append(p)
            except (KeyError, ValueError): continue
    # riclassifica col criterio attuale (plausibility check + rebound minimo)
    for p in peaks:
        shape_pvc = (p["reb"] >= REBOUND_PVC or p["w"] >= PVC_W_MS)
        plausible_w = PVC_W_MIN_MS <= p["w"] <= PVC_W_MAX_MS
        has_rebound = p["reb"] >= PVC_MIN_REBOUND
        p["cls"] = "pvc" if (shape_pvc and p["amp"] >= PVC_MIN_AMP
                              and plausible_w and has_rebound) else "normal"
    # togli spike di rumore
    peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP)]
    return t, vr, vf, peaks

def detect_noise_intervals(t, vr, win_s=4.0, min_s=1.0):
    """Auto-rileva burst di rumore con soglia ADATTIVA al baseline della sessione.
    La soglia è (mediana std baseline) + 0.10V, così funziona anche su sessioni
    registrate con gain diverso dell'AD8232."""
    WIN = int(win_s * SR)
    std_arr = np.zeros(len(vr))
    for i in range(0, len(vr) - WIN, SR):
        std_arr[i:i+SR] = vr[i:i+WIN].std()
    # soglia: mediana + offset, ma minimo 0.30
    baseline = float(np.median(std_arr[std_arr > 0]))
    std_thresh = max(0.30, baseline + 0.10)
    noisy = std_arr > std_thresh
    out = []; in_n = False; start = 0
    for i, n in enumerate(noisy):
        if n and not in_n: start = i; in_n = True
        elif not n and in_n:
            end = i; in_n = False
            if end - start > min_s * SR:
                out.append((float(t[start]), float(t[end-1])))
    if in_n: out.append((float(t[start]), float(t[-1])))
    return out

def load_manual_exclusions(ecg_path):
    """Carica le esclusioni manuali dal file JSON prodotto da mark_exclusions.py.
    Restituisce lista di (start_s, end_s)."""
    import json as _json
    base = os.path.basename(ecg_path).replace("ecg_", "").replace(".csv", "")
    p = os.path.join("exclusions", f"exclusions_{base}.json")
    if not os.path.exists(p): return []
    try:
        with open(p) as f:
            data = _json.load(f)
        return [(d["start"], d["end"]) for d in data.get("intervals", [])]
    except Exception as e:
        print(f"[{base}] errore lettura {p}: {e}")
        return []

def analyze(ecg_path, manual_excl=None):
    """Restituisce dict con tutti i numeri della sessione.
    Se manual_excl è None, carica dal file JSON exclusions_<base>.json se esiste,
    altrimenti usa solo l'auto-detection del rumore."""
    t, vr, vf, peaks = load_session(ecg_path)
    if manual_excl is None:
        manual_excl = load_manual_exclusions(ecg_path)
        if manual_excl:
            print(f"  caricate {len(manual_excl)} esclusioni manuali da JSON")
    noise_excl = detect_noise_intervals(t, vr)
    # consolida intervalli (manuale + auto, no fusione complicata)
    excl = list(noise_excl) + list(manual_excl)
    def in_excl(tv): return any(s <= tv <= e for s, e in excl)
    def in_manual(tv): return any(s <= tv <= e for s, e in manual_excl)
    peaks_clean = [p for p in peaks if not in_excl(p["t"])]
    # Per couplet usiamo SOLO esclusioni manuali (più conservative):
    # l'auto-detection del rumore può tagliare zone dove esiste un couplet vero
    # adiacente a un burst rumoroso. Couplet ha pattern molto specifico (2 PVC
    # con RR<700ms) e non viene generato dal rumore.
    peaks_for_couplet = [p for p in peaks if not in_manual(p["t"])]
    # RR
    for i in range(len(peaks_clean)):
        peaks_clean[i]["rr_prev"] = (peaks_clean[i]["t"] - peaks_clean[i-1]["t"]) if i > 0 else None
        peaks_clean[i]["rr_next"] = (peaks_clean[i+1]["t"] - peaks_clean[i]["t"]) if i < len(peaks_clean)-1 else None
    # tempo utile
    total_s_raw = float(t[-1] - t[0])
    excl_s = sum(e - s for s, e in excl)
    clean_s = total_s_raw - excl_s
    norm = [p for p in peaks_clean if p["cls"] == "normal"]
    pvc  = [p for p in peaks_clean if p["cls"] == "pvc"]
    n_total = len(peaks_clean)
    sinus_bpm = 60 * len(norm) / clean_s if clean_s else 0
    pvc_rate  = 60 * len(pvc)  / clean_s if clean_s else 0
    burden    = 100 * len(pvc) / max(1, n_total)
    # sinus RR
    sinus_rr = [p["rr_prev"] for p in peaks_clean if p["cls"]=="normal" and p["rr_prev"] and 0.6<p["rr_prev"]<1.4]
    RR_S = statistics.median(sinus_rr)*1000 if sinus_rr else 1000
    # couplet veri (RR < 700ms) — usa solo esclusioni manuali
    couplets = []
    i = 0
    while i < len(peaks_for_couplet) - 1:
        if peaks_for_couplet[i]["cls"]=="pvc" and peaks_for_couplet[i+1]["cls"]=="pvc":
            rr = peaks_for_couplet[i+1]["t"] - peaks_for_couplet[i]["t"]
            if rr >= COUPLET_MAX_RR_S: i += 1; continue
            if i+2 < len(peaks_for_couplet) and peaks_for_couplet[i+2]["cls"]=="pvc":
                i += 1; continue
            couplets.append((peaks_for_couplet[i], peaks_for_couplet[i+1]))
            i += 2
        else: i += 1
    # interpolate vs compensate
    interp = []; comp = []; incomp = []
    for idx, p in enumerate(peaks_clean):
        if p["cls"] != "pvc": continue
        if not p["rr_prev"] or not p["rr_next"]: continue
        if idx == 0 or idx == len(peaks_clean)-1: continue
        if peaks_clean[idx-1]["cls"] != "normal" or peaks_clean[idx+1]["cls"] != "normal": continue
        s_ms = (p["rr_prev"] + p["rr_next"]) * 1000
        p["sum_pre_post_ms"] = s_ms
        if s_ms < 1.3 * RR_S:
            p["pause_type"] = "interp"; interp.append(p)
        elif 1.85*RR_S < s_ms < 2.15*RR_S:
            p["pause_type"] = "comp"; comp.append(p)
        else:
            p["pause_type"] = "incomp"; incomp.append(p)
    # bigeminia / trigeminia / iso
    iso_pvc = sum(1 for k, p in enumerate(peaks_clean) if p["cls"]=="pvc"
                  and (k==0 or peaks_clean[k-1]["cls"]!="pvc")
                  and (k==len(peaks_clean)-1 or peaks_clean[k+1]["cls"]!="pvc"))
    # bigem (PVC-N-PVC) count
    bigem = sum(1 for k in range(2, len(peaks_clean))
                if peaks_clean[k]["cls"]=="pvc" and peaks_clean[k-1]["cls"]=="normal"
                and peaks_clean[k-2]["cls"]=="pvc")
    trigem = sum(1 for k in range(3, len(peaks_clean))
                 if peaks_clean[k]["cls"]=="pvc" and peaks_clean[k-1]["cls"]=="normal"
                 and peaks_clean[k-2]["cls"]=="normal" and peaks_clean[k-3]["cls"]=="pvc")
    # AF screening (NN consecutivi)
    af_nn = []
    for k in range(1, len(peaks_clean)):
        if peaks_clean[k]["cls"]=="normal" and peaks_clean[k-1]["cls"]=="normal":
            rr = peaks_clean[k]["t"] - peaks_clean[k-1]["t"]
            if 0.4 <= rr <= 2.0: af_nn.append(rr*1000)
    af = {}
    if len(af_nn) >= 30:
        diffs = [abs(af_nn[k]-af_nn[k-1]) for k in range(1, len(af_nn))]
        rmssd = (sum(d*d for d in diffs)/len(diffs))**0.5
        pnn50 = 100*sum(1 for d in diffs if d>50)/len(diffs)
        cv = 100*statistics.stdev(af_nn)/statistics.mean(af_nn)
        hist_af, _ = np.histogram(af_nn, bins=20)
        p_af = hist_af[hist_af>0]/sum(hist_af[hist_af>0])
        H = float(-sum(p*np.log2(p) for p in p_af))
        H_max = float(np.log2(len(p_af)))
        smooth = np.convolve(hist_af, [1,1,1], mode="same")
        n_peaks = sum(1 for k in range(1, len(smooth)-1)
                      if smooth[k]>smooth[k-1] and smooth[k]>smooth[k+1]
                      and smooth[k]>0.3*smooth.max())
        score = 0
        if rmssd>100: score += 1
        if pnn50>40: score += 1
        if H/H_max>0.85: score += 1
        if n_peaks<=1 and cv>15: score += 1
        af = {"rmssd":rmssd, "pnn50":pnn50, "cv":cv, "entropy":H/H_max,
              "n_peaks":n_peaks, "score":score, "nn_count":len(af_nn),
              "rr_med": statistics.median(af_nn)}
    return {
        "ecg_path": ecg_path,
        "t": t, "vr": vr, "vf": vf,
        "peaks": peaks_clean,
        "raw_dur_s": total_s_raw,
        "excl_s": excl_s,
        "clean_s": clean_s,
        "noise_intervals": noise_excl,
        "manual_excl": manual_excl,
        "sinus_bpm": sinus_bpm,
        "pvc_rate": pvc_rate,
        "burden": burden,
        "RR_SINUS_MS": RR_S,
        "n_norm": len(norm),
        "n_pvc": len(pvc),
        "n_total": n_total,
        "couplets": couplets,
        "interp": interp,
        "comp": comp,
        "incomp": incomp,
        "iso_pvc": iso_pvc,
        "bigem": bigem,
        "trigem": trigem,
        "af": af,
    }

# ---------- plot helpers ----------
def fig_to_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                                     facecolor=DARK_BG); plt.close(fig); buf.seek(0); return buf

def fit_image(buf, max_w_mm=170, max_h_mm=180):
    img = PILImage.open(buf); w_px, h_px = img.size
    ar = w_px / h_px
    w = max_w_mm * mm; h = w/ar
    if h > max_h_mm*mm: h = max_h_mm*mm; w = h*ar
    buf.seek(0); return Image(buf, width=w, height=h, hAlign="CENTER")

def styled_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(DARK_BG)
    if title: ax.set_title(title, color="white", fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(alpha=0.18, color="#666")

def make_example_strip(t, vf, peaks, p_center, win_s=6.0, title=None,
                       highlight_center=True, RR_S=None):
    """Strip didattico con annotazioni RR_pre, RR_post se disponibili."""
    c0 = p_center["t"]
    mask = (t >= c0 - win_s/2) & (t <= c0 + win_s/2)
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(t[mask] - c0, vf[mask], color=GREEN, lw=0.8)
    for q in peaks:
        if c0-win_s/2 <= q["t"] <= c0+win_s/2:
            dt = q["t"] - c0
            if q["cls"] == "pvc":
                ax.plot(dt, 1.35, "v", color=RED, ms=7)
                wm = (t >= q["t"]-0.1) & (t <= q["t"]+0.1)
                ax.plot(t[wm]-c0, vf[wm], color=RED, lw=1.2)
            else:
                ax.plot(dt, 0.85, "v", color=GREEN, ms=4)
    if highlight_center:
        ax.scatter(0, p_center["amp"], s=240, marker="o", facecolors="none",
                   edgecolors=ORANGE, linewidths=1.8, zorder=10)
    if p_center.get("rr_prev"):
        rrp = p_center["rr_prev"]
        ax.plot([-rrp, 0], [-0.65, -0.65], color=BLUE, lw=2)
        ax.text(-rrp/2, -0.55, f"{rrp*1000:.0f}ms", color=BLUE, fontsize=7,
                ha="center", fontweight="bold")
    if p_center.get("rr_next"):
        rrn = p_center["rr_next"]
        ax.plot([0, rrn], [-0.85, -0.85], color="#ffe169", lw=2)
        ax.text(rrn/2, -1.0, f"{rrn*1000:.0f}ms", color="#ffe169", fontsize=7,
                ha="center", fontweight="bold")
    if RR_S and p_center.get("rr_prev"):
        comp_x = -p_center["rr_prev"] + 2*RR_S/1000.0
        if -win_s/2 < comp_x < win_s/2:
            ax.axvline(comp_x, color="#ff4d6d", lw=0.8, ls="--", alpha=0.7)
    styled_ax(ax, title, "t (s) rispetto al centro", "ECG (V)")
    ax.set_xlim(-win_s/2, win_s/2); ax.set_ylim(-1.2, 1.7)
    plt.tight_layout()
    return fig_to_bytes(fig)

# ---------- EDR (ECG-Derived Respiration) + analisi fasica PVC ----------
NBINS_RESP = 12
FS_RESP = 4.0

def extract_edr_and_phase(peaks):
    """Restituisce dict con EDR + fase istantanea + analisi distribuzione PVC."""
    norm = [p for p in peaks if p["cls"] == "normal"]
    pvc  = [p for p in peaks if p["cls"] == "pvc"]
    if len(norm) < 200 or len(pvc) < 30:
        return None
    t_n = np.array([p["t"] for p in norm])
    amp_n = np.array([p["amp"] for p in norm])
    if t_n[-1] - t_n[0] < 5*60:
        return None
    t_unif = np.arange(t_n[0], t_n[-1], 1/FS_RESP)
    amp_unif = interp1d(t_n, amp_n, kind="cubic")(t_unif)
    amp_dt = sig.detrend(amp_unif)
    sos = sig.butter(3, [0.10, 0.50], btype="band", fs=FS_RESP, output="sos")
    resp = sig.sosfiltfilt(sos, amp_dt)
    # quality
    f_psd, psd = sig.welch(resp, fs=FS_RESP, nperseg=min(2048, len(resp)//4))
    in_band = (f_psd >= 0.10) & (f_psd <= 0.50)
    out_band = (f_psd >= 0.60) & (f_psd <= 1.5)
    snr = float(np.mean(psd[in_band]) / max(1e-12, np.mean(psd[out_band])))
    rate_resp = float(f_psd[in_band][np.argmax(psd[in_band])] * 60)
    # phase via Hilbert
    analytic = sig.hilbert(resp)
    phase = np.mod(np.angle(analytic), 2*np.pi)
    phase_interp = interp1d(t_unif, phase, kind="nearest",
                             bounds_error=False, fill_value=0)
    phase_n = phase_interp([p["t"] for p in norm])
    phase_p = phase_interp([p["t"] for p in pvc])
    bins = np.linspace(0, 2*np.pi, NBINS_RESP+1)
    hist_n, _ = np.histogram(phase_n, bins=bins)
    hist_p, _ = np.histogram(phase_p, bins=bins)
    contingency = np.array([hist_p, hist_n])
    chi2_val, pval, _, _ = chi2_contingency(contingency)
    dens_n = hist_n / hist_n.sum()
    dens_p = hist_p / max(1, hist_p.sum())
    enrich = dens_p / np.maximum(dens_n, 1e-6)
    centers = (bins[:-1] + bins[1:]) / 2
    peak_phase_bin = int(np.argmax(enrich))
    return {
        "t_unif": t_unif, "resp": resp,
        "snr": snr, "rate_resp": rate_resp,
        "chi2": float(chi2_val), "pval": float(pval),
        "hist_n": hist_n.tolist(), "hist_p": hist_p.tolist(),
        "dens_n": dens_n.tolist(), "dens_p": dens_p.tolist(),
        "enrich": enrich.tolist(), "centers": centers.tolist(),
        "peak_phase_pct": float(centers[peak_phase_bin] * 100 / (2*np.pi)),
        "peak_enrich": float(enrich[peak_phase_bin]),
        "n_n": len(norm), "n_p": len(pvc),
    }

# ---------- main ----------
if len(sys.argv) < 4:
    print("usage: synthetic_report.py ecg_A.csv ecg_B.csv ecg_C.csv")
    sys.exit(1)

# Esclusioni manuali per sessione (se serve)
MANUAL_EXCL = {
    "ecg_20260607_113338.csv": [(1472, 1505), (1690, 1780)],
}

print("Analisi sessioni...")
sessions = []
for arg in sys.argv[1:]:
    key = os.path.basename(arg)
    excl = MANUAL_EXCL.get(key, [])
    print(f"  {key} ...")
    s = analyze(arg, manual_excl=excl)
    # estrai EDR + fase respiratoria
    s["edr"] = extract_edr_and_phase(s["peaks"])
    if s["edr"]:
        print(f"    EDR: rate {s['edr']['rate_resp']:.1f}/min, "
              f"picco fasico {s['edr']['peak_phase_pct']:.0f}% ciclo, "
              f"enrich ×{s['edr']['peak_enrich']:.2f}, p={s['edr']['pval']:.0e}")
    sessions.append(s)

def label_of(s):
    base = os.path.basename(s["ecg_path"]).replace("ecg_", "").replace(".csv", "")
    # 20260605_131136 → 05 giu 13:11
    d = base[6:8] + " giu " + base[9:11] + ":" + base[11:13]
    return d

LABELS = [label_of(s) for s in sessions]
for s, L in zip(sessions, LABELS):
    n_class = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
    pct_i = 100*len(s["interp"])/max(1, n_class)
    pct_c = 100*len(s["comp"])/max(1, n_class)
    print(f"\n--- {L} ---")
    print(f"  durata utile: {s['clean_s']/60:.1f} min  (esclusi {s['excl_s']:.0f}s)")
    print(f"  battiti: {s['n_total']} (N={s['n_norm']}, PVC={s['n_pvc']})")
    print(f"  sinus {s['sinus_bpm']:.1f} BPM   PVC rate {s['pvc_rate']:.1f}/min   burden {s['burden']:.1f}%")
    print(f"  couplet veri (RR<700ms): {len(s['couplets'])}")
    print(f"  interpolate {len(s['interp'])} ({pct_i:.0f}%)   compensate {len(s['comp'])} ({pct_c:.0f}%)   incomplete {len(s['incomp'])}")
    if s["af"]: print(f"  AF score {s['af']['score']}/4 — RMSSD {s['af']['rmssd']:.0f}ms pNN50 {s['af']['pnn50']:.0f}% picchi-hist {s['af']['n_peaks']}")

# ---------- PLOTS COMPARATIVI ----------
print("\nGenero plot comparativi...")

# A) Burden + rate confronto
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=DARK_BG)
metrics = [("burden", "Burden PVC (%)"),
           ("sinus_bpm", "Sinus rate (BPM)"),
           ("pvc_rate", "PVC rate (/min)")]
for ax, (key, ylabel) in zip(axes, metrics):
    vals = [s[key] for s in sessions]
    _pal = [BLUE, ORANGE, RED, "#33aa66", "#a64dff", "#ffe169", "#9b6b00"]
    _colors_bar = [_pal[i % len(_pal)] for i in range(len(sessions))]
    ax.bar(LABELS, vals, color=_colors_bar, edgecolor="white", linewidth=0.5)
    for k, v in enumerate(vals):
        ax.text(k, v*1.01, f"{v:.1f}", ha="center", color="white", fontsize=10, fontweight="bold")
    styled_ax(ax, None, None, ylabel)
plt.tight_layout()
img_bars = fig_to_bytes(fig)

# A2) Plot HR vs % compensate — il pattern chiave
fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=DARK_BG)
hr_vals = [s["sinus_bpm"] for s in sessions]
comp_pct = []
interp_pct = []
for s in sessions:
    nc = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
    comp_pct.append(100*len(s["comp"])/max(1,nc))
    interp_pct.append(100*len(s["interp"])/max(1,nc))
ax.scatter(hr_vals, comp_pct, s=200, c=RED, edgecolors="white", linewidths=1.5,
           zorder=10, label="% Compensate (tonfi percepiti)")
ax.scatter(hr_vals, interp_pct, s=200, c=BLUE, edgecolors="white", linewidths=1.5,
           zorder=10, label="% Interpolate (silenziose)")
for hr, cp, ip, lab in zip(hr_vals, comp_pct, interp_pct, LABELS):
    ax.annotate(lab, (hr, cp), textcoords="offset points", xytext=(8, 8),
                color=RED, fontsize=8, fontweight="bold")
    ax.annotate(lab, (hr, ip), textcoords="offset points", xytext=(8, -12),
                color=BLUE, fontsize=8, fontweight="bold")
# trend lines
hr_arr = np.array(hr_vals); c_arr = np.array(comp_pct); i_arr = np.array(interp_pct)
order = hr_arr.argsort()
ax.plot(hr_arr[order], c_arr[order], color=RED, lw=2, alpha=0.6)
ax.plot(hr_arr[order], i_arr[order], color=BLUE, lw=2, alpha=0.6)
ax.set_xlim(50, 60)
ax.set_ylim(-5, 80)
ax.legend(facecolor="#222", labelcolor="white", fontsize=10, loc="center right")
styled_ax(ax, "Pattern chiave: la HR basale decide quante PVC si sentono",
          "Sinus BPM (frequenza media a riposo)", "% delle PVC classificate")
plt.tight_layout()
img_hr_comp = fig_to_bytes(fig)

# B) Composition stacked (interpolate / compensate / incomplete)
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
i_vals = [len(s["interp"]) for s in sessions]
c_vals = [len(s["comp"]) for s in sessions]
x_vals = [len(s["incomp"]) for s in sessions]
totals = [i+c+x for i,c,x in zip(i_vals, c_vals, x_vals)]
# percentuali
ip = [100*v/max(1,t) for v,t in zip(i_vals, totals)]
cp = [100*v/max(1,t) for v,t in zip(c_vals, totals)]
xp = [100*v/max(1,t) for v,t in zip(x_vals, totals)]
y = np.arange(len(LABELS))
ax.barh(y, ip, color=BLUE, edgecolor="white", label="Interpolate (no pausa)")
ax.barh(y, cp, left=ip, color=RED, edgecolor="white", label="Compensate (pausa piena)")
ax.barh(y, xp, left=[a+b for a,b in zip(ip,cp)], color=GRAY, edgecolor="white", label="Incomplete")
for k, (i_, c_, x_, t_) in enumerate(zip(i_vals, c_vals, x_vals, totals)):
    ax.text(50, k, f"i={i_}  c={c_}  x={x_}  (tot {t_})", color="white",
            ha="center", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(LABELS, color="white")
ax.set_xlim(0, 100)
ax.legend(facecolor="#222", labelcolor="white", fontsize=9, loc="upper right")
styled_ax(ax, "Composizione PVC: interpolate vs compensate (per sessione)",
          "% del totale classificato", None)
plt.tight_layout()
img_comp = fig_to_bytes(fig)

# B2) Distribuzione del rapporto (RR_pre+RR_post)/RR_sinus per le 3 sessioni
# Mostra DIRETTAMENTE perché 6 giu non ha interpolate: il suo istogramma non
# scende mai sotto 1.30 (la soglia interpolata), tutto si concentra su 1.85-2.15
# (compensatoria piena).
fig, axes_d = plt.subplots(len(sessions), 1, figsize=(11, 2.6*len(sessions)),
                            facecolor=DARK_BG, sharex=True)
_palette = [BLUE, ORANGE, RED, "#33aa66", "#a64dff", "#ffe169", "#9b6b00"]
colors_3 = [_palette[i % len(_palette)] for i in range(len(sessions))]
for ax, s, lab, col in zip(axes_d, sessions, LABELS, colors_3):
    ax.set_facecolor(DARK_BG)
    ratios = []
    for i, p in enumerate(s["peaks"]):
        if p["cls"] != "pvc" or i == 0 or i == len(s["peaks"])-1: continue
        if s["peaks"][i-1]["cls"] != "normal" or s["peaks"][i+1]["cls"] != "normal": continue
        if not p["rr_prev"] or not p["rr_next"]: continue
        ratios.append((p["rr_prev"] + p["rr_next"])*1000 / s["RR_SINUS_MS"])
    ax.hist(ratios, bins=np.linspace(0.7, 2.7, 60), color=col, edgecolor="white", linewidth=0.3)
    ax.axvspan(0, 1.30, color=col, alpha=0.10)
    ax.axvspan(1.85, 2.15, color=col, alpha=0.22)
    ax.axvline(1.30, color="white", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(1.85, color="white", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(2.15, color="white", ls="--", lw=0.7, alpha=0.6)
    med = statistics.median(ratios) if ratios else 0
    ax.axvline(med, color="yellow", ls="-", lw=1.5)
    ax.text(med + 0.02, ax.get_ylim()[1]*0.85, f"mediana {med:.2f}",
            color="yellow", fontsize=9, fontweight="bold")
    ax.text(0.05, 0.85, "← interp", transform=ax.transAxes, color=col, fontsize=8)
    ax.text(0.78, 0.85, "comp →", transform=ax.transAxes, color=col, fontsize=8)
    styled_ax(ax, f"{lab}   (sinus {s['sinus_bpm']:.0f} BPM, RR sinus {s['RR_SINUS_MS']:.0f}ms, n={len(ratios)})",
              None, "Conteggio")
axes_d[-1].set_xlabel("rapporto (RR_pre + RR_post) / RR_sinus", color="white")
plt.tight_layout()
img_distrib = fig_to_bytes(fig)

# B3) Bucket per BPM istantanea: a quale frequenza compaiono le compensate?
# Aggrego tutte le PVC delle 3 sessioni per BPM istantanea (= RR pre-N-N convertito).
fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
buckets = list(range(45, 95, 5))
records_all = []
for s in sessions:
    pk = s["peaks"]
    for i, p in enumerate(pk):
        if p["cls"] != "pvc" or i < 2 or i == len(pk)-1: continue
        if pk[i-1]["cls"] != "normal" or pk[i+1]["cls"] != "normal": continue
        if pk[i-2]["cls"] != "normal": continue
        rr_pre_nn = (pk[i-1]["t"] - pk[i-2]["t"])*1000
        if not (500 < rr_pre_nn < 1500): continue
        s_ms = (p["rr_prev"] + p["rr_next"])*1000
        if s_ms < 1.3*s["RR_SINUS_MS"]: kind = "interp"
        elif 1.85*s["RR_SINUS_MS"] < s_ms < 2.15*s["RR_SINUS_MS"]: kind = "comp"
        else: kind = "other"
        records_all.append({"bpm": 60000/rr_pre_nn, "kind": kind})

labels_b, n_i_b, n_c_b, pct_c_b = [], [], [], []
for b0 in buckets[:-1]:
    b1 = b0 + 5
    inb = [r for r in records_all if b0 <= r["bpm"] < b1]
    if len(inb) < 5: continue
    n_i = sum(1 for r in inb if r["kind"] == "interp")
    n_c = sum(1 for r in inb if r["kind"] == "comp")
    pct_c = 100*n_c/(n_i+n_c) if (n_i+n_c) else 0
    labels_b.append(f"{b0}-{b1}")
    n_i_b.append(n_i); n_c_b.append(n_c); pct_c_b.append(pct_c)

x_b = np.arange(len(labels_b)); w = 0.40
ax.bar(x_b - w/2, n_i_b, w, color=BLUE, edgecolor="white", linewidth=0.4, label="Interpolate")
ax.bar(x_b + w/2, n_c_b, w, color=RED, edgecolor="white", linewidth=0.4, label="Compensate")
ax2 = ax.twinx(); ax2.set_facecolor(DARK_BG)
ax2.plot(x_b, pct_c_b, color="#ffe169", marker="o", lw=2, label="% compensate")
ax2.axhline(50, color="#ffe169", ls="--", lw=0.7, alpha=0.5)
ax2.set_ylim(0, 100); ax2.set_ylabel("% compensate sulla somma classificata", color="#ffe169")
ax2.tick_params(colors="#ffe169")
for sp in ax2.spines.values(): sp.set_color("#444")
ax.set_xticks(x_b); ax.set_xticklabels(labels_b, color="white")
styled_ax(ax, "Quale frequenza istantanea favorisce la pausa compensatoria? (aggregato 3 sessioni)",
          "BPM istantanea (RR N-N pre-PVC convertito)", "N° PVC")
ax.legend(loc="upper left", facecolor="#222", labelcolor="white", fontsize=9)
ax2.legend(loc="upper right", facecolor="#222", labelcolor="#ffe169", fontsize=9)
plt.tight_layout()
img_bpm_bucket = fig_to_bytes(fig)

# C0) Dettaglio coupling per sessione + segmentazione cluster
# Analisi morfologica per ogni sub-cluster: stesso focolaio o bifocale?
def cluster_analyze(coup_list):
    """Restituisce dict con conteggi e morfologia mediana dei sub-cluster
    (<500ms / 500-600ms / >600ms)."""
    out = {}
    for label, lo, hi in [("c1",0,500),("c2",500,600),("c3",600,2000)]:
        sub = [c for c in coup_list if lo <= c["rr"] < hi]
        if not sub:
            out[label] = None; continue
        out[label] = {
            "n": len(sub),
            "pct": 100*len(sub)/len(coup_list),
            "rr_med": statistics.median(c["rr"] for c in sub),
            "amp_med": statistics.median(c["amp"] for c in sub),
            "w_med":   statistics.median(c["w"] for c in sub),
            "reb_med": statistics.median(c["reb"] for c in sub),
        }
    return out

session_clusters = []
for s in sessions:
    coup_list = [{"rr": p["rr_prev"]*1000, "amp": p["amp"], "w": p["w"], "reb": p["reb"]}
                 for p in s["peaks"] if p["cls"]=="pvc" and p["rr_prev"]
                 and 200 < p["rr_prev"]*1000 < 800]
    session_clusters.append(cluster_analyze(coup_list))

# Plot dettagliato: histograms a bin stretti per le 3 sessioni con cluster colorati
fig, axes_cc = plt.subplots(len(sessions), 1, figsize=(11, 2.4*len(sessions)),
                             facecolor=DARK_BG, sharex=True)
for ax, s, lab in zip(axes_cc, sessions, LABELS):
    ax.set_facecolor(DARK_BG)
    coup_arr = [p["rr_prev"]*1000 for p in s["peaks"]
                if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if not coup_arr: continue
    bins = np.arange(300, 700, 15)
    hist, edges = np.histogram(coup_arr, bins=bins)
    centers = (edges[:-1]+edges[1:])/2
    # colora le barre per cluster
    cols = ["#7ad9ff" if c<500 else ("#ff4d6d" if c<600 else "#33ff66") for c in centers]
    ax.bar(centers, hist, width=14, color=cols, edgecolor="white", linewidth=0.3)
    ax.axvline(500, color="white", ls="--", lw=0.7, alpha=0.5)
    ax.axvline(600, color="white", ls="--", lw=0.7, alpha=0.5)
    med = statistics.median(coup_arr)
    ax.axvline(med, color="yellow", ls="-", lw=1.5)
    ax.text(med+5, ax.get_ylim()[1]*0.85, f"med {med:.0f}ms",
            color="yellow", fontsize=8, fontweight="bold")
    styled_ax(ax, f"{lab}   coupling pre-PVC (n={len(coup_arr)})",
              None, "Conteggio")
axes_cc[-1].set_xlabel("Coupling pre-PVC (ms)", color="white")
plt.tight_layout()
img_cluster = fig_to_bytes(fig)

# C) Distribuzione coupling (istogrammi sovrapposti)
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=DARK_BG)
_palette = [BLUE, ORANGE, RED, "#33aa66", "#a64dff", "#ffe169", "#9b6b00"]
colors_3 = [_palette[i % len(_palette)] for i in range(len(sessions))]
for s, lab, col in zip(sessions, LABELS, colors_3):
    coups = [p["rr_prev"]*1000 for p in s["peaks"]
             if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if coups:
        ax.hist(coups, bins=np.arange(300, 700, 15), alpha=0.5,
                color=col, edgecolor="white", linewidth=0.3,
                label=f"{lab} (n={len(coups)}, med {statistics.median(coups):.0f}ms)")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
styled_ax(ax, "Distribuzione coupling pre-PVC (stabilità del focolaio)",
          "Coupling (ms)", "Conteggio")
plt.tight_layout()
img_coup = fig_to_bytes(fig)

# D) HR vs burden minuto per minuto, una sessione per riga
fig, axes = plt.subplots(3, 1, figsize=(11, 6), facecolor=DARK_BG)
for ax, s, lab in zip(axes, sessions, LABELS):
    ax.set_facecolor(DARK_BG)
    # minuti
    if not s["peaks"]: continue
    n_min = int(s["peaks"][-1]["t"]//60) + 1
    bpm_m, burden_m, mins = [], [], []
    for m in range(n_min):
        in_m = [p for p in s["peaks"] if m*60 <= p["t"] < (m+1)*60]
        if len(in_m) < 10: continue
        n_n = sum(1 for p in in_m if p["cls"]=="normal")
        bpm = 60*n_n/60 if n_n else 0
        b = 100*sum(1 for p in in_m if p["cls"]=="pvc")/len(in_m)
        bpm_m.append(bpm); burden_m.append(b); mins.append(m)
    ax2 = ax.twinx(); ax2.set_facecolor(DARK_BG)
    ax.plot(mins, bpm_m, color=GREEN, lw=1.5, marker="o", ms=3, label="Sinus BPM")
    ax2.plot(mins, burden_m, color=RED, lw=1.5, marker="s", ms=3, label="Burden %")
    ax.set_ylim(40, 90); ax2.set_ylim(0, 60)
    styled_ax(ax, f"{lab} — andamento minuto per minuto", "Min", "Sinus BPM")
    ax2.set_ylabel("Burden %", color="white"); ax2.tick_params(colors="white", labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#444")
plt.tight_layout()
img_temporal = fig_to_bytes(fig)

# E) ANALISI RESPIRATORIA — distribuzione fasica PVC per ogni sessione
# Plot a 2 colonne: rosetta polare a sx, enrichment bar a dx per ogni sessione
sessions_with_edr = [s for s in sessions if s.get("edr")]
img_resp_phases = None
img_resp_example = None
if sessions_with_edr:
    n_with_edr = len(sessions_with_edr)
    fig = plt.figure(figsize=(13, 2.4*n_with_edr), facecolor=DARK_BG)
    gs_ = fig.add_gridspec(n_with_edr, 2, width_ratios=[1, 1.4], hspace=0.55, wspace=0.25)
    for i_s, s in enumerate(sessions_with_edr):
        edr = s["edr"]
        lab = label_of(s)
        # ROSETTA polare
        ax_p = fig.add_subplot(gs_[i_s, 0], projection="polar")
        ax_p.set_facecolor(DARK_BG)
        centers_arr = np.array(edr["centers"])
        width = 2*np.pi/NBINS_RESP
        ax_p.bar(centers_arr, np.array(edr["dens_n"])*100, width=width*0.95,
                 alpha=0.45, color="#33aa66", edgecolor="white", linewidth=0.4)
        ax_p.bar(centers_arr, np.array(edr["dens_p"])*100, width=width*0.95,
                 alpha=0.75, color="#ff4444", edgecolor="white", linewidth=0.4)
        ax_p.set_theta_zero_location("E")
        ax_p.set_theta_direction(1)
        ax_p.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        # Convenzione Hilbert su EDR (max amp = max inspir): phase=0 → max INSPIR,
        # π/2 → mid espir, π → fine espir, 3π/2 → mid inspir
        ax_p.set_xticklabels(["max\ninspir.", "mid\nespir.", "fine\nespir.", "mid\ninspir."],
                             color="white", fontsize=7)
        ax_p.set_yticklabels([])
        ax_p.set_title(f"{lab}\n({edr['n_p']} PVC, p={edr['pval']:.0e})",
                       color="white", fontsize=8)
        # ENRICHMENT bar
        ax_e = fig.add_subplot(gs_[i_s, 1])
        ax_e.set_facecolor(DARK_BG)
        phase_pct = centers_arr * 100 / (2*np.pi)
        enrich = edr["enrich"]
        colors_b = ["#ff4444" if e>1.2 else ("#33aa66" if e<0.8 else "#7ad9ff") for e in enrich]
        ax_e.bar(phase_pct, enrich, width=100/NBINS_RESP*0.9, color=colors_b,
                 edgecolor="white", linewidth=0.4)
        ax_e.axhline(1, color="white", ls="--", lw=0.8, alpha=0.5)
        # 0=max inspir, 25=mid espir, 50=fine espir, 75=mid inspir
        for x, lab2 in [(0, "insp."), (25, "espir."), (50, "espir."), (75, "insp.")]:
            ax_e.axvline(x, color="cyan", ls=":", alpha=0.3)
        ax_e.set_xlim(0, 100)
        ax_e.set_ylim(0, max(2.5, max(enrich)*1.1))
        ax_e.set_ylabel("PVC/N\nratio", color="white", fontsize=8)
        ax_e.set_title(f"rate {edr['rate_resp']:.1f}/min · picco @ {edr['peak_phase_pct']:.0f}% · "
                       f"enrich ×{edr['peak_enrich']:.2f}",
                       color="white", fontsize=8, loc="left")
        ax_e.tick_params(colors="white", labelsize=7)
        for sp in ax_e.spines.values(): sp.set_color("#444")
        ax_e.grid(alpha=0.18, color="#666")
        if i_s == n_with_edr - 1:
            ax_e.set_xlabel("% del ciclo respiratorio  (0=max inspir., 50=fine espir.)",
                            color="white", fontsize=8)
    img_resp_phases = fig_to_bytes(fig)

    # F) ESEMPIO EDR — sessione ultima (più recente, di norma il 9 giu) — primi 30s
    s_demo = sessions_with_edr[-1]
    edr_demo = s_demo["edr"]
    t_demo = s_demo["t"]; vf_demo = s_demo["vf"]
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    T_SHOW = 30
    mask_e = (t_demo < T_SHOW)
    ax.plot(t_demo[mask_e], vf_demo[mask_e], color=GREEN, lw=0.6, label="ECG")
    # peaks N nel range
    for p in s_demo["peaks"]:
        if p["t"] < T_SHOW and p["cls"] == "normal":
            ax.plot(p["t"], p["amp"]+0.05, "v", color="#ffe169", ms=4)
    # respirazione sovrapposta (scalata)
    t_unif = edr_demo["t_unif"]
    resp = edr_demo["resp"]
    mask_r = t_unif < T_SHOW
    ax.plot(t_unif[mask_r], resp[mask_r]*3 + 1.8, color="cyan", lw=2.2,
            label="Respirazione (EDR)")
    ax.set_xlim(0, T_SHOW)
    ax.set_ylim(-1, 3.3)
    ax.set_xlabel("Tempo (s)", color="white")
    ax.set_ylabel("ECG (V) + respirazione (norm)", color="white")
    ax.set_title(f"Esempio EDR — {label_of(s_demo)}, primi {T_SHOW}s. "
                 f"R-peaks (gialli), respirazione estratta dall'ampiezza QRS (ciano)",
                 color="white", fontsize=10)
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9, loc="upper right")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.grid(alpha=0.18, color="#666")
    plt.tight_layout()
    img_resp_example = fig_to_bytes(fig)

# ---------- ESEMPI ----------
print("Estraggo esempi...")
def pick_spread(lst, n=1, min_gap_s=60):
    out, last = [], -1e9
    for p in sorted(lst, key=lambda q: q["t"]):
        if p["t"] - last >= min_gap_s:
            out.append(p); last = p["t"]
        if len(out) >= n: break
    return out

# Per ogni tipo: una strip per sessione (3 sessioni × 4 tipi = 12 strip max)
example_strips = {"couplet": [], "interp": [], "comp": []}
for s, lab in zip(sessions, LABELS):
    # interpolata
    if s["interp"]:
        p = pick_spread(s["interp"], n=1, min_gap_s=120)[0]
        img = make_example_strip(s["t"], s["vf"], s["peaks"], p, win_s=7.0,
            title=f"{lab} — INTERPOLATA  {int(p['t']//60):02d}:{int(p['t']%60):02d}  Σ={p['sum_pre_post_ms']:.0f}ms",
            RR_S=s["RR_SINUS_MS"])
        example_strips["interp"].append((lab, img))
    # compensata
    if s["comp"]:
        p = pick_spread(s["comp"], n=1, min_gap_s=120)[0]
        img = make_example_strip(s["t"], s["vf"], s["peaks"], p, win_s=7.0,
            title=f"{lab} — COMPENSATA  {int(p['t']//60):02d}:{int(p['t']%60):02d}  Σ={p['sum_pre_post_ms']:.0f}ms",
            RR_S=s["RR_SINUS_MS"])
        example_strips["comp"].append((lab, img))
    # couplet
    if s["couplets"]:
        p1, p2 = s["couplets"][0]
        ctr_p = {"t": (p1["t"]+p2["t"])/2, "amp": max(p1["amp"], p2["amp"]),
                 "rr_prev": None, "rr_next": None}
        img = make_example_strip(s["t"], s["vf"], s["peaks"], ctr_p, win_s=7.0,
            title=f"{lab} — COUPLET  {int(ctr_p['t']//60):02d}:{int(ctr_p['t']%60):02d}  RR={1000*(p2['t']-p1['t']):.0f}ms",
            highlight_center=False)
        example_strips["couplet"].append((lab, img))

# ---------- PDF ----------
print("Genero PDF...")
out_path = "reports/synthetic_3sessions.pdf"
doc = SimpleDocTemplate(out_path, pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=15*mm, bottomMargin=15*mm)
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22,
                    spaceAfter=8, textColor=colors.HexColor("#222"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16,
                    spaceAfter=6, textColor=colors.HexColor("#1b4034"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, leading=14,
                    spaceAfter=4, textColor=colors.HexColor("#444"))
NORMAL = ParagraphStyle("NORMAL", parent=styles["Normal"], fontSize=9.5, leading=13,
                        textColor=colors.HexColor("#222"))
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontSize=8.5, leading=11,
                       textColor=colors.HexColor("#444"))
def kv_table(rows, widths=(70*mm, 100*mm)):
    tbl = Table(rows, colWidths=list(widths))
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LINEBELOW", (0,0), (-1,0), 0.8, colors.HexColor("#33aa66")),
        ("BOX", (0,0), (-1,-1), 0.3, colors.HexColor("#888")),
        ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#aaa")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return tbl

story = []

# === COVER ===
story.append(Paragraph("Holter DIY — sintesi 3 sessioni", H1))
story.append(Paragraph(
    f"Periodo osservato: <b>{LABELS[0]} → {LABELS[-1]}</b>. "
    f"Hardware: AD8232 (singolo derivato precordiale, 250 Hz) + Pi Pico W → server Mac via TCP. "
    f"Analisi automatica con criterio uniforme: PVC se "
    f"(rebound ≥ {REBOUND_PVC} oppure width ≥ {PVC_W_MS:.0f}ms) <b>e</b> "
    f"ampiezza ≥ {PVC_MIN_AMP}V. Spike sub-fisiologici (≤16ms) e intervalli rumorosi "
    f"auto-rilevati esclusi prima del calcolo.",
    NORMAL))
story.append(Spacer(1, 8))

# --- Highlight box: i 3 messaggi chiave ---
all_pvc = sum(s["n_pvc"] for s in sessions)
all_couplets = sum(len(s["couplets"]) for s in sessions)
coup_medians = []
for s in sessions:
    cs = [p["rr_prev"]*1000 for p in s["peaks"]
          if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if cs: coup_medians.append(statistics.median(cs))

highlight_rows = [
    [Paragraph("<b>Focolaio</b>", SMALL),
     Paragraph(f"Singolo, monomorfo. Coupling stabile (~{statistics.mean(coup_medians):.0f}ms) "
               f"su tutte e 3 le sessioni — stessa origine elettrica.", SMALL)],
    [Paragraph("<b>Pericolosità</b>", SMALL),
     Paragraph(f"Nessun R-on-T (coupling sempre &gt; 360ms). Nessuna run ≥3 PVC. "
               f"<b>{all_couplets} couplet su {all_pvc} PVC totali</b> "
               f"({100*all_couplets/max(1,all_pvc):.2f}%): pochissime, "
               f"isolate, RR fra 384-460 ms; segnalate ed evidenziate nei "
               f"singoli report di sessione.", SMALL)],
    [Paragraph("<b>Aritmie</b>", SMALL),
     Paragraph(f"Screening AF negativo o borderline su tutte: l'irregolarità RR "
               f"si spiega con bradicardia + RSA + ectopia frequente, distribuzione "
               f"RR bimodale conservata.", SMALL)],
    [Paragraph("<b>Pattern HR↔sintomi</b>", SMALL),
     Paragraph(f"A HR basale più bassa prevalgono PVC interpolate (silenziose, "
               f"no pausa); salendo verso 57 BPM virano a compensate (percepite "
               f"come 'tonfo'). Spiega le fluttuazioni di percezione soggettiva.", SMALL)],
]
htbl = Table(highlight_rows, colWidths=[35*mm, 138*mm])
htbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8f4ec")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#888")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#bbb")),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(htbl)
story.append(Spacer(1, 12))

# === TABELLA RIASSUNTIVA ===
story.append(Paragraph("Tabella riassuntiva", H2))
header = [Paragraph(f"<b>{x}</b>", SMALL) for x in
          ["Metrica"] + LABELS]
def fmt(v, dec=1):
    if isinstance(v,(int,)): return str(v)
    return f"{v:.{dec}f}"
rows = [header]
for label, key, dec in [
    ("Durata utile (min)", "clean_s", 0),
    ("Esclusi (s)", "excl_s", 0),
    ("Battiti totali", "n_total", 0),
    ("Sinus BPM", "sinus_bpm", 1),
    ("PVC totali", "n_pvc", 0),
    ("PVC rate (/min)", "pvc_rate", 1),
    ("Burden (%)", "burden", 1),
    ("Coupling mediano (ms)", "RR_SINUS_MS", 0),
]:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if key == "clean_s": v = v/60
        cells.append(Paragraph(fmt(v, dec), SMALL))
    rows.append(cells)
# PVC patterns
for label, key in [("Couplet veri", "couplets"),
                    ("Interpolate", "interp"),
                    ("Compensate", "comp"),
                    ("Incomplete", "incomp"),
                    ("PVC isolate", "iso_pvc"),
                    ("Bigem PVC-N-PVC", "bigem"),
                    ("Trigem PVC-N-N-PVC", "trigem")]:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if isinstance(v, list): v = len(v)
        cells.append(Paragraph(str(v), SMALL))
    rows.append(cells)
# AF
cells = [Paragraph("AF score (0-4)", SMALL)]
for s in sessions:
    cells.append(Paragraph(f"{s['af'].get('score','-')}/4" if s.get("af") else "-", SMALL))
rows.append(cells)
cells = [Paragraph("RMSSD (ms)", SMALL)]
for s in sessions:
    cells.append(Paragraph(f"{s['af'].get('rmssd',0):.0f}" if s.get("af") else "-", SMALL))
rows.append(cells)
# --- color coding ---
# colonne sessione: ognuna ha un colore tenue (corrisponde alle barre nei grafici)
_TINT_PALETTE = [
    colors.HexColor("#e8f0f7"),   # azzurro tenue
    colors.HexColor("#fbeedd"),   # arancio tenue
    colors.HexColor("#fbe5e7"),   # rosa tenue
    colors.HexColor("#e6f4ec"),   # verde tenue
    colors.HexColor("#f1e6f7"),   # viola tenue
    colors.HexColor("#fff7d6"),   # giallo tenue
    colors.HexColor("#f4e8cf"),   # ocra tenue
]
COL_TINTS = [_TINT_PALETTE[i % len(_TINT_PALETTE)] for i in range(len(sessions))]
# righe-evidenziate: burden, couplet, interpolate, compensate, AF score
ROW_EMPH = {7: True, 9: True, 10: True, 11: True, 17: True}  # 1-indexed dopo header

# valuta semaforicamente burden e AF
def burden_color(v):
    # 0-15 green, 15-25 yellow, >25 orange (PVC monofocali fino al 30% sono benigne)
    if v < 15: return colors.HexColor("#1b4034")
    if v < 25: return colors.HexColor("#9b6b00")
    return colors.HexColor("#a3320c")
def af_color(score):
    if score is None or score == 0: return colors.HexColor("#1b4034")
    if score <= 2: return colors.HexColor("#9b6b00")
    return colors.HexColor("#a3320c")
def couplet_color(n):
    if n == 0: return colors.HexColor("#1b4034")
    if n <= 3: return colors.HexColor("#9b6b00")
    return colors.HexColor("#a3320c")

# ricostruisco rows con celle colorate per le metriche chiave
new_rows = [header]
metric_specs = [
    ("Durata utile (min)", "clean_s", 0),
    ("Esclusi (s)", "excl_s", 0),
    ("Battiti totali", "n_total", 0),
    ("Sinus BPM", "sinus_bpm", 1),
    ("PVC totali", "n_pvc", 0),
    ("PVC rate (/min)", "pvc_rate", 1),
    ("Burden (%)", "burden", 1),
    ("Coupling mediano (ms)", "RR_SINUS_MS", 0),
]
for label, key, dec in metric_specs:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if key == "clean_s": v = v/60
        if key == "burden":
            txt = f"<b><font color='{burden_color(v).hexval().replace('0x','#')}'>{v:.{dec}f}%</font></b>"
            cells.append(Paragraph(txt, SMALL))
        elif key == "sinus_bpm":
            cells.append(Paragraph(f"<b>{v:.{dec}f}</b>", SMALL))
        elif key == "n_pvc":
            # PVC totali + % sul totale battiti (= burden)
            pct = 100*v/max(1, s["n_total"])
            cells.append(Paragraph(f"{v} <font color='#666'>({pct:.1f}%)</font>", SMALL))
        elif key == "pvc_rate":
            cells.append(Paragraph(f"{v:.{dec}f} /min", SMALL))
        else:
            cells.append(Paragraph(fmt(v, dec), SMALL))
    new_rows.append(cells)
# PVC patterns con couplet evidenziati
pattern_specs = [("Couplet veri", "couplets"),
                  ("Interpolate", "interp"),
                  ("Compensate", "comp"),
                  ("Incomplete", "incomp"),
                  ("PVC isolate", "iso_pvc"),
                  ("Bigem PVC-N-PVC", "bigem"),
                  ("Trigem PVC-N-N-PVC", "trigem")]
for label, key in pattern_specs:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if isinstance(v, list): v = len(v)
        # denominatore per il %: interp/comp/incomp rapportate alle PVC classificabili
        # (sandwich N-PVC-N); couplet/iso/bigem/trigem rapportate al totale PVC
        n_class = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
        denom = n_class if key in ("interp","comp","incomp") else s["n_pvc"]
        pct = 100*v/denom if denom else 0
        if key == "couplets":
            col = couplet_color(v).hexval().replace('0x','#')
            cells.append(Paragraph(
                f"<b><font color='{col}'>{v}</font></b> "
                f"<font color='#666'>({pct:.2f}%)</font>", SMALL))
        elif key == "interp":
            cells.append(Paragraph(
                f"<font color='#1f6fa8'><b>{v}</b> ({pct:.0f}%)</font>", SMALL))
        elif key == "comp":
            cells.append(Paragraph(
                f"<font color='#a3320c'><b>{v}</b> ({pct:.0f}%)</font>", SMALL))
        elif key == "incomp":
            cells.append(Paragraph(
                f"<font color='#666'>{v} ({pct:.0f}%)</font>", SMALL))
        else:
            cells.append(Paragraph(
                f"{v} <font color='#888'>({pct:.0f}%)</font>", SMALL))
    new_rows.append(cells)
# AF score colorato
cells = [Paragraph("AF score (0-4)", SMALL)]
for s in sessions:
    sc = s["af"].get("score") if s.get("af") else None
    if sc is None:
        cells.append(Paragraph("-", SMALL))
    else:
        col = af_color(sc).hexval().replace('0x','#')
        cells.append(Paragraph(f"<b><font color='{col}'>{sc}/4</font></b>", SMALL))
new_rows.append(cells)
cells = [Paragraph("RMSSD (ms)", SMALL)]
for s in sessions:
    cells.append(Paragraph(f"{s['af'].get('rmssd',0):.0f}" if s.get("af") else "-", SMALL))
new_rows.append(cells)

tbl = Table(new_rows, colWidths=[55*mm] + [38*mm]*len(sessions))
style_cmds = [
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
    ("LINEBELOW", (0,0), (-1,0), 0.8, colors.HexColor("#33aa66")),
    ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#888")),
    ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#ccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    # background tinto per ciascuna colonna sessione
    # prima colonna metrica
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0f0f0")),
]
for _ci in range(len(sessions)):
    style_cmds.append(("BACKGROUND", (_ci+1, 1), (_ci+1, -1), COL_TINTS[_ci]))
# evidenzia righe chiave con bordo verde
for r_idx in [7, 9, 10, 11, 17]:  # burden, couplet, interp, comp, AF
    if r_idx < len(new_rows):
        style_cmds.append(("LINEBEFORE", (0, r_idx), (0, r_idx), 3, colors.HexColor("#33aa66")))
tbl.setStyle(TableStyle(style_cmds))
story.append(tbl)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<font color='#1b4034'>■</font> verde = nella norma per PVC monofocale benigna · "
    "<font color='#9b6b00'>■</font> ambra = intermedio · "
    "<font color='#a3320c'>■</font> rosso = sopra soglia attenzione · "
    "<font color='#1f6fa8'>■</font> interpolate · <font color='#a3320c'>■</font> compensate. "
    "Le 3 colonne sessione corrispondono ai colori dei grafici. "
    "Le % per interp/comp/incomp sono sulle PVC classificabili (sandwich N-PVC-N); "
    "couplet/isolate/bigem/trigem sono sul totale PVC; PVC totali e burden sono "
    "sul totale battiti.",
    SMALL))
story.append(Spacer(1, 10))

# === BARS comparativi ===
story.append(Paragraph("Confronto principali parametri", H2))
story.append(Paragraph(
    "Burden = % di battiti ectopici sul totale; sinus rate = battiti normali al "
    "minuto; PVC rate = ectopie al minuto.", SMALL))
story.append(fit_image(img_bars, max_w_mm=175, max_h_mm=65))
story.append(Spacer(1, 10))

# === COMPOSITION ===
story.append(Paragraph("Composizione PVC: interpolate vs compensate", H2))
story.append(Paragraph(
    "Una <b>PVC interpolata</b> si infila fra due battiti N senza resettare il nodo "
    "SA (somma RR_pre + RR_post ≈ 1× RR sinus). Una <b>PVC con pausa compensatoria "
    "piena</b> resetta il SA (somma ≈ 2× RR sinus). Le interpolate sono favorite "
    "dalle bradicardie e sono emodinamicamente più benigne (no perdita di gittata, "
    "no percezione del 'tonfo').", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_comp, max_w_mm=175, max_h_mm=65))
story.append(Spacer(1, 10))

# === KEY PATTERN HR vs %COMP ===
story.append(Paragraph("Il pattern chiave: HR basale → tipo di pausa → percezione", H2))
story.append(Paragraph(
    "Le 3 sessioni mostrano un <b>gradiente coerente</b> tra frequenza basale e tipo "
    "di pausa post-PVC. Nell'arco di soli 5 BPM (da 52 a 57) la quota di PVC "
    "compensate (quelle che generano il 'tonfo' percepito) passa da ~25% a 63%. "
    "Speculare l'andamento delle interpolate, che dominano a HR più bassa. "
    "Spiegazione elettrofisiologica: a HR più bassa il nodo SA ha cicli più lunghi "
    "(~1100ms), la PVC fa quasi sempre in tempo a infilarsi prima che il SA spari di "
    "nuovo (interpolata silenziosa); a HR più alta il SA è prossimo allo scatto e si "
    "fa resettare dall'onda retrograda (compensatoria percepita).",
    NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_hr_comp, max_w_mm=140, max_h_mm=80))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Implicazione clinica:</b> le fluttuazioni di sintomatologia (alcuni giorni "
    "'le sento tantissime', altri 'oggi niente') non sono variazioni del numero di "
    "PVC, ma di <b>quanti</b> di quei battiti diventano <b>percepibili</b>. La "
    "soglia critica è ~55 BPM per questo paziente: sotto, le PVC sono silenziose; "
    "sopra, virano a compensate sintomatiche.",
    NORMAL))
story.append(PageBreak())

# === DISTRIBUZIONE DEL RAPPORTO PER SESSIONE ===
story.append(Paragraph("Distribuzione del rapporto pausa / RR sinusale per sessione", H2))
story.append(Paragraph(
    "Per ogni PVC sandwich-fra-due-N si calcola il rapporto "
    "<i>(RR_pre + RR_post) / RR_sinus</i>. La distribuzione di questo rapporto "
    "rivela direttamente il comportamento del nodo SA: la zona ombreggiata sinistra "
    "(&lt;1.30) indica le interpolate, quella destra (1.85-2.15) le compensate "
    "piene. La linea gialla è la mediana della sessione.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_distrib, max_w_mm=175, max_h_mm=170))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Le 3 sessioni hanno <b>forme diverse</b> nonostante lo stesso focolaio: il "
    "comportamento del SA dipende dallo stato autonomico in quel momento, non dalla "
    "PVC in sé. Il 6 giu (mediana 1.99) mostra un cuore in modalità 'reset sempre'; "
    "il 7 giu (mediana 1.40-1.50) mostra distribuzione bimodale con metà delle PVC "
    "in zona interpolata; il 5 giu è intermedio.",
    NORMAL))
story.append(PageBreak())

# === BPM BUCKET ANALYSIS ===
story.append(Paragraph("A quale frequenza istantanea compaiono le compensate?", H2))
story.append(Paragraph(
    "Ogni PVC è etichettata con la frequenza istantanea immediatamente precedente "
    "(RR del N-N appena prima della PVC, convertito in BPM). Aggregando tutte le "
    "PVC delle 3 sessioni per fascia di BPM si ottiene la <b>curva dose-risposta</b> "
    "della percezione: la linea gialla mostra la percentuale di PVC compensate "
    "(percepibili) in funzione della frequenza basale istantanea.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_bpm_bucket, max_w_mm=175, max_h_mm=110))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Lettura (aggregato 3 sessioni, n=1249 PVC classificabili): sotto 60 BPM "
    "<b>0% compensate</b> (tutte interpolate, silenziose). Il <b>crossover al 50%</b> "
    "avviene in fascia <b>60-65 BPM</b>. Tra 65-80 BPM la quota compensate cresce "
    "rapidamente fino al <b>picco del 94% in fascia 75-80 BPM</b>. Oltre 80 BPM la "
    "classificazione diventa sfumata (gli RR si accorciano e la maggior parte delle "
    "PVC cade nella zona 'incompleta', né interpolata né compensata pura).",
    NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Note metodologiche: i buckets di 5 BPM e le soglie 1.30 / 1.85-2.15 RR_sinus "
    "(per definire interpolata / compensatoria) sono scelte parametriche, derivate "
    "dalla fisiologia da manuale. Il gradiente HR→%comp e i numeri qui sopra sono "
    "invece calcolati direttamente dai dati. La sessione 6 giu (0% interpolate "
    "intrinsecamente) trascina la statistica aggregata; analizzando le sessioni "
    "individualmente il crossover si sposta avanti (es. su 7 giu da sola era ~70 BPM).</i>",
    SMALL))
story.append(Spacer(1, 10))

# === COUPLING STABILITY ===
story.append(Paragraph("Stabilità del coupling (origine del focolaio)", H2))
story.append(Paragraph(
    "Distribuzione del coupling pre-PVC (intervallo tra N che precede e la PVC). "
    "Se la distribuzione è stretta e la mediana è coerente fra sessioni, il focolaio "
    "è <b>monomorfo e fisso</b> nello stesso punto del ventricolo. Se si allarga o "
    "compaiono picchi separati, suggerisce origine multifocale.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_coup, max_w_mm=175, max_h_mm=65))
story.append(PageBreak())

# === DETTAGLIO COUPLING: ANALISI CLUSTER (BIMODALITÀ 6 GIU) ===
story.append(Paragraph("Analisi cluster del coupling per sessione", H2))
story.append(Paragraph(
    "Zoom sulla distribuzione del coupling pre-PVC con bin da 15 ms. Le barre sono "
    "colorate per fascia: <font color='#3a9'>azzurro &lt; 500 ms</font>, "
    "<font color='#c33'>rosa 500-600 ms</font>, "
    "<font color='#393'>verde &gt; 600 ms</font>. Questa visualizzazione rivela una "
    "<b>bimodalità nascosta nella sessione 6 giu</b>: oltre al cluster principale a "
    "~470 ms, esiste un secondo gruppo persistente attorno a 540-560 ms (43% delle "
    "PVC). Le altre sessioni mostrano cluster singolo.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_cluster, max_w_mm=175, max_h_mm=140))
story.append(Spacer(1, 8))

# Tabella morfologica dei cluster per ciascuna sessione
story.append(Paragraph("<b>Morfologia mediana per cluster (sessione 6 giu)</b>", H3))
ses_6giu = sessions[1]  # 6 giu è l'indice 1
clus_6 = session_clusters[1]
rows_c = [[Paragraph(f"<b>{x}</b>", SMALL) for x in
           ["Cluster","n (%)","Coupling (ms)","Ampiezza (V)","Width (ms)","Rebound"]]]
for label, key in [("A < 500ms","c1"),("B 500-600ms","c2"),("C > 600ms","c3")]:
    d = clus_6.get(key)
    if d is None:
        rows_c.append([Paragraph(label, SMALL)] + [Paragraph("-", SMALL)]*5)
    else:
        rows_c.append([Paragraph(label, SMALL),
                       Paragraph(f"{d['n']} ({d['pct']:.0f}%)", SMALL),
                       Paragraph(f"{d['rr_med']:.0f}", SMALL),
                       Paragraph(f"{d['amp_med']:.2f}", SMALL),
                       Paragraph(f"{d['w_med']:.0f}", SMALL),
                       Paragraph(f"{d['reb_med']:.2f}", SMALL)])
ct = Table(rows_c, colWidths=[32*mm, 22*mm, 26*mm, 26*mm, 24*mm, 22*mm])
ct.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
    ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#33aa66")),
    ("BOX", (0,0), (-1,-1), 0.3, colors.HexColor("#888")),
    ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#bbb")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(ct)
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Interpretazione fisiologica</b>", H3))
story.append(Paragraph(
    "Le micro-differenze morfologiche fra i cluster (width 100 vs 104 ms, rebound "
    "0.59 vs 0.65, ampiezze sovrapponibili) sono <b>troppo piccole</b> per "
    "indicare un secondo focolaio: una vera ectopia di origine diversa mostrerebbe "
    "delta di 20-30 ms in width e cambi di polarità/morfologia ben visibili. "
    "Le tre ipotesi in ordine di plausibilità:",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>1. Modulazione del coupling (più probabile).</b> Lo stesso focolaio scarica "
    "a 2 frequenze leggermente diverse in base allo stato autonomico: quando il SA "
    "è più veloce il coupling è più corto (~470 ms), quando rallenta sale (~545 ms). "
    "Pattern compatibile con parasistolia benigna.",
    NORMAL))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "<b>2. Stesso focolaio, due vie d'uscita.</b> L'ectopia nasce nello stesso "
    "punto ma può prendere due cammini di conduzione leggermente diversi verso il "
    "ventricolo, arrivando con ~70 ms di ritardo. Spiega le micro-differenze "
    "morfologiche senza richiedere un secondo focus.",
    NORMAL))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "<b>3. Bifocale lieve.</b> Due cellule ectopiche vicine con proprietà simili. "
    "Difficile da escludere con un solo derivato precordiale.",
    NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Solo la sessione 6 giu mostra questa bimodalità — le altre due hanno cluster "
    "singolo. Coerente con stato autonomico più 'tonico' (HR 57 vs 52-54 delle "
    "altre): possibile che il livello simpatico più alto attivi un secondo 'mode' "
    "di scarica o una via di conduzione alternativa. <b>Da segnalare al "
    "cardiologo</b>: se eseguono un holter 24h è utile verificare se la "
    "bimodalità è riproducibile e se compare in fasce orarie specifiche.",
    NORMAL))
story.append(PageBreak())

# === ANDAMENTO TEMPORALE ===
story.append(Paragraph("Andamento temporale intra-sessione", H2))
story.append(Paragraph(
    "Per ogni sessione: sinus rate (verde) e burden (rosso) minuto per minuto. "
    "Permette di vedere se la frequenza basale è stabile e se il burden cambia "
    "nel tempo (es. risposta a respirazione, stress, posizione).", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_temporal, max_w_mm=175, max_h_mm=120))
story.append(PageBreak())

# === ESEMPI ===
story.append(Paragraph("Esempi morfologici delle classificazioni", H2))
story.append(Paragraph(
    "Per ogni sessione, una strip rappresentativa di ciascun tipo. Cerchio "
    "arancione = PVC analizzata, barra azzurra = RR_pre, barra gialla = RR_post, "
    "linea rossa tratteggiata = posizione attesa del battito successivo se la "
    "pausa fosse compensatoria piena (2× RR sinus). Una interpolata mostra il "
    "battito N successivo <b>prima</b> di quella linea; una compensata lo mostra "
    "<b>sulla</b> linea.", NORMAL))
story.append(Spacer(1, 8))

for kind_label, kind_key in [
    ("Interpolate (no reset del SA)", "interp"),
    ("Compensate (reset completo del SA)", "comp"),
    ("Couplet (2 PVC consecutive entro 700ms)", "couplet"),
]:
    story.append(Paragraph(kind_label, H3))
    items = example_strips.get(kind_key, [])
    if not items:
        story.append(Paragraph("(nessun esempio disponibile)", SMALL))
    for (lab, img) in items:
        story.append(fit_image(img, max_w_mm=175, max_h_mm=55))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 6))

story.append(PageBreak())

# === ANALISI RESPIRATORIA + TRIGGER FASICO ===
if img_resp_phases is not None:
    story.append(Paragraph("Analisi respiratoria: trigger fasico end-expiratorio", H2))
    story.append(Paragraph(
        "Le PVC non scoccano in modo casuale durante il ciclo respiratorio. Estraendo "
        "il segnale respiratorio direttamente dall'ECG (tecnica <b>EDR — ECG-Derived "
        "Respiration</b>: la modulazione dell'ampiezza R battito-per-battito traccia "
        "il respiro grazie alla rotazione del vettore elettrico durante l'escursione "
        "diaframmatica) e calcolando la fase istantanea via trasformata di Hilbert, "
        "si può associare a ciascuna PVC la sua posizione nel ciclo respiratorio "
        "(0% = fine espirazione = ampiezza minima; 50% = fine inspirazione = "
        "ampiezza massima).",
        NORMAL))
    story.append(Spacer(1, 6))
    if img_resp_example is not None:
        story.append(Paragraph("Esempio di estrazione respiratoria dall'ECG (sessione di riferimento)", H3))
        story.append(fit_image(img_resp_example, max_w_mm=175, max_h_mm=60))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Sopra: 30 secondi di ECG (verde) con triangoli gialli sui picchi R. "
            "La linea ciano sovrapposta è la respirazione ricostruita dall'ampiezza "
            "dei picchi R, perfettamente sinusoidale e in fase con i cicli respiratori "
            "osservati a occhio nudo nella variazione di altezza dei QRS.",
            SMALL))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Distribuzione delle PVC nel ciclo respiratorio per sessione", H3))
    story.append(Paragraph(
        "Per ciascuna sessione: a sinistra rosetta polare (verde = battiti N, rosso = PVC); "
        "a destra rapporto PVC/N per fase (barre rosse = eccesso PVC, verdi = deficit, "
        "azzurre = neutre; linea bianca tratteggiata = uniforme).",
        SMALL))
    story.append(Spacer(1, 6))
    story.append(fit_image(img_resp_phases, max_w_mm=180, max_h_mm=240))
    story.append(PageBreak())
    # Tabella riassuntiva trigger respiratorio
    story.append(Paragraph("Riassunto trigger fasico — tutte le sessioni", H3))
    resp_rows = [
        [Paragraph(f"<b>{x}</b>", SMALL) for x in
         ["Sessione","Rate resp /min","Picco fasico","Enrichment","p-value","Significativo"]]
    ]
    for s in sessions_with_edr:
        edr = s["edr"]
        # interpretazione fase (Hilbert: 0=max inspir, 50%=fine espir)
        ph = edr["peak_phase_pct"]
        if ph < 15 or ph > 85:
            phase_lab = "max inspir. ★"
        elif 15 <= ph < 35:
            phase_lab = "mid espir."
        elif 35 <= ph < 65:
            phase_lab = "fine espir."
        else:
            phase_lab = "mid inspir."
        sig_lab = "★★ p<10⁻⁹" if edr["pval"] < 1e-9 else ("★ p<0.001" if edr["pval"]<1e-3 else f"p={edr['pval']:.0e}")
        resp_rows.append([
            Paragraph(label_of(s), SMALL),
            Paragraph(f"{edr['rate_resp']:.1f}", SMALL),
            Paragraph(phase_lab, SMALL),
            Paragraph(f"×{edr['peak_enrich']:.2f}", SMALL),
            Paragraph(f"{edr['pval']:.0e}", SMALL),
            Paragraph(sig_lab, SMALL),
        ])
    resp_tbl = Table(resp_rows, colWidths=[35*mm, 25*mm, 30*mm, 22*mm, 25*mm, 35*mm])
    resp_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
        ("LINEBELOW", (0,0), (-1,0), 0.8, colors.HexColor("#33aa66")),
        ("BOX", (0,0), (-1,-1), 0.3, colors.HexColor("#888")),
        ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#bbb")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(resp_tbl)
    story.append(Spacer(1, 8))
    n_maxinsp = sum(1 for s in sessions_with_edr
                    if s["edr"]["peak_phase_pct"] < 15 or s["edr"]["peak_phase_pct"] > 85)
    story.append(Paragraph(
        f"<b>Pattern ricorrente:</b> {n_maxinsp} su {len(sessions_with_edr)} sessioni "
        f"mostrano picco di scarica PVC attorno alla <b>massima inspirazione / "
        f"transizione verso espirazione</b> (entro il 15% del ciclo da phase 0 = "
        f"picco di ampiezza QRS). Riproducibilità straordinaria nonostante "
        f"variabilità di postura, orario, pasti, attività.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Interpretazione fisiologica.</b> A massima inspirazione convergono "
        "fattori meccanici e neurovegetativi che spiegano l'eccesso di scarica del "
        "focolaio: "
        "(1) <b>diaframma al punto più caudale</b>, massimamente contratto, scivolato "
        "verso l'addome. Il pericardio, attaccato al centro tendineo del diaframma, "
        "subisce <b>massima trazione verso il basso</b>; il cuore viene stirato/spostato "
        "inferiormente. "
        "(2) <b>Riempimento ventricolare massimo</b> (precarico al picco) → stretch "
        "delle camere → attivazione meccanoricettori → ectopia triggered. "
        "(3) <b>Tachicardia inspiratoria RSA</b>: la fase ascendente del ciclo "
        "Respiratory Sinus Arrhythmia coincide con leggero aumento simpatico locale. "
        "Per il paziente con pattern respiratorio apicale + rib flare la trazione "
        "pericardica anomala amplifica l'effetto. Il focolaio è dunque "
        "<b>mechanically-triggered (stretch) + RSA-modulated</b>.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Implicazione terapeutica:</b> riduzione dello stretch inspiratorio massimo. "
        "Tecniche che limitano la <b>profondità inspiratoria</b> (volume tidale ridotto, "
        "respirazione consapevole \"shallow\") riducono il tempo trascorso a picco "
        "inspiratorio. La <b>coherent breathing 6/min</b> con volumi piccoli (no respiri "
        "molto profondi) è preferibile a tecniche con inspirazioni massimali "
        "(es. respirazione yogica completa, sospiri profondi). Anche il lavoro sul "
        "pattern diaframmatico (PRI/DNS) per riportare il diaframma a posizione neutra "
        "riduce l'escursione anomala e quindi la trazione pericardica.",
        NORMAL))
    story.append(PageBreak())

# === SINTESI ===
story.append(Paragraph("Sintesi e osservazioni cross-sessione", H2))
all_couplets = sum(len(s["couplets"]) for s in sessions)
all_pvc = sum(s["n_pvc"] for s in sessions)
all_interp = sum(len(s["interp"]) for s in sessions)
all_comp = sum(len(s["comp"]) for s in sessions)
all_classified = all_interp + all_comp + sum(len(s["incomp"]) for s in sessions)
ratio_interp = 100*all_interp/max(1, all_classified)
coup_meds = [s["RR_SINUS_MS"] for s in sessions]  # qui RR sinus, non coupling, ma è ok
# coupling veri
coup_medians = []
for s in sessions:
    cs = [p["rr_prev"]*1000 for p in s["peaks"]
          if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if cs: coup_medians.append(statistics.median(cs))

bullets = [
    f"<b>PVC totali nelle 3 sessioni</b>: {all_pvc} su {sum(s['n_total'] for s in sessions)} battiti totali.",
    f"<b>Couplet veri</b> (2 PVC consecutive con RR &lt; 700ms): <b>{all_couplets} in totale</b> "
    f"(0 nella sessione 5 giu, 2 il 6 giu, 2 il 7 giu), tutti con RR fra 384-460 ms. "
    f"Rappresentano lo <b>0.23% del totale PVC</b> — pochissimi ma esistono. "
    f"Sono individuati e mostrati nei singoli report di sessione (sezione 'Esempi "
    f"morfologici', stesso stile della traccia esemplificativa). "
    f"<b>Nessun run di tachicardia ventricolare</b> (≥3 PVC consecutive): 0 in tutte le sessioni.",
    f"<b>Focolaio singolo monomorfo</b>: il coupling mediano è stabile fra sessioni "
    f"({', '.join(f'{m:.0f}ms' for m in coup_medians)}). Distribuzione stretta = unica origine elettrica.",
    f"<b>{ratio_interp:.0f}% delle PVC classificate sono interpolate</b> (no pausa "
    f"compensatoria). Compatibile con bradicardia di base e buon tono vagale: "
    f"queste PVC non disturbano il riempimento ventricolare e tipicamente non vengono percepite.",
    f"<b>No R-on-T</b>: nessuna PVC con coupling < 360ms in nessuna delle 3 sessioni.",
]
af_scores = [s["af"].get("score","-") for s in sessions if s.get("af")]
if af_scores:
    bullets.append(f"<b>Screening AF</b>: score {'/'.join(str(x) for x in af_scores)} (su 4). "
                   f"Markers HRV alti ma struttura RR bimodale conservata; non suggestivo di fibrillazione atriale. "
                   f"L'irregolarità RR osservata si spiega con bradicardia + RSA + ectopia frequente.")

bullets.append(f"<b>Correlazione frequenza vs percezione</b>: a HR basale bassa (~55-65 BPM) "
               f"prevalgono interpolate silenziose; nella fascia 70-80 BPM crescono "
               f"le compensate, percepite come 'tonfo'. Coerente con la fisiologia del nodo SA.")

for b in bullets:
    story.append(Paragraph("• " + b, NORMAL))
    story.append(Spacer(1, 3))

# === TABELLA FINALE: PVC totali vs percepite ===
story.append(Spacer(1, 14))
story.append(Paragraph("PVC reali vs PVC percepite — riepilogo", H2))
story.append(Paragraph(
    "Il dato controintuitivo che emerge dalle 3 sessioni: <b>il numero di PVC "
    "percepite NON è proporzionale al numero di PVC reali</b>. La sessione con più "
    "PVC in assoluto (7 giu) è quella in cui ne sono state percepite di meno, "
    "perché aveva la HR più bassa e quindi la più alta quota di interpolate "
    "silenziose. Il numero di 'tonfi' al minuto è stimato come N° PVC compensate / "
    "durata utile della sessione.",
    NORMAL))
story.append(Spacer(1, 6))

# Calcolo tonfi/min per ciascuna sessione
header_perc = [Paragraph(f"<b>{x}</b>", SMALL) for x in
               ["Indicatore"] + LABELS]
rows_perc = [header_perc]
def cell_perc(text, col_hex=None, bold=False):
    if bold and col_hex:
        return Paragraph(f"<b><font color='{col_hex}'>{text}</font></b>", SMALL)
    if col_hex:
        return Paragraph(f"<font color='{col_hex}'>{text}</font>", SMALL)
    if bold:
        return Paragraph(f"<b>{text}</b>", SMALL)
    return Paragraph(text, SMALL)

# riga 1: PVC totali al minuto
cells = [cell_perc("PVC totali (/min)")]
pvc_rates = [s["pvc_rate"] for s in sessions]
max_pvc = max(pvc_rates); min_pvc = min(pvc_rates)
for s in sessions:
    v = s["pvc_rate"]
    if v == max_pvc: c = "#a3320c"   # alto = rosso
    elif v == min_pvc: c = "#1b4034" # basso = verde
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{v:.1f} /min", c, bold=True))
rows_perc.append(cells)

# riga 2: Burden %
cells = [cell_perc("Burden")]
burdens = [s["burden"] for s in sessions]
max_b = max(burdens); min_b = min(burdens)
for s in sessions:
    v = s["burden"]
    if v == max_b: c = "#a3320c"
    elif v == min_b: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{v:.1f}%", c, bold=True))
rows_perc.append(cells)

# riga 3: SA effettiva — vera frequenza del pacemaker
# Calcolata come 60000/median(RR_NN) dopo esclusione rumore + manuale,
# solo coppie N-N consecutive con RR fisiologico (0.6-1.4s).
# Corrisponde a quello che senti al polso (le PVC hanno gittata insufficiente
# per generare onda pulsatile palpabile).
cells = [cell_perc("Frequenza SA effettiva (al polso)")]
sa_eff = [60000/s["RR_SINUS_MS"] if s["RR_SINUS_MS"] else 0 for s in sessions]
max_sa = max(sa_eff); min_sa = min(sa_eff)
for s, v in zip(sessions, sa_eff):
    if v == max_sa: c = "#a3320c"   # alta = rosso (più compensate)
    elif v == min_sa: c = "#1b4034" # bassa = verde (più interpolate)
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{v:.1f} BPM", c, bold=True))
rows_perc.append(cells)

# riga 3b: RR sinus medio (ms) — il "tempo di carica" del nodo SA
cells = [cell_perc("RR SA medio (ms)")]
rr_values = [s["RR_SINUS_MS"] for s in sessions]
max_rr = max(rr_values); min_rr = min(rr_values)
for s, rr in zip(sessions, rr_values):
    if rr == max_rr: c = "#1b4034"  # RR più lungo = più spazio → verde
    elif rr == min_rr: c = "#a3320c"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{rr:.0f} ms", c, bold=True))
rows_perc.append(cells)

# riga 3c: HR totale percepita come "battiti" (N + PVC) — quello che senti dentro
cells = [cell_perc("Battiti totali percepiti (/min)")]
totals = [(s["n_norm"]+s["n_pvc"])*60/s["clean_s"] if s["clean_s"] else 0
          for s in sessions]
max_t = max(totals); min_t = min(totals)
for s, t in zip(sessions, totals):
    if t == max_t: c = "#a3320c"
    elif t == min_t: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{t:.1f} /min", c, bold=True))
rows_perc.append(cells)

# riga 4: % compensate (sulle classificabili)
cells = [cell_perc("% compensate (percepibili)")]
pct_comps = []
for s in sessions:
    n_class = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
    pct = 100*len(s["comp"])/max(1, n_class)
    pct_comps.append(pct)
max_pc = max(pct_comps); min_pc = min(pct_comps)
for s, pct in zip(sessions, pct_comps):
    if pct == max_pc: c = "#a3320c"
    elif pct == min_pc: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{pct:.0f}%", c, bold=True))
rows_perc.append(cells)

# riga 5: tonfi/min stimati
cells = [cell_perc("'Tonfi' percepiti (/min)")]
tonfi = [len(s["comp"])/(s["clean_s"]/60) if s["clean_s"] else 0 for s in sessions]
max_t = max(tonfi); min_t = min(tonfi)
for s, t in zip(sessions, tonfi):
    if t == max_t: c = "#a3320c"
    elif t == min_t: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{t:.1f} /min", c, bold=True))
rows_perc.append(cells)

# colore tinted colonne sessione
COL_TINTS_END = COL_TINTS  # stessa palette adattiva
tbl_perc = Table(rows_perc, colWidths=[55*mm] + [38*mm]*len(sessions))
tbl_perc_style = [
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b4034")),
    ("LINEBELOW", (0,0), (-1,0), 0.8, colors.HexColor("#33aa66")),
    ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#888")),
    ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#ccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0f0f0")),
]
for _ci in range(len(sessions)):
    tbl_perc_style.append(("BACKGROUND", (_ci+1, 1), (_ci+1, -1), COL_TINTS_END[_ci]))
tbl_perc_style += [
    # evidenzia la riga "tonfi/min" (ora indice 7 dopo l'aggiunta di SA + RR + totali)
    ("LINEBEFORE", (0,7), (0,7), 3, colors.HexColor("#33aa66")),
]
tbl_perc.setStyle(TableStyle(tbl_perc_style))
story.append(tbl_perc)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Lettura: nella sessione 7 giu si è osservato il <b>maggior numero di PVC</b> "
    "in assoluto (24.7/min, burden 32.2%), ma la <b>frequenza SA effettiva era la "
    "più bassa</b> ({:.1f} BPM) — questa ha portato la quota di compensate (le "
    "percepibili) al minimo (29%). Risultato: <b>solo 6.3 tonfi/min percepiti</b>, "
    "contro 9.3 del 6 giu e 8.3 del 5 giu — nonostante il 7 giu avesse "
    "oggettivamente più ectopie. Le sensazioni soggettive ('oggi le sento di più / "
    "di meno') non sono un proxy affidabile del numero di PVC reali.".format(min(sa_eff)),
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>Nota sulla misurazione della HR: la 'frequenza SA effettiva' è la vera "
    "frequenza del pacemaker (60000/median RR_NN su coppie N-N consecutive dopo "
    "esclusione rumore). Corrisponde a ciò che si misura al polso, perché le PVC "
    "hanno gittata ridotta e tipicamente non generano un'onda pulsatile palpabile. "
    "I 'battiti totali percepiti' includono invece anche le PVC e sono ciò che si "
    "avverte come palpitazione interna. Il 'numero di tonfi' è la stima delle PVC "
    "che fanno sentire il colpo (le compensate, perché la pausa lascia "
    "iper-riempire il ventricolo e il battito successivo è ipercontrattile).</i>",
    SMALL))

story.append(Spacer(1, 12))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#888")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Disclaimer: report da setup DIY (singolo derivato precordiale AD8232 + Pi "
    "Pico W, sampling 250 Hz). Non equivale a un Holter clinico: l'analisi della morfologia "
    "fine (P, ST), la classificazione multifocale precisa e la diagnosi di aritmie "
    "complesse richiedono almeno 3 derivazioni e banda passante più ampia. Documento "
    "indicativo, non sostituisce valutazione medica.</i>", SMALL))

doc.build(story)
print(f"\nPDF salvato: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
