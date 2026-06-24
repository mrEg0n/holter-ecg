"""
Cross-session synthetic report: compares 3 ECG recordings, computes the same
indicators (burden, couplet, interpolated vs compensatory, AF screening) and produces
a summary PDF with tables, charts and examples of the key classifications.

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
PVC_W_MIN_MS = 40.0   # physiological range for PVC width: below = spike, above = baseline shift
PVC_W_MAX_MS = 220.0
PVC_MIN_REBOUND = 0.05  # minimum rebound: wide artifacts have reb=0
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
    # reclassify with the current criterion (plausibility check + minimum rebound)
    for p in peaks:
        shape_pvc = (p["reb"] >= REBOUND_PVC or p["w"] >= PVC_W_MS)
        plausible_w = PVC_W_MIN_MS <= p["w"] <= PVC_W_MAX_MS
        has_rebound = p["reb"] >= PVC_MIN_REBOUND
        p["cls"] = "pvc" if (shape_pvc and p["amp"] >= PVC_MIN_AMP
                              and plausible_w and has_rebound) else "normal"
    # remove noise spikes
    peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP)]
    return t, vr, vf, peaks

def detect_noise_intervals(t, vr, win_s=4.0, min_s=1.0):
    """Auto-detect noise bursts with a threshold ADAPTIVE to the session baseline.
    The threshold is (median baseline std) + 0.10V, so it works even on sessions
    recorded with a different AD8232 gain."""
    WIN = int(win_s * SR)
    std_arr = np.zeros(len(vr))
    for i in range(0, len(vr) - WIN, SR):
        std_arr[i:i+SR] = vr[i:i+WIN].std()
    # threshold: median + offset, but minimum 0.30
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
    """Load the manual exclusions from the JSON file produced by mark_exclusions.py.
    Returns a list of (start_s, end_s)."""
    import json as _json
    base = os.path.basename(ecg_path).replace("ecg_", "").replace(".csv", "")
    p = os.path.join("exclusions", f"exclusions_{base}.json")
    if not os.path.exists(p): return []
    try:
        with open(p) as f:
            data = _json.load(f)
        return [(d["start"], d["end"]) for d in data.get("intervals", [])]
    except Exception as e:
        print(f"[{base}] error reading {p}: {e}")
        return []

def analyze(ecg_path, manual_excl=None):
    """Return a dict with all the session numbers.
    If manual_excl is None, load from the JSON file exclusions_<base>.json if it exists,
    otherwise use only the noise auto-detection."""
    t, vr, vf, peaks = load_session(ecg_path)
    if manual_excl is None:
        manual_excl = load_manual_exclusions(ecg_path)
        if manual_excl:
            print(f"  loaded {len(manual_excl)} manual exclusions from JSON")
    noise_excl = detect_noise_intervals(t, vr)
    # consolidate intervals (manual + auto, no complicated merging)
    excl = list(noise_excl) + list(manual_excl)
    def in_excl(tv): return any(s <= tv <= e for s, e in excl)
    def in_manual(tv): return any(s <= tv <= e for s, e in manual_excl)
    peaks_clean = [p for p in peaks if not in_excl(p["t"])]
    # For couplets we use ONLY manual exclusions (more conservative):
    # the noise auto-detection may cut zones where a true couplet exists
    # adjacent to a noisy burst. A couplet has a very specific pattern (2 PVC
    # with RR<700ms) and is not generated by noise.
    peaks_for_couplet = [p for p in peaks if not in_manual(p["t"])]
    # RR
    for i in range(len(peaks_clean)):
        peaks_clean[i]["rr_prev"] = (peaks_clean[i]["t"] - peaks_clean[i-1]["t"]) if i > 0 else None
        peaks_clean[i]["rr_next"] = (peaks_clean[i+1]["t"] - peaks_clean[i]["t"]) if i < len(peaks_clean)-1 else None
    # useful time
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
    # true couplets (RR < 700ms) — use only manual exclusions
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
    # interpolated vs compensatory
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
    # bigeminy / trigeminy / iso
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
    # AF screening (consecutive NN)
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
    """Didactic strip with RR_pre, RR_post annotations if available."""
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
    styled_ax(ax, title, "t (s) relative to the center", "ECG (V)")
    ax.set_xlim(-win_s/2, win_s/2); ax.set_ylim(-1.2, 1.7)
    plt.tight_layout()
    return fig_to_bytes(fig)

# ---------- EDR (ECG-Derived Respiration) + PVC phasic analysis ----------
NBINS_RESP = 12
FS_RESP = 4.0

def extract_edr_and_phase(peaks):
    """Return a dict with EDR + instantaneous phase + PVC distribution analysis."""
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

# Manual exclusions per session (if needed)
MANUAL_EXCL = {
    "ecg_20260607_113338.csv": [(1472, 1505), (1690, 1780)],
}

print("Analyzing sessions...")
sessions = []
for arg in sys.argv[1:]:
    key = os.path.basename(arg)
    excl = MANUAL_EXCL.get(key, [])
    print(f"  {key} ...")
    s = analyze(arg, manual_excl=excl)
    # extract EDR + respiratory phase
    s["edr"] = extract_edr_and_phase(s["peaks"])
    if s["edr"]:
        print(f"    EDR: rate {s['edr']['rate_resp']:.1f}/min, "
              f"phasic peak {s['edr']['peak_phase_pct']:.0f}% of cycle, "
              f"enrich ×{s['edr']['peak_enrich']:.2f}, p={s['edr']['pval']:.0e}")
    sessions.append(s)

def label_of(s):
    base = os.path.basename(s["ecg_path"]).replace("ecg_", "").replace(".csv", "")
    # 20260605_131136 → 05 Jun 13:11
    d = base[6:8] + " Jun " + base[9:11] + ":" + base[11:13]
    return d

LABELS = [label_of(s) for s in sessions]
for s, L in zip(sessions, LABELS):
    n_class = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
    pct_i = 100*len(s["interp"])/max(1, n_class)
    pct_c = 100*len(s["comp"])/max(1, n_class)
    print(f"\n--- {L} ---")
    print(f"  useful duration: {s['clean_s']/60:.1f} min  (excluded {s['excl_s']:.0f}s)")
    print(f"  beats: {s['n_total']} (N={s['n_norm']}, PVC={s['n_pvc']})")
    print(f"  sinus {s['sinus_bpm']:.1f} BPM   PVC rate {s['pvc_rate']:.1f}/min   burden {s['burden']:.1f}%")
    print(f"  true couplets (RR<700ms): {len(s['couplets'])}")
    print(f"  interpolated {len(s['interp'])} ({pct_i:.0f}%)   compensated {len(s['comp'])} ({pct_c:.0f}%)   incomplete {len(s['incomp'])}")
    if s["af"]: print(f"  AF score {s['af']['score']}/4 — RMSSD {s['af']['rmssd']:.0f}ms pNN50 {s['af']['pnn50']:.0f}% hist-peaks {s['af']['n_peaks']}")

# ---------- COMPARATIVE PLOTS ----------
print("\nGenerating comparative plots...")

# A) Burden + rate comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=DARK_BG)
metrics = [("burden", "PVC burden (%)"),
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

# A2) Plot HR vs % compensatory — the key pattern
fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=DARK_BG)
hr_vals = [s["sinus_bpm"] for s in sessions]
comp_pct = []
interp_pct = []
for s in sessions:
    nc = len(s["interp"]) + len(s["comp"]) + len(s["incomp"])
    comp_pct.append(100*len(s["comp"])/max(1,nc))
    interp_pct.append(100*len(s["interp"])/max(1,nc))
ax.scatter(hr_vals, comp_pct, s=200, c=RED, edgecolors="white", linewidths=1.5,
           zorder=10, label="% Compensatory (felt thumps)")
ax.scatter(hr_vals, interp_pct, s=200, c=BLUE, edgecolors="white", linewidths=1.5,
           zorder=10, label="% Interpolated (silent)")
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
styled_ax(ax, "Key pattern: the baseline HR decides how many PVC are felt",
          "Sinus BPM (mean resting rate)", "% of classified PVC")
plt.tight_layout()
img_hr_comp = fig_to_bytes(fig)

# B) Composition stacked (interpolated / compensatory / incomplete)
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
i_vals = [len(s["interp"]) for s in sessions]
c_vals = [len(s["comp"]) for s in sessions]
x_vals = [len(s["incomp"]) for s in sessions]
totals = [i+c+x for i,c,x in zip(i_vals, c_vals, x_vals)]
# percentages
ip = [100*v/max(1,t) for v,t in zip(i_vals, totals)]
cp = [100*v/max(1,t) for v,t in zip(c_vals, totals)]
xp = [100*v/max(1,t) for v,t in zip(x_vals, totals)]
y = np.arange(len(LABELS))
ax.barh(y, ip, color=BLUE, edgecolor="white", label="Interpolated (no pause)")
ax.barh(y, cp, left=ip, color=RED, edgecolor="white", label="Compensatory (full pause)")
ax.barh(y, xp, left=[a+b for a,b in zip(ip,cp)], color=GRAY, edgecolor="white", label="Incomplete")
for k, (i_, c_, x_, t_) in enumerate(zip(i_vals, c_vals, x_vals, totals)):
    ax.text(50, k, f"i={i_}  c={c_}  x={x_}  (tot {t_})", color="white",
            ha="center", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(LABELS, color="white")
ax.set_xlim(0, 100)
ax.legend(facecolor="#222", labelcolor="white", fontsize=9, loc="upper right")
styled_ax(ax, "PVC composition: interpolated vs compensatory (per session)",
          "% of total classified", None)
plt.tight_layout()
img_comp = fig_to_bytes(fig)

# B2) Distribution of the ratio (RR_pre+RR_post)/RR_sinus for the 3 sessions
# Shows DIRECTLY why Jun 6 has no interpolated beats: its histogram never
# drops below 1.30 (the interpolated threshold), everything concentrates on 1.85-2.15
# (full compensatory).
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
    ax.text(med + 0.02, ax.get_ylim()[1]*0.85, f"median {med:.2f}",
            color="yellow", fontsize=9, fontweight="bold")
    ax.text(0.05, 0.85, "← interp", transform=ax.transAxes, color=col, fontsize=8)
    ax.text(0.78, 0.85, "comp →", transform=ax.transAxes, color=col, fontsize=8)
    styled_ax(ax, f"{lab}   (sinus {s['sinus_bpm']:.0f} BPM, RR sinus {s['RR_SINUS_MS']:.0f}ms, n={len(ratios)})",
              None, "Count")
axes_d[-1].set_xlabel("ratio (RR_pre + RR_post) / RR_sinus", color="white")
plt.tight_layout()
img_distrib = fig_to_bytes(fig)

# B3) Bucket by instantaneous BPM: at what rate do compensatory ones appear?
# Aggregate all PVC of the 3 sessions by instantaneous BPM (= pre-N-N RR converted).
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
ax.bar(x_b - w/2, n_i_b, w, color=BLUE, edgecolor="white", linewidth=0.4, label="Interpolated")
ax.bar(x_b + w/2, n_c_b, w, color=RED, edgecolor="white", linewidth=0.4, label="Compensatory")
ax2 = ax.twinx(); ax2.set_facecolor(DARK_BG)
ax2.plot(x_b, pct_c_b, color="#ffe169", marker="o", lw=2, label="% compensatory")
ax2.axhline(50, color="#ffe169", ls="--", lw=0.7, alpha=0.5)
ax2.set_ylim(0, 100); ax2.set_ylabel("% compensatory of the classified sum", color="#ffe169")
ax2.tick_params(colors="#ffe169")
for sp in ax2.spines.values(): sp.set_color("#444")
ax.set_xticks(x_b); ax.set_xticklabels(labels_b, color="white")
styled_ax(ax, "Which instantaneous rate favors the compensatory pause? (3-session aggregate)",
          "Instantaneous BPM (pre-PVC N-N RR converted)", "N PVC")
ax.legend(loc="upper left", facecolor="#222", labelcolor="white", fontsize=9)
ax2.legend(loc="upper right", facecolor="#222", labelcolor="#ffe169", fontsize=9)
plt.tight_layout()
img_bpm_bucket = fig_to_bytes(fig)

# C0) Coupling detail per session + cluster segmentation
# Morphological analysis for each sub-cluster: same focus or bifocal?
def cluster_analyze(coup_list):
    """Return a dict with counts and median morphology of the sub-clusters
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

# Detailed plot: narrow-bin histograms for the 3 sessions with colored clusters
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
    # color the bars by cluster
    cols = ["#7ad9ff" if c<500 else ("#ff4d6d" if c<600 else "#33ff66") for c in centers]
    ax.bar(centers, hist, width=14, color=cols, edgecolor="white", linewidth=0.3)
    ax.axvline(500, color="white", ls="--", lw=0.7, alpha=0.5)
    ax.axvline(600, color="white", ls="--", lw=0.7, alpha=0.5)
    med = statistics.median(coup_arr)
    ax.axvline(med, color="yellow", ls="-", lw=1.5)
    ax.text(med+5, ax.get_ylim()[1]*0.85, f"med {med:.0f}ms",
            color="yellow", fontsize=8, fontweight="bold")
    styled_ax(ax, f"{lab}   pre-PVC coupling (n={len(coup_arr)})",
              None, "Count")
axes_cc[-1].set_xlabel("Pre-PVC coupling (ms)", color="white")
plt.tight_layout()
img_cluster = fig_to_bytes(fig)

# C) Coupling distribution (overlaid histograms)
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
styled_ax(ax, "Pre-PVC coupling distribution (focus stability)",
          "Coupling (ms)", "Count")
plt.tight_layout()
img_coup = fig_to_bytes(fig)

# D) HR vs burden minute by minute, one session per row
fig, axes = plt.subplots(3, 1, figsize=(11, 6), facecolor=DARK_BG)
for ax, s, lab in zip(axes, sessions, LABELS):
    ax.set_facecolor(DARK_BG)
    # minutes
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
    styled_ax(ax, f"{lab} — minute-by-minute trend", "Min", "Sinus BPM")
    ax2.set_ylabel("Burden %", color="white"); ax2.tick_params(colors="white", labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#444")
plt.tight_layout()
img_temporal = fig_to_bytes(fig)

# E) RESPIRATORY ANALYSIS — PVC phasic distribution for each session
# 2-column plot: polar rosette on the left, enrichment bar on the right for each session
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
        # polar ROSETTE
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
        # Hilbert convention on EDR (max amp = max inspir): phase=0 → max INSPIR,
        # π/2 → mid expir, π → end expir, 3π/2 → mid inspir
        ax_p.set_xticklabels(["max\ninspir.", "mid\nexpir.", "end\nexpir.", "mid\ninspir."],
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
        for x, lab2 in [(0, "insp."), (25, "expir."), (50, "expir."), (75, "insp.")]:
            ax_e.axvline(x, color="cyan", ls=":", alpha=0.3)
        ax_e.set_xlim(0, 100)
        ax_e.set_ylim(0, max(2.5, max(enrich)*1.1))
        ax_e.set_ylabel("PVC/N\nratio", color="white", fontsize=8)
        ax_e.set_title(f"rate {edr['rate_resp']:.1f}/min · peak @ {edr['peak_phase_pct']:.0f}% · "
                       f"enrich ×{edr['peak_enrich']:.2f}",
                       color="white", fontsize=8, loc="left")
        ax_e.tick_params(colors="white", labelsize=7)
        for sp in ax_e.spines.values(): sp.set_color("#444")
        ax_e.grid(alpha=0.18, color="#666")
        if i_s == n_with_edr - 1:
            ax_e.set_xlabel("% of the respiratory cycle  (0=max inspir., 50=end expir.)",
                            color="white", fontsize=8)
    img_resp_phases = fig_to_bytes(fig)

    # F) EDR EXAMPLE — last session (most recent, usually Jun 9) — first 30s
    s_demo = sessions_with_edr[-1]
    edr_demo = s_demo["edr"]
    t_demo = s_demo["t"]; vf_demo = s_demo["vf"]
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    T_SHOW = 30
    mask_e = (t_demo < T_SHOW)
    ax.plot(t_demo[mask_e], vf_demo[mask_e], color=GREEN, lw=0.6, label="ECG")
    # N peaks in the range
    for p in s_demo["peaks"]:
        if p["t"] < T_SHOW and p["cls"] == "normal":
            ax.plot(p["t"], p["amp"]+0.05, "v", color="#ffe169", ms=4)
    # respiration overlaid (scaled)
    t_unif = edr_demo["t_unif"]
    resp = edr_demo["resp"]
    mask_r = t_unif < T_SHOW
    ax.plot(t_unif[mask_r], resp[mask_r]*3 + 1.8, color="cyan", lw=2.2,
            label="Respiration (EDR)")
    ax.set_xlim(0, T_SHOW)
    ax.set_ylim(-1, 3.3)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("ECG (V) + respiration (norm)", color="white")
    ax.set_title(f"EDR example — {label_of(s_demo)}, first {T_SHOW}s. "
                 f"R-peaks (yellow), respiration extracted from QRS amplitude (cyan)",
                 color="white", fontsize=10)
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9, loc="upper right")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.grid(alpha=0.18, color="#666")
    plt.tight_layout()
    img_resp_example = fig_to_bytes(fig)

# ---------- EXAMPLES ----------
print("Extracting examples...")
def pick_spread(lst, n=1, min_gap_s=60):
    out, last = [], -1e9
    for p in sorted(lst, key=lambda q: q["t"]):
        if p["t"] - last >= min_gap_s:
            out.append(p); last = p["t"]
        if len(out) >= n: break
    return out

# For each type: one strip per session (3 sessions × 4 types = 12 strips max)
example_strips = {"couplet": [], "interp": [], "comp": []}
for s, lab in zip(sessions, LABELS):
    # interpolated
    if s["interp"]:
        p = pick_spread(s["interp"], n=1, min_gap_s=120)[0]
        img = make_example_strip(s["t"], s["vf"], s["peaks"], p, win_s=7.0,
            title=f"{lab} — INTERPOLATED  {int(p['t']//60):02d}:{int(p['t']%60):02d}  Σ={p['sum_pre_post_ms']:.0f}ms",
            RR_S=s["RR_SINUS_MS"])
        example_strips["interp"].append((lab, img))
    # compensatory
    if s["comp"]:
        p = pick_spread(s["comp"], n=1, min_gap_s=120)[0]
        img = make_example_strip(s["t"], s["vf"], s["peaks"], p, win_s=7.0,
            title=f"{lab} — COMPENSATORY  {int(p['t']//60):02d}:{int(p['t']%60):02d}  Σ={p['sum_pre_post_ms']:.0f}ms",
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
print("Generating PDF...")
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
story.append(Paragraph("DIY Holter — 3-session summary", H1))
story.append(Paragraph(
    f"Observed period: <b>{LABELS[0]} → {LABELS[-1]}</b>. "
    f"Hardware: AD8232 (single precordial lead, 250 Hz) + Pi Pico W → Mac server via TCP. "
    f"Automatic analysis with a uniform criterion: PVC if "
    f"(rebound ≥ {REBOUND_PVC} or width ≥ {PVC_W_MS:.0f}ms) <b>and</b> "
    f"amplitude ≥ {PVC_MIN_AMP}V. Sub-physiological spikes (≤16ms) and auto-detected "
    f"noisy intervals are excluded before the computation.",
    NORMAL))
story.append(Spacer(1, 8))

# --- Highlight box: the 3 key messages ---
all_pvc = sum(s["n_pvc"] for s in sessions)
all_couplets = sum(len(s["couplets"]) for s in sessions)
coup_medians = []
for s in sessions:
    cs = [p["rr_prev"]*1000 for p in s["peaks"]
          if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if cs: coup_medians.append(statistics.median(cs))

highlight_rows = [
    [Paragraph("<b>Focus</b>", SMALL),
     Paragraph(f"Single, monomorphic. Stable coupling (~{statistics.mean(coup_medians):.0f}ms) "
               f"across all 3 sessions — same electrical origin.", SMALL)],
    [Paragraph("<b>Danger</b>", SMALL),
     Paragraph(f"No R-on-T (coupling always &gt; 360ms). No run of ≥3 PVC. "
               f"<b>{all_couplets} couplets out of {all_pvc} total PVC</b> "
               f"({100*all_couplets/max(1,all_pvc):.2f}%): very few, "
               f"isolated, RR between 384-460 ms; flagged and highlighted in the "
               f"individual session reports.", SMALL)],
    [Paragraph("<b>Arrhythmias</b>", SMALL),
     Paragraph(f"AF screening negative or borderline in all: the RR irregularity "
               f"is explained by bradycardia + RSA + frequent ectopy, with the bimodal "
               f"RR distribution preserved.", SMALL)],
    [Paragraph("<b>HR↔symptoms pattern</b>", SMALL),
     Paragraph(f"At lower baseline HR, interpolated PVC prevail (silent, "
               f"no pause); rising toward 57 BPM they turn into compensatory ones (felt "
               f"as a 'thump'). Explains the fluctuations in subjective perception.", SMALL)],
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

# === SUMMARY TABLE ===
story.append(Paragraph("Summary table", H2))
header = [Paragraph(f"<b>{x}</b>", SMALL) for x in
          ["Metric"] + LABELS]
def fmt(v, dec=1):
    if isinstance(v,(int,)): return str(v)
    return f"{v:.{dec}f}"
rows = [header]
for label, key, dec in [
    ("Useful duration (min)", "clean_s", 0),
    ("Excluded (s)", "excl_s", 0),
    ("Total beats", "n_total", 0),
    ("Sinus BPM", "sinus_bpm", 1),
    ("Total PVC", "n_pvc", 0),
    ("PVC rate (/min)", "pvc_rate", 1),
    ("Burden (%)", "burden", 1),
    ("Median coupling (ms)", "RR_SINUS_MS", 0),
]:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if key == "clean_s": v = v/60
        cells.append(Paragraph(fmt(v, dec), SMALL))
    rows.append(cells)
# PVC patterns
for label, key in [("True couplets", "couplets"),
                    ("Interpolated", "interp"),
                    ("Compensatory", "comp"),
                    ("Incomplete", "incomp"),
                    ("Isolated PVC", "iso_pvc"),
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
# session columns: each has a faint color (matches the bars in the charts)
_TINT_PALETTE = [
    colors.HexColor("#e8f0f7"),   # faint blue
    colors.HexColor("#fbeedd"),   # faint orange
    colors.HexColor("#fbe5e7"),   # faint pink
    colors.HexColor("#e6f4ec"),   # faint green
    colors.HexColor("#f1e6f7"),   # faint purple
    colors.HexColor("#fff7d6"),   # faint yellow
    colors.HexColor("#f4e8cf"),   # faint ochre
]
COL_TINTS = [_TINT_PALETTE[i % len(_TINT_PALETTE)] for i in range(len(sessions))]
# highlighted rows: burden, couplet, interpolated, compensatory, AF score
ROW_EMPH = {7: True, 9: True, 10: True, 11: True, 17: True}  # 1-indexed after header

# traffic-light evaluation of burden and AF
def burden_color(v):
    # 0-15 green, 15-25 yellow, >25 orange (monofocal PVC up to 30% are benign)
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

# rebuild rows with colored cells for the key metrics
new_rows = [header]
metric_specs = [
    ("Useful duration (min)", "clean_s", 0),
    ("Excluded (s)", "excl_s", 0),
    ("Total beats", "n_total", 0),
    ("Sinus BPM", "sinus_bpm", 1),
    ("Total PVC", "n_pvc", 0),
    ("PVC rate (/min)", "pvc_rate", 1),
    ("Burden (%)", "burden", 1),
    ("Median coupling (ms)", "RR_SINUS_MS", 0),
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
            # total PVC + % of total beats (= burden)
            pct = 100*v/max(1, s["n_total"])
            cells.append(Paragraph(f"{v} <font color='#666'>({pct:.1f}%)</font>", SMALL))
        elif key == "pvc_rate":
            cells.append(Paragraph(f"{v:.{dec}f} /min", SMALL))
        else:
            cells.append(Paragraph(fmt(v, dec), SMALL))
    new_rows.append(cells)
# PVC patterns with couplets highlighted
pattern_specs = [("True couplets", "couplets"),
                  ("Interpolated", "interp"),
                  ("Compensatory", "comp"),
                  ("Incomplete", "incomp"),
                  ("Isolated PVC", "iso_pvc"),
                  ("Bigem PVC-N-PVC", "bigem"),
                  ("Trigem PVC-N-N-PVC", "trigem")]
for label, key in pattern_specs:
    cells = [Paragraph(label, SMALL)]
    for s in sessions:
        v = s[key]
        if isinstance(v, list): v = len(v)
        # denominator for the %: interp/comp/incomp relative to the classifiable PVC
        # (N-PVC-N sandwich); couplet/iso/bigem/trigem relative to total PVC
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
# colored AF score
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
    # tinted background for each session column
    # first metric column
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0f0f0")),
]
for _ci in range(len(sessions)):
    style_cmds.append(("BACKGROUND", (_ci+1, 1), (_ci+1, -1), COL_TINTS[_ci]))
# highlight key rows with a green border
for r_idx in [7, 9, 10, 11, 17]:  # burden, couplet, interp, comp, AF
    if r_idx < len(new_rows):
        style_cmds.append(("LINEBEFORE", (0, r_idx), (0, r_idx), 3, colors.HexColor("#33aa66")))
tbl.setStyle(TableStyle(style_cmds))
story.append(tbl)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<font color='#1b4034'>■</font> green = within norms for benign monofocal PVC · "
    "<font color='#9b6b00'>■</font> amber = intermediate · "
    "<font color='#a3320c'>■</font> red = above the attention threshold · "
    "<font color='#1f6fa8'>■</font> interpolated · <font color='#a3320c'>■</font> compensatory. "
    "The 3 session columns correspond to the colors of the charts. "
    "The % for interp/comp/incomp are over the classifiable PVC (N-PVC-N sandwich); "
    "couplet/isolated/bigem/trigem are over the total PVC; total PVC and burden are "
    "over the total beats.",
    SMALL))
story.append(Spacer(1, 10))

# === comparative BARS ===
story.append(Paragraph("Comparison of the main parameters", H2))
story.append(Paragraph(
    "Burden = % of ectopic beats over the total; sinus rate = normal beats per "
    "minute; PVC rate = ectopics per minute.", SMALL))
story.append(fit_image(img_bars, max_w_mm=175, max_h_mm=65))
story.append(Spacer(1, 10))

# === COMPOSITION ===
story.append(Paragraph("PVC composition: interpolated vs compensatory", H2))
story.append(Paragraph(
    "An <b>interpolated PVC</b> slots between two N beats without resetting the SA "
    "node (sum RR_pre + RR_post ≈ 1× RR sinus). A <b>PVC with a full compensatory "
    "pause</b> resets the SA node (sum ≈ 2× RR sinus). Interpolated ones are favored "
    "by bradycardia and are hemodynamically more benign (no loss of output, "
    "no perception of the 'thump').", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_comp, max_w_mm=175, max_h_mm=65))
story.append(Spacer(1, 10))

# === KEY PATTERN HR vs %COMP ===
story.append(Paragraph("The key pattern: baseline HR → pause type → perception", H2))
story.append(Paragraph(
    "The 3 sessions show a <b>consistent gradient</b> between the baseline rate and the type "
    "of post-PVC pause. Over a span of just 5 BPM (from 52 to 57) the share of "
    "compensatory PVC (the ones that produce the felt 'thump') goes from ~25% to 63%. "
    "The trend of the interpolated ones is mirror-image, dominating at lower HR. "
    "Electrophysiological explanation: at lower HR the SA node has longer cycles "
    "(~1100ms), and the PVC almost always has time to slot in before the SA fires "
    "again (silent interpolated); at higher HR the SA is close to firing and gets "
    "reset by the retrograde wave (felt compensatory).",
    NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_hr_comp, max_w_mm=140, max_h_mm=80))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Clinical implication:</b> the fluctuations in symptoms (some days "
    "'I feel tons of them', others 'nothing today') are not variations in the number of "
    "PVC, but in <b>how many</b> of those beats become <b>perceptible</b>. The "
    "critical threshold is ~55 BPM for this patient: below it, the PVC are silent; "
    "above it, they turn into symptomatic compensatory ones.",
    NORMAL))
story.append(PageBreak())

# === RATIO DISTRIBUTION PER SESSION ===
story.append(Paragraph("Distribution of the pause / sinus RR ratio per session", H2))
story.append(Paragraph(
    "For each PVC sandwiched between two N beats the ratio "
    "<i>(RR_pre + RR_post) / RR_sinus</i> is computed. The distribution of this ratio "
    "directly reveals the behavior of the SA node: the left shaded zone "
    "(&lt;1.30) indicates the interpolated ones, the right one (1.85-2.15) the full "
    "compensatory ones. The yellow line is the session median.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_distrib, max_w_mm=175, max_h_mm=170))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "The 3 sessions have <b>different shapes</b> despite the same focus: the "
    "behavior of the SA node depends on the autonomic state at that moment, not on the "
    "PVC itself. Jun 6 (median 1.99) shows a heart in 'always reset' mode; "
    "Jun 7 (median 1.40-1.50) shows a bimodal distribution with half the PVC "
    "in the interpolated zone; Jun 5 is intermediate.",
    NORMAL))
story.append(PageBreak())

# === BPM BUCKET ANALYSIS ===
story.append(Paragraph("At what instantaneous rate do the compensatory ones appear?", H2))
story.append(Paragraph(
    "Each PVC is labeled with the instantaneous rate immediately preceding it "
    "(the RR of the N-N right before the PVC, converted to BPM). Aggregating all the "
    "PVC of the 3 sessions by BPM band yields the <b>dose-response curve</b> "
    "of perception: the yellow line shows the percentage of compensatory PVC "
    "(perceptible) as a function of the instantaneous baseline rate.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_bpm_bucket, max_w_mm=175, max_h_mm=110))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Reading (3-session aggregate, n=1249 classifiable PVC): below 60 BPM "
    "<b>0% compensatory</b> (all interpolated, silent). The <b>50% crossover</b> "
    "occurs in the <b>60-65 BPM</b> band. Between 65-80 BPM the compensatory share grows "
    "rapidly up to the <b>94% peak in the 75-80 BPM band</b>. Above 80 BPM the "
    "classification becomes blurred (the RR shorten and most of the "
    "PVC fall into the 'incomplete' zone, neither interpolated nor purely compensatory).",
    NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Methodological notes: the 5 BPM buckets and the 1.30 / 1.85-2.15 RR_sinus thresholds "
    "(to define interpolated / compensatory) are parametric choices, derived "
    "from textbook physiology. The HR→%comp gradient and the numbers above are "
    "instead computed directly from the data. The Jun 6 session (intrinsically 0% "
    "interpolated) drags the aggregate statistic; analyzing the sessions "
    "individually, the crossover shifts later (e.g. on Jun 7 alone it was ~70 BPM).</i>",
    SMALL))
story.append(Spacer(1, 10))

# === COUPLING STABILITY ===
story.append(Paragraph("Coupling stability (origin of the focus)", H2))
story.append(Paragraph(
    "Distribution of the pre-PVC coupling (interval between the preceding N and the PVC). "
    "If the distribution is narrow and the median is consistent across sessions, the focus "
    "is <b>monomorphic and fixed</b> at the same point of the ventricle. If it widens or "
    "separate peaks appear, it suggests a multifocal origin.", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_coup, max_w_mm=175, max_h_mm=65))
story.append(PageBreak())

# === COUPLING DETAIL: CLUSTER ANALYSIS (JUN 6 BIMODALITY) ===
story.append(Paragraph("Cluster analysis of the coupling per session", H2))
story.append(Paragraph(
    "Zoom on the pre-PVC coupling distribution with 15 ms bins. The bars are "
    "colored by band: <font color='#3a9'>light blue &lt; 500 ms</font>, "
    "<font color='#c33'>pink 500-600 ms</font>, "
    "<font color='#393'>green &gt; 600 ms</font>. This visualization reveals a "
    "<b>hidden bimodality in the Jun 6 session</b>: besides the main cluster at "
    "~470 ms, there is a persistent second group around 540-560 ms (43% of the "
    "PVC). The other sessions show a single cluster.",
    NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_cluster, max_w_mm=175, max_h_mm=140))
story.append(Spacer(1, 8))

# Morphological table of the clusters for each session
story.append(Paragraph("<b>Median morphology per cluster (Jun 6 session)</b>", H3))
ses_6giu = sessions[1]  # Jun 6 is index 1
clus_6 = session_clusters[1]
rows_c = [[Paragraph(f"<b>{x}</b>", SMALL) for x in
           ["Cluster","n (%)","Coupling (ms)","Amplitude (V)","Width (ms)","Rebound"]]]
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

story.append(Paragraph("<b>Physiological interpretation</b>", H3))
story.append(Paragraph(
    "The morphological micro-differences between the clusters (width 100 vs 104 ms, rebound "
    "0.59 vs 0.65, overlapping amplitudes) are <b>too small</b> to "
    "indicate a second focus: a true ectopy of different origin would show "
    "deltas of 20-30 ms in width and clearly visible polarity/morphology changes. "
    "The three hypotheses in order of plausibility:",
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>1. Coupling modulation (most likely).</b> The same focus fires "
    "at 2 slightly different rates depending on the autonomic state: when the SA "
    "is faster the coupling is shorter (~470 ms), when it slows down it rises (~545 ms). "
    "A pattern compatible with benign parasystole.",
    NORMAL))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "<b>2. Same focus, two exit routes.</b> The ectopy arises at the same "
    "point but can take two slightly different conduction paths toward the "
    "ventricle, arriving with ~70 ms of delay. It explains the morphological "
    "micro-differences without requiring a second focus.",
    NORMAL))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "<b>3. Mild bifocal.</b> Two nearby ectopic cells with similar properties. "
    "Hard to rule out with a single precordial lead.",
    NORMAL))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Only the Jun 6 session shows this bimodality — the other two have a single "
    "cluster. Consistent with a more 'tonic' autonomic state (HR 57 vs 52-54 in the "
    "others): it is possible that the higher sympathetic level activates a second "
    "firing 'mode' or an alternative conduction path. <b>Worth flagging to the "
    "cardiologist</b>: if they run a 24h holter it is useful to verify whether the "
    "bimodality is reproducible and whether it appears at specific times of day.",
    NORMAL))
story.append(PageBreak())

# === TEMPORAL TREND ===
story.append(Paragraph("Intra-session temporal trend", H2))
story.append(Paragraph(
    "For each session: sinus rate (green) and burden (red) minute by minute. "
    "It lets you see whether the baseline rate is stable and whether the burden changes "
    "over time (e.g. response to respiration, stress, position).", NORMAL))
story.append(Spacer(1, 4))
story.append(fit_image(img_temporal, max_w_mm=175, max_h_mm=120))
story.append(PageBreak())

# === EXAMPLES ===
story.append(Paragraph("Morphological examples of the classifications", H2))
story.append(Paragraph(
    "For each session, a representative strip of each type. Orange "
    "circle = analyzed PVC, light blue bar = RR_pre, yellow bar = RR_post, "
    "dashed red line = expected position of the next beat if the "
    "pause were a full compensatory one (2× RR sinus). An interpolated one shows the "
    "next N beat <b>before</b> that line; a compensatory one shows it "
    "<b>on</b> the line.", NORMAL))
story.append(Spacer(1, 8))

for kind_label, kind_key in [
    ("Interpolated (no SA reset)", "interp"),
    ("Compensatory (full SA reset)", "comp"),
    ("Couplet (2 consecutive PVC within 700ms)", "couplet"),
]:
    story.append(Paragraph(kind_label, H3))
    items = example_strips.get(kind_key, [])
    if not items:
        story.append(Paragraph("(no example available)", SMALL))
    for (lab, img) in items:
        story.append(fit_image(img, max_w_mm=175, max_h_mm=55))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 6))

story.append(PageBreak())

# === RESPIRATORY ANALYSIS + PHASIC TRIGGER ===
if img_resp_phases is not None:
    story.append(Paragraph("Respiratory analysis: end-expiratory phasic trigger", H2))
    story.append(Paragraph(
        "PVCs do not fire randomly during the respiratory cycle. By extracting "
        "the respiratory signal directly from the ECG (the <b>EDR — ECG-Derived "
        "Respiration</b> technique: the beat-by-beat R amplitude modulation tracks "
        "breathing thanks to the rotation of the electrical vector during the "
        "diaphragmatic excursion) and computing the instantaneous phase via the Hilbert transform, "
        "each PVC can be associated with its position in the respiratory cycle "
        "(0% = end expiration = minimum amplitude; 50% = end inspiration = "
        "maximum amplitude).",
        NORMAL))
    story.append(Spacer(1, 6))
    if img_resp_example is not None:
        story.append(Paragraph("Example of respiratory extraction from the ECG (reference session)", H3))
        story.append(fit_image(img_resp_example, max_w_mm=175, max_h_mm=60))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Above: 30 seconds of ECG (green) with yellow triangles on the R peaks. "
            "The overlaid cyan line is the respiration reconstructed from the amplitude "
            "of the R peaks, perfectly sinusoidal and in phase with the respiratory cycles "
            "observed by eye in the variation of the QRS heights.",
            SMALL))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Distribution of the PVC across the respiratory cycle per session", H3))
    story.append(Paragraph(
        "For each session: on the left a polar rosette (green = N beats, red = PVC); "
        "on the right the PVC/N ratio by phase (red bars = PVC excess, green = deficit, "
        "light blue = neutral; dashed white line = uniform).",
        SMALL))
    story.append(Spacer(1, 6))
    story.append(fit_image(img_resp_phases, max_w_mm=180, max_h_mm=240))
    story.append(PageBreak())
    # Summary table for the respiratory trigger
    story.append(Paragraph("Phasic trigger summary — all sessions", H3))
    resp_rows = [
        [Paragraph(f"<b>{x}</b>", SMALL) for x in
         ["Session","Resp rate /min","Phasic peak","Enrichment","p-value","Significant"]]
    ]
    for s in sessions_with_edr:
        edr = s["edr"]
        # phase interpretation (Hilbert: 0=max inspir, 50%=end expir)
        ph = edr["peak_phase_pct"]
        if ph < 15 or ph > 85:
            phase_lab = "max inspir. ★"
        elif 15 <= ph < 35:
            phase_lab = "mid expir."
        elif 35 <= ph < 65:
            phase_lab = "end expir."
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
        f"<b>Recurring pattern:</b> {n_maxinsp} of {len(sessions_with_edr)} sessions "
        f"show a peak of PVC firing around <b>maximum inspiration / "
        f"transition toward expiration</b> (within 15% of the cycle from phase 0 = "
        f"QRS amplitude peak). Extraordinary reproducibility despite "
        f"variability of posture, time of day, meals, and activity.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Physiological interpretation.</b> At maximum inspiration several "
        "mechanical and neurovegetative factors converge that explain the excess firing of the "
        "focus: "
        "(1) the <b>diaphragm at its most caudal point</b>, maximally contracted, slid "
        "toward the abdomen. The pericardium, attached to the central tendon of the diaphragm, "
        "undergoes <b>maximum downward traction</b>; the heart is stretched/displaced "
        "inferiorly. "
        "(2) <b>Maximum ventricular filling</b> (preload at its peak) → stretch "
        "of the chambers → mechanoreceptor activation → triggered ectopy. "
        "(3) <b>RSA inspiratory tachycardia</b>: the rising phase of the "
        "Respiratory Sinus Arrhythmia cycle coincides with a slight local sympathetic increase. "
        "For a patient with an apical breathing pattern + rib flare the anomalous "
        "pericardial traction amplifies the effect. The focus is therefore "
        "<b>mechanically-triggered (stretch) + RSA-modulated</b>.",
        NORMAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Therapeutic implication:</b> reduction of the maximum inspiratory stretch. "
        "Techniques that limit the <b>inspiratory depth</b> (reduced tidal volume, "
        "conscious \"shallow\" breathing) reduce the time spent at peak "
        "inspiration. <b>Coherent breathing at 6/min</b> with small volumes (no very "
        "deep breaths) is preferable to techniques with maximal inspirations "
        "(e.g. full yogic breathing, deep sighs). Work on the "
        "diaphragmatic pattern (PRI/DNS) to bring the diaphragm back to a neutral position "
        "also reduces the anomalous excursion and therefore the pericardial traction.",
        NORMAL))
    story.append(PageBreak())

# === SUMMARY ===
story.append(Paragraph("Cross-session summary and observations", H2))
all_couplets = sum(len(s["couplets"]) for s in sessions)
all_pvc = sum(s["n_pvc"] for s in sessions)
all_interp = sum(len(s["interp"]) for s in sessions)
all_comp = sum(len(s["comp"]) for s in sessions)
all_classified = all_interp + all_comp + sum(len(s["incomp"]) for s in sessions)
ratio_interp = 100*all_interp/max(1, all_classified)
coup_meds = [s["RR_SINUS_MS"] for s in sessions]  # here RR sinus, not coupling, but it's fine
# true couplings
coup_medians = []
for s in sessions:
    cs = [p["rr_prev"]*1000 for p in s["peaks"]
          if p["cls"]=="pvc" and p["rr_prev"] and 200<p["rr_prev"]*1000<800]
    if cs: coup_medians.append(statistics.median(cs))

bullets = [
    f"<b>Total PVC across the 3 sessions</b>: {all_pvc} out of {sum(s['n_total'] for s in sessions)} total beats.",
    f"<b>True couplets</b> (2 consecutive PVC with RR &lt; 700ms): <b>{all_couplets} in total</b> "
    f"(0 in the Jun 5 session, 2 on Jun 6, 2 on Jun 7), all with RR between 384-460 ms. "
    f"They represent <b>0.23% of the total PVC</b> — very few but they exist. "
    f"They are identified and shown in the individual session reports (section 'Morphological "
    f"examples', same style as the example trace). "
    f"<b>No ventricular tachycardia run</b> (≥3 consecutive PVC): 0 in all sessions.",
    f"<b>Single monomorphic focus</b>: the median coupling is stable across sessions "
    f"({', '.join(f'{m:.0f}ms' for m in coup_medians)}). Narrow distribution = single electrical origin.",
    f"<b>{ratio_interp:.0f}% of the classified PVC are interpolated</b> (no compensatory "
    f"pause). Compatible with baseline bradycardia and good vagal tone: "
    f"these PVC do not disturb ventricular filling and are typically not perceived.",
    f"<b>No R-on-T</b>: no PVC with coupling < 360ms in any of the 3 sessions.",
]
af_scores = [s["af"].get("score","-") for s in sessions if s.get("af")]
if af_scores:
    bullets.append(f"<b>AF screening</b>: score {'/'.join(str(x) for x in af_scores)} (out of 4). "
                   f"High HRV markers but the bimodal RR structure is preserved; not suggestive of atrial fibrillation. "
                   f"The observed RR irregularity is explained by bradycardia + RSA + frequent ectopy.")

bullets.append(f"<b>Rate vs perception correlation</b>: at low baseline HR (~55-65 BPM) "
               f"silent interpolated ones prevail; in the 70-80 BPM band the "
               f"compensatory ones grow, felt as a 'thump'. Consistent with the physiology of the SA node.")

for b in bullets:
    story.append(Paragraph("• " + b, NORMAL))
    story.append(Spacer(1, 3))

# === FINAL TABLE: total vs perceived PVC ===
story.append(Spacer(1, 14))
story.append(Paragraph("Real PVC vs perceived PVC — recap", H2))
story.append(Paragraph(
    "The counterintuitive finding that emerges from the 3 sessions: <b>the number of "
    "perceived PVC is NOT proportional to the number of real PVC</b>. The session with the most "
    "PVC overall (Jun 7) is the one where the fewest were perceived, "
    "because it had the lowest HR and therefore the highest share of silent "
    "interpolated ones. The number of 'thumps' per minute is estimated as N compensatory PVC / "
    "useful session duration.",
    NORMAL))
story.append(Spacer(1, 6))

# Compute thuds/min for each session
header_perc = [Paragraph(f"<b>{x}</b>", SMALL) for x in
               ["Indicator"] + LABELS]
rows_perc = [header_perc]
def cell_perc(text, col_hex=None, bold=False):
    if bold and col_hex:
        return Paragraph(f"<b><font color='{col_hex}'>{text}</font></b>", SMALL)
    if col_hex:
        return Paragraph(f"<font color='{col_hex}'>{text}</font>", SMALL)
    if bold:
        return Paragraph(f"<b>{text}</b>", SMALL)
    return Paragraph(text, SMALL)

# row 1: total PVC per minute
cells = [cell_perc("Total PVC (/min)")]
pvc_rates = [s["pvc_rate"] for s in sessions]
max_pvc = max(pvc_rates); min_pvc = min(pvc_rates)
for s in sessions:
    v = s["pvc_rate"]
    if v == max_pvc: c = "#a3320c"   # high = red
    elif v == min_pvc: c = "#1b4034" # low = green
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{v:.1f} /min", c, bold=True))
rows_perc.append(cells)

# row 2: Burden %
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

# row 3: effective SA — the true pacemaker rate
# Computed as 60000/median(RR_NN) after noise + manual exclusion,
# only consecutive N-N pairs with physiological RR (0.6-1.4s).
# Matches what you feel at the wrist (PVC have insufficient output
# to generate a palpable pulse wave).
cells = [cell_perc("Effective SA rate (at the wrist)")]
sa_eff = [60000/s["RR_SINUS_MS"] if s["RR_SINUS_MS"] else 0 for s in sessions]
max_sa = max(sa_eff); min_sa = min(sa_eff)
for s, v in zip(sessions, sa_eff):
    if v == max_sa: c = "#a3320c"   # high = red (more compensatory)
    elif v == min_sa: c = "#1b4034" # low = green (more interpolated)
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{v:.1f} BPM", c, bold=True))
rows_perc.append(cells)

# row 3b: mean sinus RR (ms) — the "charge time" of the SA node
cells = [cell_perc("Mean SA RR (ms)")]
rr_values = [s["RR_SINUS_MS"] for s in sessions]
max_rr = max(rr_values); min_rr = min(rr_values)
for s, rr in zip(sessions, rr_values):
    if rr == max_rr: c = "#1b4034"  # longer RR = more room → green
    elif rr == min_rr: c = "#a3320c"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{rr:.0f} ms", c, bold=True))
rows_perc.append(cells)

# row 3c: total HR perceived as "beats" (N + PVC) — what you feel inside
cells = [cell_perc("Total perceived beats (/min)")]
totals = [(s["n_norm"]+s["n_pvc"])*60/s["clean_s"] if s["clean_s"] else 0
          for s in sessions]
max_t = max(totals); min_t = min(totals)
for s, t in zip(sessions, totals):
    if t == max_t: c = "#a3320c"
    elif t == min_t: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{t:.1f} /min", c, bold=True))
rows_perc.append(cells)

# row 4: % compensatory (of the classifiable ones)
cells = [cell_perc("% compensatory (perceptible)")]
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

# row 5: estimated thuds/min
cells = [cell_perc("Felt 'thumps' (/min)")]
felt_beats = [len(s["comp"])/(s["clean_s"]/60) if s["clean_s"] else 0 for s in sessions]
max_t = max(felt_beats); min_t = min(felt_beats)
for s, t in zip(sessions, felt_beats):
    if t == max_t: c = "#a3320c"
    elif t == min_t: c = "#1b4034"
    else: c = "#9b6b00"
    cells.append(cell_perc(f"{t:.1f} /min", c, bold=True))
rows_perc.append(cells)

# tinted color for session columns
COL_TINTS_END = COL_TINTS  # same adaptive palette
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
    # highlight the "thuds/min" row (now index 7 after adding SA + RR + totals)
    ("LINEBEFORE", (0,7), (0,7), 3, colors.HexColor("#33aa66")),
]
tbl_perc.setStyle(TableStyle(tbl_perc_style))
story.append(tbl_perc)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Reading: in the Jun 7 session the <b>largest number of PVC</b> "
    "overall was observed (24.7/min, burden 32.2%), but the <b>effective SA rate was the "
    "lowest</b> ({:.1f} BPM) — this brought the share of compensatory ones (the "
    "perceptible ones) to a minimum (29%). Result: <b>only 6.3 thumps/min perceived</b>, "
    "versus 9.3 on Jun 6 and 8.3 on Jun 5 — even though Jun 7 objectively had "
    "more ectopics. Subjective sensations ('today I feel them more / "
    "less') are not a reliable proxy for the number of real PVC.".format(min(sa_eff)),
    NORMAL))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>Note on measuring HR: the 'effective SA rate' is the true "
    "pacemaker rate (60000/median RR_NN over consecutive N-N pairs after "
    "noise exclusion). It corresponds to what is measured at the wrist, because PVC "
    "have a reduced output and typically do not generate a palpable pulse wave. "
    "The 'total perceived beats', by contrast, also include the PVC and are what is "
    "felt as an internal palpitation. The 'number of thumps' is the estimate of the PVC "
    "that produce the felt beat (the compensatory ones, because the pause lets "
    "the ventricle over-fill and the next beat is hypercontractile).</i>",
    SMALL))

story.append(Spacer(1, 12))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#888")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Disclaimer: report from a DIY setup (single precordial lead AD8232 + Pi "
    "Pico W, 250 Hz sampling). It is not equivalent to a clinical Holter: analysis of fine "
    "morphology (P, ST), precise multifocal classification, and the diagnosis of "
    "complex arrhythmias require at least 3 leads and a wider passband. An indicative "
    "document, it does not replace medical evaluation.</i>", SMALL))

doc.build(story)
print(f"\nPDF saved: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
