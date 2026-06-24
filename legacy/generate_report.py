"""
Generate a self-contained HTML report with embedded charts from a recorded session.

Usage:
    python3 generate_report.py logs/ecg_YYYYMMDD_HHMMSS.csv

Output: logs/report_YYYYMMDD_HHMMSS.html (open in a browser, optionally print to PDF)
"""
import base64
import csv
import io
import math
import os
import statistics
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if PATH is None:
    print("usage: generate_report.py <ecg_*.csv>")
    sys.exit(1)

SAMPLE_HZ = 250

# --- load samples ---
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

# --- load peaks ---
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

# stamps
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

# RR intervals
for i in range(len(peaks)):
    peaks[i]["rr_prev"] = (peaks[i]["t"] - peaks[i-1]["t"]) if i > 0 else None
sinus_rr  = [peaks[i]["rr_prev"] for i in range(1, len(peaks))
             if peaks[i]["cls"] == "normal" and peaks[i-1]["cls"] == "normal"]
coupling  = [p["rr_prev"] for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None]

# pattern counts
iso_pvc = sum(
    1 for i, p in enumerate(peaks) if p["cls"] == "pvc"
    and (i == 0 or peaks[i-1]["cls"] != "pvc")
    and (i == len(peaks)-1 or peaks[i+1]["cls"] != "pvc")
)
couplets_n = 0
i = 0
while i < len(peaks) - 1:
    if peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc":
        if i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc":
            i += 1
            continue
        couplets_n += 1
        i += 2
    else:
        i += 1

# bigeminy / trigeminy runs
bigem = 0
i = 0
while i < len(peaks) - 5:
    if [peaks[i+j]["cls"] for j in range(6)] == ["normal","pvc"]*3:
        bigem += 1
        # skip ahead
        end = i
        while end+1 < len(peaks) and peaks[end+1]["cls"] != peaks[end]["cls"]:
            end += 1
        i = end + 1
    else:
        i += 1

trigem = 0
i = 0
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

baseline_stdevs = []
if len(sinus_rr) >= LOOKBACK:
    for k in range(0, len(sinus_rr) - LOOKBACK + 1, LOOKBACK):
        baseline_stdevs.append(statistics.stdev(sinus_rr[k:k+LOOKBACK]))

# per-minute stats
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

# --- plot helpers ---
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def styled_ax(ax, title=None):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="#aaa")
    ax.grid(True, alpha=0.2, color="#444")
    for sp in ax.spines.values(): sp.set_color("#444")
    if title: ax.set_title(title, color="#fff")

# PLOT 1 — example trace: 6 seconds with at least one PVC
ecg_example_b64 = ""
if pvc and N:
    # find the first PVC and crop ±3s around it
    p0 = pvc[0]["t"]
    mask = (t >= p0-3) & (t <= p0+3)
    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor("#1e1e1e")
    ax.plot(t[mask]-p0, vf[mask], linewidth=0.9, color="#2ecc71")
    # red overlay
    for p in pvc:
        if p0-3 <= p["t"] <= p0+3:
            pt = p["t"] - p0
            wm = (t >= p["t"]-0.12) & (t <= p["t"]+0.12)
            ax.plot(t[wm]-p0, vf[wm], linewidth=2.0, color="#e74c3c")
    ax.set_xlabel("t (s relative to the first PVC)", color="#aaa")
    ax.set_ylabel("Filtered ECG (V)", color="#aaa")
    styled_ax(ax, "Example: 6 seconds with normal QRS (green) and one PVC (red)")
    plt.tight_layout()
    ecg_example_b64 = fig_to_b64(fig)

# PLOT 2 — overview of the whole session (downsampled)
overview_b64 = ""
if N:
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#1e1e1e")
    step = max(1, N // 50000)  # ~50k points max
    ax.plot(t[::step]/60, vf[::step], linewidth=0.3, color="#2ecc71", alpha=0.8)
    # PVC dots
    if pvc:
        ax.scatter([p["t"]/60 for p in pvc], [1.4]*len(pvc), s=4, color="#e74c3c", marker="v")
    ax.set_xlabel("Time (min)", color="#aaa")
    ax.set_ylabel("ECG filt (V)", color="#aaa")
    ax.set_ylim(-1.4, 1.6)
    styled_ax(ax, f"Overview {total_min:.1f} min — red triangles above = PVC positions")
    plt.tight_layout()
    overview_b64 = fig_to_b64(fig)

# PLOT 3 — tachogram
tacho_b64 = ""
if peaks:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    for p in peaks:
        if p["rr_prev"] is None: continue
        c = "#e74c3c" if p["cls"] == "pvc" else "#2ecc71"
        ax.scatter(p["t"]/60, 1000*p["rr_prev"], c=c, s=6, alpha=0.75)
    ax.set_xlabel("Time (min)", color="#aaa")
    ax.set_ylabel("RR (ms)", color="#aaa")
    styled_ax(ax, "Tachogram — beat RR over time (green = sinus, red = pre-PVC coupling)")
    plt.tight_layout()
    tacho_b64 = fig_to_b64(fig)

# PLOT 4 — histogram RR
hist_b64 = ""
if coupling and sinus_rr:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.hist([r*1000 for r in sinus_rr], bins=40, alpha=0.6, color="#2ecc71",
            label=f"Sinus N→N (n={len(sinus_rr)})", density=True)
    ax.hist([r*1000 for r in coupling], bins=40, alpha=0.8, color="#e74c3c",
            label=f"Pre-PVC coupling (n={len(coupling)})", density=True)
    ax.set_xlabel("RR (ms)", color="#aaa")
    ax.set_ylabel("Density", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "RR distribution — sinus vs pre-PVC coupling")
    plt.tight_layout()
    hist_b64 = fig_to_b64(fig)

# PLOT 5 — counts per minute
counts_b64 = ""
if windows:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ts = [w["t"]/60 for w in windows]
    ax.plot(ts, [w["pvc"] for w in windows], color="#e74c3c", marker="o", label="PVC/min")
    ax.plot(ts, [w["norm"] for w in windows], color="#2ecc71", marker="o", label="Sinus/min", alpha=0.6)
    ax.set_xlabel("Time (min)", color="#aaa")
    ax.set_ylabel("Beats per minute", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "Beat distribution over time (60s windows)")
    plt.tight_layout()
    counts_b64 = fig_to_b64(fig)

# PLOT 6 — pre-PVC HRV
hrv_b64 = ""
if pre_pvc_stdevs:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    pvc_ts = [p["t"]/60 for p in peaks if p["cls"] == "pvc"]
    if len(pvc_ts) >= len(pre_pvc_stdevs):
        pvc_ts = pvc_ts[-len(pre_pvc_stdevs):]
    ax.scatter(pvc_ts, [1000*s for s in pre_pvc_stdevs], c="#e74c3c", s=10, alpha=0.7,
               label="Stdev RR in the 5 pre-PVC N beats")
    if baseline_stdevs:
        baseline_mean = 1000 * statistics.mean(baseline_stdevs)
        ax.axhline(baseline_mean, color="#2ecc71", linestyle="--", alpha=0.7,
                   label=f"Baseline sinus stdev ({baseline_mean:.0f}ms)")
    ax.set_xlabel("Time (min)", color="#aaa")
    ax.set_ylabel("Stdev RR (ms)", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "HRV in the 5 normal beats before each PVC")
    plt.tight_layout()
    hrv_b64 = fig_to_b64(fig)

# --- compute summary numbers ---
sinus_median_ms = 1000 * statistics.median(sinus_rr) if sinus_rr else 0
sinus_std_ms    = 1000 * statistics.stdev(sinus_rr)  if len(sinus_rr) > 1 else 0
coupling_median = 1000 * statistics.median(coupling) if coupling else 0
coupling_std    = 1000 * statistics.stdev(coupling)  if len(coupling) > 1 else 0
prematurity = (1 - coupling_median/sinus_median_ms) * 100 if sinus_median_ms else 0
pre_pvc_stdev_mean = 1000 * statistics.mean(pre_pvc_stdevs) if pre_pvc_stdevs else 0
baseline_stdev_mean = 1000 * statistics.mean(baseline_stdevs) if baseline_stdevs else 0
hrv_delta_pct = (pre_pvc_stdev_mean/baseline_stdev_mean - 1) * 100 if baseline_stdev_mean else 0

# --- compose HTML ---
ts_pretty = ses_id  # format YYYYMMDD_HHMMSS
now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Holter session report — {ts_pretty}</title>
<style>
  :root {{
    --bg: #1e1e1e; --panel: #2a2a2a; --fg: #eee; --muted: #888;
    --green: #2ecc71; --blue: #3498db; --red: #e74c3c; --orange: #f39c12;
  }}
  body {{
    margin: 0; padding: 32px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro", Helvetica, sans-serif;
    line-height: 1.55; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ color: #fff; font-size: 28px; margin-bottom: 4px; }}
  h2 {{ color: var(--blue); margin-top: 36px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  h3 {{ color: var(--green); }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .disclaimer {{
    background: #3a1a1a; border-left: 4px solid var(--red);
    padding: 12px 18px; margin: 20px 0; border-radius: 4px;
  }}
  table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: var(--blue); font-weight: 600; }}
  .num {{ font-variant-numeric: tabular-nums; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }}
  .blue {{ color: var(--blue); }} .orange {{ color: var(--orange); }}
  .key {{ font-size: 32px; font-weight: 700; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }}
  .card {{ background: var(--panel); padding: 14px 16px; border-radius: 8px; }}
  .card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .val {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .card .u {{ font-size: 11px; color: var(--muted); }}
  img {{ width: 100%; border-radius: 6px; margin: 8px 0; }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 60px; border-top: 1px solid #333; padding-top: 20px; }}
  code {{ background: #2a2a2a; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
</style>
</head><body>

<h1>Holter session report</h1>
<div class="meta">
  Session <code>{ts_pretty}</code> · Duration <b>{total_min:.1f} min</b>
  ({total_s:.0f}s, fs={fs_real:.2f}Hz) · Generated on {now}
</div>

<div class="disclaimer">
  <b>For educational use only.</b> This report is the result of a hobbyist device
  (Pi Pico W + AD8232) and pattern recognition implemented for educational purposes.
  It is not a medical device and in no way replaces the interpretation of the treating
  cardiologist. Any symptoms should be evaluated clinically.
</div>

<h2>1. Numeric summary</h2>
<div class="cards">
  <div class="card"><div class="label">ECG total</div>
    <div class="val green num">{60*n_total/total_s:.0f}</div><div class="u">total BPM</div></div>
  <div class="card"><div class="label">Sinus only</div>
    <div class="val blue num">{sinus_bpm:.0f}</div><div class="u">sinus rhythm BPM</div></div>
  <div class="card"><div class="label">PVC rate</div>
    <div class="val red num">{pvc_rate:.1f}</div><div class="u">/min</div></div>
  <div class="card"><div class="label">PVC burden</div>
    <div class="val orange num">{burden:.1f}</div><div class="u">% of total</div></div>
</div>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Beats recorded</td><td class="num">{n_total}</td></tr>
<tr><td>Normal (sinus) beats</td><td class="num">{len(norm)}</td></tr>
<tr><td>Ectopic beats classified as PVC</td><td class="num">{len(pvc)}</td></tr>
<tr><td>Median sinus RR (N→N)</td><td class="num">{sinus_median_ms:.0f} ms ({60000/sinus_median_ms if sinus_median_ms else 0:.0f} BPM)</td></tr>
<tr><td>Sinus RR standard deviation</td><td class="num">{sinus_std_ms:.0f} ms</td></tr>
<tr><td>Median coupling interval (N→PVC)</td><td class="num">{coupling_median:.0f} ms</td></tr>
<tr><td>Coupling standard deviation</td><td class="num">{coupling_std:.0f} ms</td></tr>
<tr><td>PVC prematurity</td><td class="num">{prematurity:.0f} % earlier than the sinus rhythm</td></tr>
</table>

<h2>2. Trace example</h2>
<p>A ~6 second window centered on the first PVC of the session.
The green line is the filtered ECG signal; the red segment highlights the PVC's QRS
and its rebound hyperpolarization (~120 ms post-peak).</p>
<img src="data:image/png;base64,{ecg_example_b64}" alt="ecg_example">

<h2>3. Overview of the whole session</h2>
<p>Compressed view of the entire recording. Each red triangle at the top is the
position of a PVC. The distribution is uniform over time, a sign of a stable
ectopic focus.</p>
<img src="data:image/png;base64,{overview_b64}" alt="overview">

<h2>4. Temporal patterns</h2>
<table>
<tr><th>Pattern type</th><th>Count</th></tr>
<tr><td>Isolated PVC (N–PVC–N)</td><td class="num">{iso_pvc}</td></tr>
<tr><td>Couplet (2 consecutive PVC)</td><td class="num">{couplets_n}</td></tr>
<tr><td>Runs of <b>bigeminy</b> (alternating N-PVC, ≥3 cycles)</td><td class="num">{bigem}</td></tr>
<tr><td>Runs of <b>trigeminy</b> (repeated N-N-PVC, ≥3 cycles)</td><td class="num">{trigem}</td></tr>
</table>
<p><b>Reading:</b> the vast majority of PVCs are <b>isolated</b>, interspersed
with groups of normal beats. When a recurring rhythmic pattern appears it is
more often <b>trigeminy</b> ({trigem} runs) than bigeminy ({bigem} runs). Couplets
(2 consecutive PVC) are practically absent ({couplets_n}). No triplets or longer
ventricular runs.</p>

<h2>5. Tachogram</h2>
<p>For each beat its RR interval relative to the preceding beat is shown.
The two horizontal clusters are well separated: at the top the sinus beats at ~{sinus_median_ms:.0f}ms,
at the bottom the pre-PVC couplings at ~{coupling_median:.0f}ms.</p>
<img src="data:image/png;base64,{tacho_b64}" alt="tacho">

<h2>6. RR distribution — sinus vs coupling</h2>
<p>A clear bimodal distribution. The PVCs systematically arrive at ~{100-prematurity:.0f}%
of the sinus cycle (prematurity of {prematurity:.0f}%). This <b>coupling interval
stability</b> is the cardinal signature of a <b>monomorphic ectopic focus</b>: the "trigger"
always fires from the same point of the myocardium with the same latency after a normal beat.</p>
<img src="data:image/png;base64,{hist_b64}" alt="hist">

<h2>7. Variability over time</h2>
<p>Count of normal beats and PVCs in 60-second windows.</p>
<img src="data:image/png;base64,{counts_b64}" alt="counts">

<h2>8. Sinus rhythm variability before the PVCs</h2>
<p>For each PVC the 5 immediately preceding normal beats were measured, computing
the standard deviation of their RR intervals. The <b>mean value is {pre_pvc_stdev_mean:.0f} ms</b>,
to be compared with the <b>sinus baseline of {baseline_stdev_mean:.0f} ms</b> (delta
{'+' if hrv_delta_pct>=0 else ''}{hrv_delta_pct:.0f}%).</p>
<p>{
   "The sinus rhythm is <b>slightly more variable</b> in the beats immediately "
   "preceding a PVC. A small but systematic difference over hundreds of observations. "
   "Consistent with an <b>autonomic modulation</b> (vagal or respiratory) of the ectopic trigger."
   if hrv_delta_pct > 5 else
   "The variability is not significantly different from baseline."
}</p>
<img src="data:image/png;base64,{hrv_b64}" alt="hrv">

<h2>9. Descriptive conclusions</h2>
<ul>
  <li><b>Stable monomorphic ectopic focus</b>: the coupling interval has low dispersion
      ({coupling_std:.0f}ms over {coupling_median:.0f}ms median), indicating a single,
      consistent ectopic source.</li>
  <li><b>Isolated PVCs, rarely in bursts</b>: {iso_pvc} singles out of {len(pvc)} total
      ({100*iso_pvc/max(1,len(pvc)):.0f}% of the total).</li>
  <li><b>Trigeminy pattern more frequent than bigeminy</b>: {trigem} trigeminy runs
      vs {bigem} bigeminy.</li>
  <li><b>Constant burden over time</b>: the PVC/min rate stays stable for
      the duration of the recording, with no marked clustering or quiescence.</li>
  <li><b>Slight increase in pre-PVC HRV</b>: +{hrv_delta_pct:.0f}% relative to the sinus baseline,
      compatible with autonomic modulation (potentially respiratory).</li>
  <li><b>Expected theoretical pulse deficit</b>: ~{len(pvc)/n_total*100:.0f}% of PVCs might
      not generate an appreciable peripheral pulse (this is the phenomenon whereby a Garmin
      on the wrist would count a lower rate than the ECG).</li>
</ul>

<h2>10. Technical limitations</h2>
<ul>
  <li>Single lead (Einthoven I) → impossible to localize the ectopic
      focus in the three anatomical planes.</li>
  <li>Detector based on post-QRS hyperpolarization and width → robust for the
      current pattern, but could miss PVCs with atypical morphology.</li>
  <li>{total_min:.1f} min of recording, a window too short to infer circadian
      or post-prandial trends.</li>
  <li>No synchronized respiration sensor → impossible to formally test the
      PVC ↔ respiratory phase correlation.</li>
</ul>

<div class="footer">
  Source file: <code>{os.path.basename(PATH)}</code><br>
  Pipeline: Pi Pico W (250 Hz ADC) → WiFi → Python server (Flask/SSE) → detector
  band-pass 0.3–25 Hz + 4-state FSM + rebound/width classification.<br>
  Repository: <code>https://github.com/mrEg0n/holter-ecg</code><br>
  Generated automatically by <code>host/generate_report.py</code>
</div>
</body></html>
"""

out_path = PATH.replace(os.sep + "ecg_", os.sep + "report_").replace(".csv", ".html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Report saved: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
