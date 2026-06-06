"""
Plot a recorded session from the server logs.

Reads ecg_YYYYMMDD_HHMMSS.csv (samples) and the matching peaks_*.csv
(beat classifications) and produces a multi-row stacked plot, clinical
holter style: one row per ROW_S seconds, vertical stack.

Usage:
    python3 plot_session.py logs/ecg_YYYYMMDD_HHMMSS.csv [row_seconds]
"""
import csv
import math
import os
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if PATH is None:
    print("usage: plot_session.py <ecg_*.csv> [row_seconds]")
    sys.exit(1)

ROW_S = int(sys.argv[2]) if len(sys.argv) > 2 else 60   # 1 minuto per riga di default
SAMPLE_HZ = 250
ROW_SAMPLES = ROW_S * SAMPLE_HZ

# --- load ecg ---
t_s, v_raw, v_filt = [], [], []
with open(PATH) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        try:
            t_s.append(float(row["t_s"]))
            v_raw.append(float(row["raw"]))
            v_filt.append(float(row["filt"]))
        except (KeyError, ValueError):
            continue

if not t_s:
    print("CSV vuoto o malformato")
    sys.exit(1)

t  = np.array(t_s)
vr = np.array(v_raw)
vf = np.array(v_filt)
N  = len(vf)
print(f"loaded {N} samples, duration {t[-1]-t[0]:.1f}s, rate {N/(t[-1]-t[0]):.2f} Hz")

# --- load peaks (matching file) ---
peaks_path = PATH.replace(os.sep + "ecg_", os.sep + "peaks_")
peaks = []
if os.path.exists(peaks_path):
    with open(peaks_path) as f:
        for row in csv.DictReader(f):
            try:
                peaks.append({
                    "t":    float(row["t_s"]),
                    "amp":  float(row["amp_V"]),
                    "w":    float(row["width_ms"]),
                    "reb":  float(row["rebound_ratio"]),
                    "cls":  row["class"],
                })
            except (KeyError, ValueError):
                continue
    print(f"loaded {len(peaks)} peaks from {os.path.basename(peaks_path)}")
else:
    print("no peaks file found, skipping markers")

# --- stats ---
total_s = t[-1] - t[0]
norm = [p for p in peaks if p["cls"] == "normal"]
pvc  = [p for p in peaks if p["cls"] == "pvc"]
print(f"\n=== SESSION STATS ===")
print(f"durata totale:  {total_s/60:.1f} min ({total_s:.0f} s)")
print(f"battiti totali: {len(peaks)}")
print(f"  normali:      {len(norm)}  ({60*len(norm)/total_s:.0f} BPM sinus)")
print(f"  PVC:          {len(pvc)}  ({60*len(pvc)/total_s:.1f}/min, burden {100*len(pvc)/max(1,len(peaks)):.1f}%)")

# --- plot multi-row ---
n_rows = math.ceil(total_s / ROW_S)
fig, axes = plt.subplots(n_rows, 1, figsize=(16, max(3, 1.4 * n_rows)), sharex=False)
if n_rows == 1:
    axes = [axes]
fig.patch.set_facecolor("#1e1e1e")

for i, ax in enumerate(axes):
    t0 = i * ROW_S
    t1 = t0 + ROW_S
    mask = (t >= t0) & (t < t1)
    tt = t[mask] - t0
    vv = vf[mask]

    ax.set_facecolor("#0d0d0d")
    ax.plot(tt, vv, linewidth=0.6, color="#2ecc71")

    # red overlay around each PVC (±0.12s)
    row_peaks = [p for p in peaks if t0 <= p["t"] < t1]
    for p in row_peaks:
        pt = p["t"] - t0
        if p["cls"] == "pvc":
            # overlay finestra ±120ms
            window_mask = (tt >= pt - 0.12) & (tt <= pt + 0.12)
            if window_mask.any():
                ax.plot(tt[window_mask], vv[window_mask],
                        linewidth=1.2, color="#e74c3c")
            ax.scatter(pt, min(1.45, p["amp"] + 0.3), s=80, marker="v",
                       color="#e74c3c", edgecolors="white", linewidths=0.5, zorder=5)
        else:
            ax.scatter(pt, min(1.35, p["amp"] + 0.2), s=18, marker="v",
                       color="#2ecc71", edgecolors="white", linewidths=0.3, zorder=4)

    ax.set_xlim(0, ROW_S)
    ax.set_ylim(-1.0, 1.6)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, alpha=0.2, color="#444444")
    # label sull'asse Y: che minuto è
    mm = int(t0 // 60)
    ss = int(t0 % 60)
    ax.set_ylabel(f"{mm:02d}:{ss:02d}", color="#aaaaaa", fontsize=8, rotation=0,
                  ha="right", va="center")
    for sp in ax.spines.values():
        sp.set_color("#444444")
    # show counts per row
    rn = sum(1 for p in row_peaks if p["cls"] == "normal")
    rp = sum(1 for p in row_peaks if p["cls"] == "pvc")
    if rn or rp:
        ax.text(0.995, 0.95, f"{rn}N + {rp}PVC", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#888888")

axes[0].set_title(
    f"{os.path.basename(PATH)} — {total_s/60:.1f} min — "
    f"{len(norm)} sinus ({60*len(norm)/total_s:.0f} BPM) + "
    f"{len(pvc)} PVC ({100*len(pvc)/max(1,len(peaks)):.1f}% burden)",
    color="#ffffff", fontsize=12,
)
axes[-1].set_xlabel(f"Time within row (s, each row = {ROW_S}s)", color="#aaaaaa")

plt.tight_layout()
out = PATH.replace(".csv", "_session.png")
plt.savefig(out, dpi=110, facecolor="#1e1e1e")
print(f"\nsaved {out}")
