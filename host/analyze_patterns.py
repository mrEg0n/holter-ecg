"""
Analyzes temporal patterns in a holter recording:
  - distribution of pattern types (isolated, couplet, bigeminy, etc.)
  - PVC frequency over time (variable?)
  - coupling interval (RR from the preceding beat to the PVC)
  - comparison of pre-PVC RR vs stable sinus rhythm
  - HRV (SDNN, RMSSD) over windows

Usage: python3 analyze_patterns.py logs/peaks_YYYYMMDD_HHMMSS.csv
"""
import csv
import math
import os
import statistics
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if PATH is None:
    print("usage: analyze_patterns.py <peaks_*.csv>")
    sys.exit(1)

peaks = []
with open(PATH) as f:
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

if not peaks:
    print("No beats in the CSV")
    sys.exit(1)

# consecutive RR
for i in range(len(peaks)):
    peaks[i]["rr_prev"] = (peaks[i]["t"] - peaks[i-1]["t"]) if i > 0 else None
    peaks[i]["rr_next"] = (peaks[i+1]["t"] - peaks[i]["t"]) if i < len(peaks)-1 else None

total_s = peaks[-1]["t"] - peaks[0]["t"]
norm = [p for p in peaks if p["cls"] == "normal"]
pvc  = [p for p in peaks if p["cls"] == "pvc"]
print(f"=== {os.path.basename(PATH)} ===")
print(f"Duration: {total_s/60:.1f} min")
print(f"Beats: {len(peaks)} ({len(norm)} normal, {len(pvc)} PVC)")
print(f"Sinus: {60*len(norm)/total_s:.0f} BPM  PVC rate: {60*len(pvc)/total_s:.1f}/min  burden: {100*len(pvc)/len(peaks):.1f}%")

# ----------------------- 1) pattern types -----------------------
# classify each PVC based on its context: what precedes it, what follows it
def context(i):
    prev_cls = peaks[i-1]["cls"] if i > 0 else None
    next_cls = peaks[i+1]["cls"] if i < len(peaks)-1 else None
    return prev_cls, next_cls

iso = couplet_lead = couplet_trail = triplet = bigem_run = 0
for i, p in enumerate(peaks):
    if p["cls"] != "pvc": continue
    prev_cls, next_cls = context(i)
    if prev_cls != "pvc" and next_cls != "pvc":
        iso += 1
    elif prev_cls != "pvc" and next_cls == "pvc":
        # could be couplet_lead or the start of a triplet
        # check 2 PVC ahead
        nn = peaks[i+2]["cls"] if i+2 < len(peaks) else None
        if nn == "pvc":
            triplet += 1
        else:
            couplet_lead += 1
    elif prev_cls == "pvc" and next_cls != "pvc":
        # if 2 before was normal → couplet trail
        pp = peaks[i-2]["cls"] if i >= 2 else None
        if pp == "normal":
            couplet_trail += 1
        # else: it was already counted in triplet
    elif prev_cls == "pvc" and next_cls == "pvc":
        # PVC in the middle of a run >= 3
        pass  # already counted in triplet or long run

# Bigeminy: look for runs of the N-PVC-N-PVC pattern with at least 3 repetitions
bigem_starts = []
i = 0
while i < len(peaks) - 5:
    # check whether an N-PVC-N-PVC block starts here
    pattern = [peaks[i+j]["cls"] for j in range(6)]
    if pattern == ["normal","pvc","normal","pvc","normal","pvc"]:
        # it is bigeminy (at least 3 alternating PVCs). Find how far it extends.
        end = i
        while end+1 < len(peaks) and peaks[end+1]["cls"] != peaks[end]["cls"]:
            end += 1
        bigem_starts.append((i, end))
        i = end + 1
    else:
        i += 1

# Trigeminy: N-N-PVC-N-N-PVC-N-N-PVC
trigem_starts = []
i = 0
while i < len(peaks) - 8:
    pattern = [peaks[i+j]["cls"] for j in range(9)]
    if pattern == ["normal","normal","pvc","normal","normal","pvc","normal","normal","pvc"]:
        end = i
        # extend if it continues
        while end+3 < len(peaks) and \
              peaks[end+1]["cls"]=="normal" and peaks[end+2]["cls"]=="normal" and peaks[end+3]["cls"]=="pvc":
            end += 3
        trigem_starts.append((i, end))
        i = end + 1
    else:
        i += 1

print(f"\n=== PATTERN TYPES ===")
print(f"Couplets (2 adjacent PVC):        {couplet_lead}")
print(f"Triplets (3+ adjacent PVC):       {triplet}")
print(f"Isolated PVC:                     {iso}")
print(f"BIGEMINY runs (>=3 alt PVC):      {len(bigem_starts)}  (mean length {statistics.mean([e-s+1 for s,e in bigem_starts])/2:.1f} PVC each)" if bigem_starts else "BIGEMINY runs (>=3 alt PVC):      0")
print(f"TRIGEMINY runs (>=3 cycles):      {len(trigem_starts)}")

# ----------------------- 2) coupling interval distribution -----------------------
coupling = [p["rr_prev"] for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None]
sinus_rr = [p["rr_prev"] for p in peaks
            if p["cls"] == "normal" and p["rr_prev"] is not None
            and peaks[peaks.index(p)-1]["cls"] == "normal"]  # only N-N

print(f"\n=== COUPLING INTERVAL (RR before the PVC) ===")
if coupling:
    print(f"  median: {1000*statistics.median(coupling):.0f}ms")
    print(f"  mean:   {1000*statistics.mean(coupling):.0f}ms")
    print(f"  std:    {1000*statistics.stdev(coupling):.0f}ms")
    print(f"  range:  {1000*min(coupling):.0f}-{1000*max(coupling):.0f}ms")
print(f"\n=== SINUS RR (N-N) for comparison ===")
if sinus_rr:
    print(f"  median: {1000*statistics.median(sinus_rr):.0f}ms (= {60/statistics.median(sinus_rr):.0f} BPM)")
    print(f"  std:    {1000*statistics.stdev(sinus_rr):.0f}ms")

# if coupling << sinus_RR → premature. By how much?
if coupling and sinus_rr:
    med_coupling = statistics.median(coupling)
    med_sinus = statistics.median(sinus_rr)
    print(f"\n  Coupling/Sinus ratio: {med_coupling/med_sinus:.2f}  ({100*(1-med_coupling/med_sinus):.0f}% prematurity)")

# ----------------------- 3) RR in the PRE-PVC phase vs stable sinus rhythm -----------------------
# for each PVC, take the N preceding normal beats and compute statistics
LOOKBACK = 5
pre_pvc_rrs_per_pvc = []
for i, p in enumerate(peaks):
    if p["cls"] != "pvc": continue
    # go back until finding LOOKBACK consecutive normal beats
    nn_rrs = []
    j = i - 1
    while j >= 1 and len(nn_rrs) < LOOKBACK:
        if peaks[j]["cls"] == "normal" and peaks[j-1]["cls"] == "normal":
            nn_rrs.append(peaks[j]["t"] - peaks[j-1]["t"])
        j -= 1
    if len(nn_rrs) >= LOOKBACK:
        pre_pvc_rrs_per_pvc.append({
            "pvc_t": p["t"],
            "rrs": nn_rrs[::-1],  # in chronological order
            "mean": statistics.mean(nn_rrs),
            "stdev": statistics.stdev(nn_rrs) if len(nn_rrs) > 1 else 0,
        })

print(f"\n=== RR IN THE {LOOKBACK} NORMAL BEATS PRE-PVC ===")
if pre_pvc_rrs_per_pvc:
    means = [x["mean"] for x in pre_pvc_rrs_per_pvc]
    stds  = [x["stdev"] for x in pre_pvc_rrs_per_pvc]
    print(f"  N PVC analyzed: {len(pre_pvc_rrs_per_pvc)}")
    print(f"  mean of pre-PVC RR: {1000*statistics.mean(means):.0f}ms (= {60/statistics.mean(means):.0f} BPM)")
    print(f"  stdev across windows:   {1000*statistics.stdev(means):.0f}ms")
    print(f"  mean INTRA-window RR stdev: {1000*statistics.mean(stds):.0f}ms  ← if higher than the sinus baseline = the rhythm was more irregular before the PVC")

# baseline sinus stdev (random windows)
if len(sinus_rr) > LOOKBACK:
    # sample windows of LOOKBACK consecutive beats
    sinus_window_stds = []
    for k in range(0, len(sinus_rr) - LOOKBACK + 1, LOOKBACK):
        window = sinus_rr[k:k+LOOKBACK]
        sinus_window_stds.append(statistics.stdev(window))
    if sinus_window_stds:
        print(f"  Sinus baseline stdev (same-size windows): {1000*statistics.mean(sinus_window_stds):.0f}ms")

# ----------------------- 4) PVC rate in time windows -----------------------
WINDOW = 60  # 60 sec
windows = []
t0 = peaks[0]["t"]
i_w = 0
while t0 + i_w * WINDOW < peaks[-1]["t"]:
    ws = t0 + i_w * WINDOW
    we = ws + WINDOW
    in_w = [p for p in peaks if ws <= p["t"] < we]
    nn = sum(1 for p in in_w if p["cls"] == "normal")
    np_ = sum(1 for p in in_w if p["cls"] == "pvc")
    windows.append({"t_start": ws, "norm": nn, "pvc": np_, "burden": 100*np_/max(1,nn+np_)})
    i_w += 1

print(f"\n=== VARIATIONS OVER TIME ({WINDOW}s windows) ===")
print(f"  N windows: {len(windows)}")
burdens = [w["burden"] for w in windows]
print(f"  PVC burden % per minute: min={min(burdens):.0f}  max={max(burdens):.0f}  median={statistics.median(burdens):.0f}  std={statistics.stdev(burdens):.0f}")
sinus_rates = [60*w["norm"]/WINDOW for w in windows]
print(f"  Sinus BPM per minute:    min={min(sinus_rates):.0f}  max={max(sinus_rates):.0f}  median={statistics.median(sinus_rates):.0f}")

# ----------------------- 5) PLOT -----------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.patch.set_facecolor("#1e1e1e")

# (a) tachogram: RR vs time, colored by class
ax = axes[0]
ax.set_facecolor("#0d0d0d")
for i, p in enumerate(peaks):
    if p["rr_prev"] is None: continue
    color = "#e74c3c" if p["cls"] == "pvc" else "#2ecc71"
    ax.scatter(p["t"], 1000*p["rr_prev"], c=color, s=8, alpha=0.7)
ax.set_ylabel("RR (ms)", color="#aaa")
ax.set_title("Tachogram: RR vs time (red = PVC)", color="#fff")
ax.tick_params(colors="#aaa"); ax.grid(True, alpha=0.2, color="#444")
for sp in ax.spines.values(): sp.set_color("#444")

# (b) coupling interval distribution
ax = axes[1]
ax.set_facecolor("#0d0d0d")
if coupling and sinus_rr:
    ax.hist([r*1000 for r in sinus_rr], bins=40, alpha=0.6, color="#2ecc71",
            label=f"Sinus N→N (n={len(sinus_rr)})", density=True)
    ax.hist([r*1000 for r in coupling], bins=40, alpha=0.8, color="#e74c3c",
            label=f"Pre-PVC coupling (n={len(coupling)})", density=True)
ax.set_xlabel("RR (ms)", color="#aaa")
ax.set_ylabel("density", color="#aaa")
ax.set_title("RR distribution — sinus vs pre-PVC coupling", color="#fff")
ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
ax.tick_params(colors="#aaa"); ax.grid(True, alpha=0.2, color="#444")
for sp in ax.spines.values(): sp.set_color("#444")

# (c) PVC rate / burden over time
ax = axes[2]
ax.set_facecolor("#0d0d0d")
ts = [w["t_start"]/60 for w in windows]
ax.plot(ts, [w["pvc"] for w in windows], color="#e74c3c", marker="o", label="PVC/min")
ax.plot(ts, [w["norm"] for w in windows], color="#2ecc71", marker="o", label="Sinus/min", alpha=0.5)
ax.set_xlabel("Time (min)", color="#aaa")
ax.set_ylabel("Beats per minute", color="#aaa")
ax.set_title("Per-minute count over time", color="#fff")
ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
ax.tick_params(colors="#aaa"); ax.grid(True, alpha=0.2, color="#444")
for sp in ax.spines.values(): sp.set_color("#444")

# (d) pre-PVC HRV: variability of the RR in the 5 preceding beats
ax = axes[3]
ax.set_facecolor("#0d0d0d")
if pre_pvc_rrs_per_pvc:
    times = [x["pvc_t"]/60 for x in pre_pvc_rrs_per_pvc]
    stdevs = [1000*x["stdev"] for x in pre_pvc_rrs_per_pvc]
    ax.scatter(times, stdevs, c="#e74c3c", s=10, alpha=0.7, label="StdDev RR in the 5 pre-PVC N beats")
    if len(sinus_window_stds) > 0:
        baseline = 1000*statistics.mean(sinus_window_stds)
        ax.axhline(baseline, color="#2ecc71", linestyle="--", alpha=0.7,
                   label=f"Sinus baseline ({baseline:.0f}ms)")
ax.set_xlabel("Time (min)", color="#aaa")
ax.set_ylabel("Stdev RR (ms)", color="#aaa")
ax.set_title("RR variability in the 5 normal beats before each PVC", color="#fff")
ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
ax.tick_params(colors="#aaa"); ax.grid(True, alpha=0.2, color="#444")
for sp in ax.spines.values(): sp.set_color("#444")

plt.tight_layout()
out = PATH.replace(".csv", "_patterns.png")
plt.savefig(out, dpi=110, facecolor="#1e1e1e")
print(f"\nplot saved: {out}")
