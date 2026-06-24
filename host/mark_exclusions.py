"""
Interactive editor to manually mark the noisy stretches of an ECG session.

Opens a paginated strip-chart (10 minutes per page, 1 min/row). The user
selects with the mouse (click-and-drag) the intervals to exclude; each interval
appears as a red overlay. The keys allow navigating pages, saving,
undoing the last one, and quitting.

Output: JSON in exclusions/exclusions_<base>.json, readable by the report
generators via EXCLUDE_INTERVALS or direct loading.

Keys:
  drag mouse     mark interval to exclude
  n / →          next page
  p / ←          previous page
  u              undo (removes the last interval added)
  s              save
  q              save and quit
  d              delete ALL intervals (asks for confirmation in the terminal)

Usage:
    python3 host/mark_exclusions.py logs/ecg_YYYYMMDD_HHMMSS.csv
"""
import csv
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

if len(sys.argv) < 2:
    print("usage: mark_exclusions.py <ecg_*.csv>")
    sys.exit(1)

ECG = sys.argv[1]
ROWS_PER_PAGE = 10
ROW_S = 60   # 1 min per row
SR = 250

# ---- load ECG (filt) ----
ts, vf = [], []
with open(ECG) as f:
    for r in csv.DictReader(f):
        try:
            ts.append(float(r["t_s"]))
            vf.append(float(r["filt"]))
        except (KeyError, ValueError):
            continue
t = np.array(ts)
vf = np.array(vf)
total_s = float(t[-1] - t[0])
total_rows = int(np.ceil(total_s / ROW_S))
total_pages = int(np.ceil(total_rows / ROWS_PER_PAGE))
print(f"Session {os.path.basename(ECG)}: {total_s/60:.1f} min, "
      f"{total_rows} rows, {total_pages} pages")

# ---- load peaks (for overlay markers) ----
PK = ECG.replace("ecg_", "peaks_")
peaks = []
if os.path.exists(PK):
    with open(PK) as f:
        for r in csv.DictReader(f):
            try:
                peaks.append({
                    "t":   float(r["t_s"]),
                    "amp": float(r["amp_V"]),
                    "w":   float(r["width_ms"]),
                    "reb": float(r["rebound_ratio"]),
                    "cls": r["class"],
                })
            except (KeyError, ValueError):
                continue
# reclassify with the current criterion (amp >= 0.70)
for p in peaks:
    shape = (p["reb"] >= 0.40 or p["w"] >= 95.0)
    p["cls"] = "pvc" if (shape and p["amp"] >= 0.70) else "normal"

# ---- load existing exclusions ----
os.makedirs("exclusions", exist_ok=True)
base = os.path.basename(ECG).replace("ecg_", "").replace(".csv", "")
EXCL_FILE = f"exclusions/exclusions_{base}.json"
exclusions = []   # list of (start_s, end_s)
if os.path.exists(EXCL_FILE):
    with open(EXCL_FILE) as f:
        data = json.load(f)
    exclusions = [(d["start"], d["end"]) for d in data.get("intervals", [])]
    print(f"Loaded {len(exclusions)} existing exclusions from {EXCL_FILE}")

# ---- global UI state ----
state = {
    "page": 0,
    "fig": None,
    "axes": [],
    "selectors": [],
    "dirty": False,
}

def fmt_ts(s):
    return f"{int(s//60):02d}:{s%60:05.2f}"

def save_excl():
    """Writes the JSON with the intervals (sorted, no overlap merge for now)."""
    intervals_sorted = sorted(exclusions, key=lambda x: x[0])
    with open(EXCL_FILE, "w") as f:
        json.dump({
            "ecg_file": os.path.basename(ECG),
            "n_intervals": len(intervals_sorted),
            "total_excluded_s": sum(e - s for s, e in intervals_sorted),
            "intervals": [{"start": round(s, 3), "end": round(e, 3)} for s, e in intervals_sorted],
        }, f, indent=2)
    state["dirty"] = False
    print(f"💾 Saved {len(intervals_sorted)} exclusions "
          f"({sum(e-s for s,e in intervals_sorted):.1f}s) → {EXCL_FILE}")

def draw_page():
    """Redraws the current page. Called after every modification."""
    fig = state["fig"]
    if fig is None:
        return
    state["selectors"] = []  # recreate all span selectors
    for k, ax in enumerate(state["axes"]):
        ax.clear()
        row = state["page"] * ROWS_PER_PAGE + k
        ax.set_facecolor("black")
        if row >= total_rows:
            ax.axis("off")
            continue
        t0 = row * ROW_S
        t1 = t0 + ROW_S
        mask = (t >= t0) & (t < t1)
        if mask.any():
            ax.plot(t[mask] - t0, vf[mask], color="#33ff66", lw=0.5)
        # peaks
        n_norm = n_pvc = 0
        for p in peaks:
            if t0 <= p["t"] < t1:
                x = p["t"] - t0
                if p["cls"] == "pvc":
                    ax.plot(x, 1.3, "v", color="#ff4444", ms=5)
                    n_pvc += 1
                else:
                    ax.plot(x, 0.85, "v", color="#33aa66", ms=3)
                    n_norm += 1
        # exclusions that intersect this row
        for (s, e) in exclusions:
            if e < t0 or s > t1:
                continue
            x0 = max(s, t0) - t0
            x1 = min(e, t1) - t0
            ax.axvspan(x0, x1, color="red", alpha=0.40, zorder=10)
        ax.text(0.995, 0.93, f"{n_norm}N+{n_pvc}PVC",
                ha="right", va="top", transform=ax.transAxes,
                color="white", fontsize=7)
        ax.set_xlim(0, ROW_S)
        ax.set_ylim(-1.2, 1.6)
        ax.set_ylabel(f"{row:02d}:00", color="white", rotation=0,
                      ha="right", va="center", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.grid(alpha=0.15, color="#666")
        # span selector
        def make_on_select(row_t0=t0):
            def on_select(xmin, xmax):
                start = row_t0 + xmin
                end = row_t0 + xmax
                if end - start < 0.5:
                    return
                exclusions.append((start, end))
                state["dirty"] = True
                print(f"+ {fmt_ts(start)} → {fmt_ts(end)}  "
                      f"({end-start:.1f}s)  [tot: {len(exclusions)}]")
                draw_page()
            return on_select
        sel = SpanSelector(ax, make_on_select(t0), "horizontal",
                           useblit=True,
                           props=dict(alpha=0.4, facecolor="red"))
        state["selectors"].append(sel)
    # title
    n_excl = len(exclusions)
    tot_s = sum(e - s for s, e in exclusions)
    title = (f"Page {state['page']+1}/{total_pages}  ·  "
             f"exclusions: {n_excl} ({tot_s:.0f}s)  ·  "
             f"DIRTY={'yes' if state['dirty'] else 'no'}   "
             f"[ drag=mark · n/→=next · p/←=prev · u=undo · s=save · q=save+quit · d=clear ]")
    fig.suptitle(title, color="white", fontsize=9)
    fig.canvas.draw_idle()

def on_key(event):
    if event.key == "s":
        save_excl()
        draw_page()
    elif event.key == "u":
        if exclusions:
            removed = exclusions.pop()
            state["dirty"] = True
            print(f"- Undo: removed {fmt_ts(removed[0])} → {fmt_ts(removed[1])}")
            draw_page()
    elif event.key in ("n", "right", "pagedown"):
        if state["page"] < total_pages - 1:
            state["page"] += 1
            draw_page()
    elif event.key in ("p", "left", "pageup"):
        if state["page"] > 0:
            state["page"] -= 1
            draw_page()
    elif event.key == "q":
        if state["dirty"]:
            save_excl()
        plt.close("all")
    elif event.key == "d":
        # asks for confirmation in the terminal
        print(f"⚠️  Really delete all {len(exclusions)} exclusions? "
              f"Press 'd' again within 3s to confirm.")
        state["clear_pending"] = True
    elif event.key == "D" or (event.key == "d" and state.get("clear_pending")):
        exclusions.clear()
        state["dirty"] = True
        state["clear_pending"] = False
        print("🗑  All exclusions deleted.")
        draw_page()
    elif event.key == "g":
        # goto: input from terminal
        try:
            v = input("Go to minute (e.g. 45): ").strip()
            m = int(v)
            target_page = m // (ROWS_PER_PAGE)
            if 0 <= target_page < total_pages:
                state["page"] = target_page
                draw_page()
        except Exception as e:
            print(f"invalid input: {e}")

# ---- build figure once ----
fig, axes = plt.subplots(ROWS_PER_PAGE, 1, figsize=(16, ROWS_PER_PAGE*1.1),
                          facecolor="black", sharex=False)
state["fig"] = fig
state["axes"] = list(axes) if ROWS_PER_PAGE > 1 else [axes]
fig.canvas.mpl_connect("key_press_event", on_key)

# set close → save
def on_close(event):
    if state["dirty"]:
        save_excl()
fig.canvas.mpl_connect("close_event", on_close)

draw_page()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
