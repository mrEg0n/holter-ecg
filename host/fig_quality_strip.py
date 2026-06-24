"""
Full-page "showcase" figure for the Recording quality & PVC auto-detection section:
a CONTINUOUS, multi-row recording (Holter style), with a clean signal, reliable
automatic detection (PVCs in red, sinus beats in green) and ONE noisy segment shaded
as "excluded" (manually marked, no markers inside).

Automatically picks the best window: ~108 s around a single exclusion of medium
duration, with a clean signal around it, good R amplitude and a few PVCs.

Output: reports/figs_manual/quality_strip.png   (folder NOT touched by export_latex.py)
Run from the repo root:  python3 host/fig_quality_strip.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# paper font: clean sans (Helvetica/Arial), convention for published figures
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import dashboard as D

TRACE   = "#2f8a63"   # clean trace (green, like the report strips)
PVC     = "#cc3b30"   # QRS PVC + marker
SINUS   = "#2f8a63"   # sinus markers
EXCL_TR = "#9aa0a6"   # trace inside the exclusion (gray)
EXCL_BG = "#fbe2b0"   # exclusion background (yellow-orange: distinct from PVC red)
EXCL_EDGE = "#e3b25e"
# transient sinus slowing after the 2nd PVC (absolute times; valid for the
# auto-selected window 06-08 22:01 = 3679-3751 s). If the window changes it is not drawn.
SLOW_MARK = (3734.8, 3741.0, "transient sinus slowing")

ROW_DUR = 12.0        # seconds per row
N_ROWS  = 6           # rows -> 72 s; tall rows -> still fills the page
OUT_DIR = "reports/figs_manual"


def median_abs_diff(v):
    """Noise proxy: median |diff|. Low = clean ECG (little between beats),
    high = diffuse noise. Robust to QRS (sparse)."""
    if len(v) < 8:
        return 0.0
    return float(np.median(np.abs(np.diff(v))))


def load_all():
    files = sorted(glob.glob("logs/ecg_*.csv"))
    files = [f for f in files if os.path.getsize(f) > D.MIN_FILE_SIZE_MB * 1_000_000]
    files = [f for f in files if D.label_from_path(f)[1] not in D.SKIP_SESSIONS]
    out = []
    for f in files:
        label, _ = D.label_from_path(f)
        d = D.load_session(f)
        if d is None:
            continue
        t, v, peaks, excl = d
        if not peaks or len(t) == 0:
            continue
        out.append((label, t, v, peaks, excl))
    return out


def best_window(sessions):
    """Returns (label, t, v, peaks, excl, w0, w1, (es,ee)) for the best window."""
    cand = []
    T = ROW_DUR * N_ROWS
    for label, t, v, peaks, excl in sessions:
        t0, t1 = float(t[0]), float(t[-1])
        for (es, ee) in excl:
            dur = ee - es
            if not (3.0 <= dur <= 11.0):
                continue
            # window with the exclusion ~centered
            w0 = es - (T - dur) / 2.0
            w1 = w0 + T
            if w0 < t0:
                w0, w1 = t0, t0 + T
            if w1 > t1:
                w1, w0 = t1, t1 - T
            if w0 < t0 or w1 > t1:
                continue
            # no OTHER exclusions in the window (we want a single noisy segment)
            others = [(os_, oe) for (os_, oe) in excl if (os_, oe) != (es, ee)
                      and os_ < w1 and oe > w0]
            if others:
                continue
            mex  = (t >= es) & (t <= ee)
            if mex.sum() < 100:
                continue
            n_pvc = sum(1 for p in peaks if w0 <= p["t"] <= w1 and p["cls"] == "pvc")
            n_sin = sum(1 for p in peaks if w0 <= p["t"] <= w1 and p["cls"] == "normal")
            if n_pvc < 3 or n_sin < 40:
                continue
            # UNIFORM cleanliness: noise over 1.5 s sub-windows outside the exclusion
            sub, x = [], w0
            while x + 1.5 <= w1:
                if not (x < ee and x + 1.5 > es):           # fully clean sub-window
                    ms = (t >= x) & (t < x + 1.5)
                    if ms.sum() > 30:
                        sub.append(median_abs_diff(v[ms]))
                x += 1.5
            if len(sub) < 20:
                continue
            sub = np.array(sub)
            clean_med = float(np.median(sub))                 # typical noise outside the exclusion
            clean_p95 = float(np.percentile(sub, 95))         # the WORST clean segment
            uniform   = clean_p95 / max(clean_med, 1e-6)      # ~1.5-2 if uniformly clean
            noise_in  = median_abs_diff(v[mex])
            contrast  = noise_in / max(clean_med, 1e-6)        # excluded vs typical clean
            if contrast < 5.0 or uniform > 2.6:               # excluded clearly the worst + rest uniform
                continue
            amp = float(np.median([p["amp"] for p in peaks
                                   if w0 <= p["t"] <= w1 and p["cls"] == "normal"]))
            score = min(contrast, 15.0) * (amp + 0.3) / uniform
            cand.append((score, contrast, uniform, amp, n_pvc, n_sin, dur,
                         label, t, v, peaks, excl, w0, w1, (es, ee)))
    if not cand:
        return None
    cand.sort(key=lambda c: -c[0])
    print("Top candidate windows:")
    for c in cand[:8]:
        print(f"  {c[7]}  excl_dur={c[6]:.1f}s  contrast={c[1]:.1f}x  uniform={c[2]:.2f}"
              f"  ampR={c[3]:.2f}V  PVC={c[4]} sin={c[5]}")
    return cand[0][7:]


def render(label, t, v, peaks, excl, w0, w1, ex):
    os.makedirs(OUT_DIR, exist_ok=True)
    es, ee = ex
    # robust y-lim from the clean part + room for the markers
    mwin = (t >= w0) & (t <= w1)
    vv = v[mwin]
    ylo = min(-1.0, float(np.percentile(vv, 0.3)) - 0.25)
    yhi = max(1.6, float(np.percentile(vv, 99.7)) + 0.55)

    fig, axes = plt.subplots(N_ROWS, 1, figsize=(8.1, 0.8 * N_ROWS + 1.0),
                             facecolor="#ffffff")
    for r, ax in enumerate(axes):
        rs = w0 + r * ROW_DUR
        re = rs + ROW_DUR
        ax.set_facecolor("#ffffff")
        m = (t >= rs) & (t <= re)
        x = t[m] - rs
        y = v[m]
        # exclusion that falls in this row
        mex_row = (t[m] >= es) & (t[m] <= ee)
        # clean trace (green) and inside-exclusion (gray)
        yc = np.where(mex_row, np.nan, y)
        yg = np.where(mex_row, y, np.nan)
        ax.plot(x, yc, lw=0.45, color=TRACE)
        if np.isfinite(yg).any():
            ax.plot(x, yg, lw=0.4, color=EXCL_TR)
            # background + label on the excluded segment
            a = max(es, rs) - rs
            b = min(ee, re) - rs
            ax.axvspan(a, b, color=EXCL_BG, zorder=0)
        # detection markers (peaks are already outside the exclusions)
        for p in peaks:
            if not (rs <= p["t"] <= re):
                continue
            xp = p["t"] - rs
            if p["cls"] == "pvc":
                wm = (t >= p["t"] - 0.12) & (t <= p["t"] + 0.12)
                ax.plot(t[wm] - rs, v[wm], lw=0.9, color=PVC, zorder=4)
                ax.scatter(xp, min(yhi - 0.18, p["amp"] + 0.30), s=46, marker="v",
                           color=PVC, edgecolors="#1a1a1a", linewidths=0.45, zorder=6)
            else:
                ax.scatter(xp, min(yhi - 0.3, p["amp"] + 0.16), s=15, marker="v",
                           color=SINUS, edgecolors="#1a1a1a", linewidths=0.3, zorder=5)
        # discreet bracket on the sinus slowing (if it falls in this row)
        s0, s1, slab = SLOW_MARK
        if w0 <= s0 and s1 <= w1 and s0 < re and s1 > rs:
            a = max(s0, rs) - rs
            b = min(s1, re) - rs
            rng = yhi - ylo
            ybr = ylo + 0.17 * rng          # offset below the trace
            tick = 0.05 * rng
            ax.plot([a, b], [ybr, ybr], color="#9a9a9a", lw=0.8, zorder=7)
            if s0 >= rs:                      # tick + label only where it STARTS
                ax.plot([a, a], [ybr, ybr + tick], color="#9a9a9a", lw=0.8, zorder=7)
                ax.text((a + b) / 2, ybr - 0.03 * rng, slab, ha="center", va="top",
                        color="#777777", fontsize=7, fontstyle="italic")
            if s1 <= re:                      # right tick only where it ENDS
                ax.plot([b, b], [ybr, ybr + tick], color="#9a9a9a", lw=0.8, zorder=7)
        ax.set_xlim(0, ROW_DUR)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, which="both", alpha=0.14, color="#dcdcdc", lw=0.4)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors="#777777", labelsize=8.5)
        if r < N_ROWS - 1:
            ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        # absolute time label on the left
        mm, ss = int(rs // 60), int(rs % 60)
        ax.text(-0.012, 0.5, f"{mm:02d}:{ss:02d}", transform=ax.transAxes,
                ha="right", va="center", color="#888888", fontsize=9)
    axes[-1].set_xlabel("time (s)", color="#666666", fontsize=10)
    # grouped legend: green (line/▽) | red (line/▽) | exclusion
    leg = [Line2D([0], [0], color=TRACE, lw=1.8, label="clean ECG"),
           Line2D([0], [0], marker="v", color=SINUS, lw=0, markersize=8,
                  label="sinus beat (auto)"),
           Line2D([0], [0], color=PVC, lw=1.8, label="PVC"),
           Line2D([0], [0], marker="v", color=PVC, lw=0, markersize=10,
                  label="PVC (auto)"),
           Patch(facecolor=EXCL_BG, edgecolor=EXCL_EDGE,
                 label="manually excluded (noise)")]
    fig.legend(handles=leg, loc="upper center", ncol=3, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.995), columnspacing=2.4)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.90, bottom=0.075, hspace=0.30)
    out = os.path.join(OUT_DIR, "quality_strip.png")
    fig.savefig(out, dpi=450, facecolor="#ffffff")
    plt.close(fig)
    print(f"\n✓ Written {out}")
    print(f"  Session {label}, window {w0:.0f}-{w1:.0f}s, exclusion {es:.0f}-{ee:.0f}s")


def main():
    print("Loading sessions...")
    sessions = load_all()
    sel = best_window(sessions)
    if sel is None:
        print("No suitable window found."); return
    render(*sel)


if __name__ == "__main__":
    main()
