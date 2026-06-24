"""
"example strips" figure for the PVC characterization section, IN THE SAME STYLE as the
quality strip (host/fig_quality_strip.py): Helvetica font, thin trace, 450 dpi,
grouped legend, no internal title. CURATED and clean content, one-two per type:
isolated PVC, couplet, triplet (run of 3), interpolated, short run. Discards noisy /
mis-detected windows by picking those with the lowest window_noise_score.

Output: reports/figs_manual/example_strips.png   (folder not touched by export_latex.py)
Run from the repo root:  python3 host/fig_example_strips.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# same paper font as the quality strip
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import dashboard as D

TRACE = "#2f8a63"     # clean trace (green)
PVC   = "#cc3b30"     # QRS PVC + marker
SINUS = "#2f8a63"     # sinus markers
HALF  = 7.0           # half window (s) -> 14 s strip, less compressed in 2 columns
YLO, YHI = -1.2, 1.9


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
        if peaks and len(t):
            out.append((label, t, v, peaks, excl))
    return out


def find_triplets(t, v, peaks, excl, lo=200.0, hi=900.0, half=HALF):
    """Run of EXACTLY 3 consecutive PVCs (adjacent RR lo<RR<hi), not part of
    longer runs, outside the excluded segments, after the warm-up. Centered on the middle PVC."""
    far = lambda ts, m=2.0: all(not (s - m <= ts <= e + m) for s, e in excl)
    out, n = [], len(peaks)
    for i in range(1, n - 2):
        a, b, c = peaks[i], peaks[i+1], peaks[i+2]
        if not (a["cls"] == b["cls"] == c["cls"] == "pvc") or a["t"] < 60:
            continue
        if i - 1 >= 0 and peaks[i-1]["cls"] == "pvc":      # no 4th before
            continue
        if i + 3 < n and peaks[i+3]["cls"] == "pvc":       # no 4th after
            continue
        rr1 = (b["t"] - a["t"]) * 1000
        rr2 = (c["t"] - b["t"]) * 1000
        if not (lo < rr1 < hi and lo < rr2 < hi):
            continue
        if not (far(a["t"]) and far(b["t"]) and far(c["t"])):
            continue
        snip = D._snip(t, v, peaks, b["t"], half)
        if snip is not None:
            out.append((D.window_noise_score(snip), snip))
    out.sort(key=lambda x: x[0])
    return out


def best_per_session(sessions, finder):
    """For each session applies `finder`, collects (noise, snip, label), sorts cleanest->first."""
    res = []
    for label, t, v, peaks, excl in sessions:
        s = finder(t, v, peaks, excl)
        if s is not None:
            res.append((D.window_noise_score(s), s, label))
    res.sort(key=lambda x: x[0])
    return res


def pick(lst, n):
    """Takes the n cleanest, preferring distinct sessions."""
    out, seen = [], set()
    for _, s, label in lst:
        if label in seen:
            continue
        out.append((s, label)); seen.add(label)
        if len(out) >= n:
            return out
    for _, s, label in lst:                 # fill if distinct sessions are not enough
        if any(s is o for o, _ in out):
            continue
        out.append((s, label))
        if len(out) >= n:
            break
    return out


def draw_panel(ax, strip, title):
    c = strip["center"]
    ax.set_facecolor("#ffffff")
    ax.plot(strip["t"] - c, strip["v"], lw=0.45, color=TRACE)
    for p in strip["peaks"]:
        xp = p["t"] - c
        if p["cls"] == "pvc":
            wm = (strip["t"] >= p["t"] - 0.12) & (strip["t"] <= p["t"] + 0.12)
            ax.plot(strip["t"][wm] - c, strip["v"][wm], lw=0.9, color=PVC, zorder=4)
            ax.scatter(xp, min(YHI - 0.18, p["amp"] + 0.30), s=28, marker="v",
                       color=PVC, edgecolors="#1a1a1a", linewidths=0.4, zorder=6)
        else:
            ax.scatter(xp, min(YHI - 0.30, p["amp"] + 0.16), s=9, marker="v",
                       color=SINUS, edgecolors="#1a1a1a", linewidths=0.25, zorder=5)
    ax.set_xlim(-strip["pre"], strip["post"])
    ax.set_ylim(YLO, YHI)
    ax.grid(True, alpha=0.14, color="#dcdcdc", lw=0.4)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#777777", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.set_title(title, color="#222222", fontsize=8.5, pad=3)


def main():
    print("Loading sessions...")
    sessions = load_all()
    if not sessions:
        print("No sessions."); return

    # --- ORIGINAL selection (like dashboard.py img_examples) ---
    # TRUE couplet (not part of a run >=3, so it does not coincide with the triplet), the cleanest
    best_couplet = None   # (snip, label, noise)
    for label, t, v, peaks, excl in sessions:
        for c in D.find_all_couplets(t, v, peaks, excl):
            snip = D._snip(t, v, peaks, (c["t1"] + c["t2"]) / 2.0, HALF)
            if snip is None:
                continue
            noise = c.get("noise", D.window_noise_score(snip))
            if best_couplet is None or noise < best_couplet[2]:
                best_couplet = (snip, label, noise)
    # burst with more PVCs
    bursts = []
    for label, t, v, peaks, excl in sessions:
        s = D.find_burst_strip(t, v, peaks, excl, half=HALF)
        if s:
            bursts.append((s, label))
    bursts.sort(key=lambda x: -x[0]["n"])
    best_burst = bursts[0] if bursts else None
    # isolated (regular) in session order
    regulars = []
    for label, t, v, peaks, excl in sessions:
        s = D.pick_example_strip(t, v, peaks, excl, HALF, HALF)
        if s:
            regulars.append((s, label))
    # --- the 2 strips in second position (they were mis-detected interpolated ones; the only
    #     triplet in the dataset is noisy) -> TWO CLEAN short-runs from different sessions ---
    burst_sess = best_burst[1] if best_burst else None
    runs = [(D.window_noise_score(s), s, label) for s, label in bursts
            if s.get("n", 99) <= 6 and label != burst_sess]
    runs.sort(key=lambda x: x[0])

    panels = []   # (strip, title)
    def push(s, label, typ):
        ctr = s["center"]; mm, ss = int(ctr // 60), int(ctr % 60)
        extra = f" (n={s['n']})" if "n" in s else ""
        panels.append((s, f"{D.short_label(label)} @{mm:02d}:{ss:02d}  ·  {typ}{extra}"))

    if best_couplet: push(best_couplet[0], best_couplet[1], "couplet")   # [0]
    if best_burst:   push(best_burst[0], best_burst[1], "burst")        # [1]
    for _, s, label in runs[:2]:                                        # [2],[3] clean short-runs
        push(s, label, "short run")
    for s, label in regulars:                                           # [4..] isolated
        if len(panels) >= 10:
            break
        push(s, label, "isolated PVC")

    ncol, nrow = 2, (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.1, 1.18 * nrow + 0.85),
                             facecolor="#ffffff", squeeze=False)
    flat = axes.ravel()
    for ax, (s, title) in zip(flat, panels):
        draw_panel(ax, s, title)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for ax in flat[max(0, len(panels) - ncol):len(panels)]:
        ax.set_xlabel("time (s)", color="#666666", fontsize=9)

    leg = [Line2D([0], [0], color=TRACE, lw=1.8, label="clean ECG"),
           Line2D([0], [0], marker="v", color=SINUS, lw=0, markersize=8,
                  label="sinus beat (auto)"),
           Line2D([0], [0], color=PVC, lw=1.8, label="PVC"),
           Line2D([0], [0], marker="v", color=PVC, lw=0, markersize=10,
                  label="PVC (auto)")]
    fig.legend(handles=leg, loc="upper center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.995), columnspacing=2.4)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.07,
                        hspace=0.55, wspace=0.08)
    os.makedirs("reports/figs_manual", exist_ok=True)
    out = "reports/figs_manual/example_strips.png"
    fig.savefig(out, dpi=450, facecolor="#ffffff")
    plt.close(fig)
    print(f"\n✓ Written {out}  ({len(panels)} panels)")
    for s, title in panels:
        print(f"  - {title}")


if __name__ == "__main__":
    main()
