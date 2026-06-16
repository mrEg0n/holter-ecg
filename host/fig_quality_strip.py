"""
Figura "showcase" a piena pagina per la sezione Recording quality & PVC auto-detection:
una registrazione CONTINUA, multi-riga (stile Holter), con segnale pulito, detection
automatica affidabile (PVC in rosso, sinusali in verde) e UN tratto rumoroso ombreggiato
come "escluso" (marcato a mano, niente marker dentro).

Sceglie automaticamente la finestra migliore: ~108 s attorno a una singola esclusione
di durata media, con segnale pulito attorno, ampiezza R buona e qualche PVC.

Output: reports/figs_manual/quality_strip.png   (cartella NON toccata da export_latex.py)
Eseguire dalla root del repo:  python3 host/fig_quality_strip.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# font da paper: sans pulito (Helvetica/Arial), convenzione delle figure pubblicate
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import dashboard as D

TRACE   = "#2f8a63"   # traccia pulita (verde, come le strip del report)
PVC     = "#cc3b30"   # QRS PVC + marker
SINUS   = "#2f8a63"   # marker sinusali
EXCL_TR = "#9aa0a6"   # traccia dentro l'esclusione (grigia)
EXCL_BG = "#fbe2b0"   # sfondo esclusione (giallo-arancio: distinto dal rosso PVC)
EXCL_EDGE = "#e3b25e"
# rallentamento sinusale transitorio dopo la 2a PVC (tempi assoluti; validi per la
# finestra auto-selezionata di 06-08 22:01 = 3679-3751 s). Se la finestra cambia non si disegna.
SLOW_MARK = (3734.8, 3741.0, "transient sinus slowing")

ROW_DUR = 12.0        # secondi per riga
N_ROWS  = 6           # righe -> 72 s; righe alte -> riempie comunque la pagina
OUT_DIR = "reports/figs_manual"


def median_abs_diff(v):
    """Proxy di rumore: |diff| mediano. Basso = ECG pulito (poco tra i battiti),
    alto = rumore diffuso. Robusto al QRS (sparso)."""
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
    """Ritorna (label, t, v, peaks, excl, w0, w1, (es,ee)) per la finestra migliore."""
    cand = []
    T = ROW_DUR * N_ROWS
    for label, t, v, peaks, excl in sessions:
        t0, t1 = float(t[0]), float(t[-1])
        for (es, ee) in excl:
            dur = ee - es
            if not (3.0 <= dur <= 11.0):
                continue
            # finestra con l'esclusione ~centrata
            w0 = es - (T - dur) / 2.0
            w1 = w0 + T
            if w0 < t0:
                w0, w1 = t0, t0 + T
            if w1 > t1:
                w1, w0 = t1, t1 - T
            if w0 < t0 or w1 > t1:
                continue
            # niente ALTRE esclusioni nella finestra (vogliamo un solo tratto rumoroso)
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
            # cleanliness UNIFORME: rumore su sotto-finestre da 1.5 s fuori esclusione
            sub, x = [], w0
            while x + 1.5 <= w1:
                if not (x < ee and x + 1.5 > es):           # sotto-finestra tutta pulita
                    ms = (t >= x) & (t < x + 1.5)
                    if ms.sum() > 30:
                        sub.append(median_abs_diff(v[ms]))
                x += 1.5
            if len(sub) < 20:
                continue
            sub = np.array(sub)
            clean_med = float(np.median(sub))                 # rumore tipico fuori esclusione
            clean_p95 = float(np.percentile(sub, 95))         # il tratto pulito PEGGIORE
            uniform   = clean_p95 / max(clean_med, 1e-6)      # ~1.5-2 se pulito uniforme
            noise_in  = median_abs_diff(v[mex])
            contrast  = noise_in / max(clean_med, 1e-6)        # escluso vs pulito tipico
            if contrast < 5.0 or uniform > 2.6:               # escluso nettamente il peggio + rest uniforme
                continue
            amp = float(np.median([p["amp"] for p in peaks
                                   if w0 <= p["t"] <= w1 and p["cls"] == "normal"]))
            score = min(contrast, 15.0) * (amp + 0.3) / uniform
            cand.append((score, contrast, uniform, amp, n_pvc, n_sin, dur,
                         label, t, v, peaks, excl, w0, w1, (es, ee)))
    if not cand:
        return None
    cand.sort(key=lambda c: -c[0])
    print("Top finestre candidate:")
    for c in cand[:8]:
        print(f"  {c[7]}  excl_dur={c[6]:.1f}s  contrast={c[1]:.1f}x  uniform={c[2]:.2f}"
              f"  ampR={c[3]:.2f}V  PVC={c[4]} sin={c[5]}")
    return cand[0][7:]


def render(label, t, v, peaks, excl, w0, w1, ex):
    os.makedirs(OUT_DIR, exist_ok=True)
    es, ee = ex
    # y-lim robusti dalla parte pulita + spazio per i marker
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
        # esclusione che cade in questa riga
        mex_row = (t[m] >= es) & (t[m] <= ee)
        # traccia pulita (verde) e dentro-esclusione (grigia)
        yc = np.where(mex_row, np.nan, y)
        yg = np.where(mex_row, y, np.nan)
        ax.plot(x, yc, lw=0.45, color=TRACE)
        if np.isfinite(yg).any():
            ax.plot(x, yg, lw=0.4, color=EXCL_TR)
            # sfondo + etichetta sul tratto escluso
            a = max(es, rs) - rs
            b = min(ee, re) - rs
            ax.axvspan(a, b, color=EXCL_BG, zorder=0)
        # marker detection (i picchi sono gia' fuori dalle esclusioni)
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
        # parentesi discreta sul rallentamento sinusale (se cade in questa riga)
        s0, s1, slab = SLOW_MARK
        if w0 <= s0 and s1 <= w1 and s0 < re and s1 > rs:
            a = max(s0, rs) - rs
            b = min(s1, re) - rs
            rng = yhi - ylo
            ybr = ylo + 0.17 * rng          # staccata sotto la traccia
            tick = 0.05 * rng
            ax.plot([a, b], [ybr, ybr], color="#9a9a9a", lw=0.8, zorder=7)
            if s0 >= rs:                      # tick + etichetta solo dove INIZIA
                ax.plot([a, a], [ybr, ybr + tick], color="#9a9a9a", lw=0.8, zorder=7)
                ax.text((a + b) / 2, ybr - 0.03 * rng, slab, ha="center", va="top",
                        color="#777777", fontsize=7, fontstyle="italic")
            if s1 <= re:                      # tick destro solo dove FINISCE
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
        # etichetta tempo assoluto a sinistra
        mm, ss = int(rs // 60), int(rs % 60)
        ax.text(-0.012, 0.5, f"{mm:02d}:{ss:02d}", transform=ax.transAxes,
                ha="right", va="center", color="#888888", fontsize=9)
    axes[-1].set_xlabel("time (s)", color="#666666", fontsize=10)
    # legenda raggruppata: verde (linea/▽) | rosso (linea/▽) | esclusione
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
    print(f"\n✓ Scritto {out}")
    print(f"  Sessione {label}, finestra {w0:.0f}-{w1:.0f}s, esclusione {es:.0f}-{ee:.0f}s")


def main():
    print("Carico sessioni...")
    sessions = load_all()
    sel = best_window(sessions)
    if sel is None:
        print("Nessuna finestra adatta trovata."); return
    render(*sel)


if __name__ == "__main__":
    main()
