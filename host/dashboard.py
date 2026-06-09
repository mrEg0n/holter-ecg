"""
Master Dashboard for self-monitored Holter ECG.

Scans all session files under logs/, applies the same classification pipeline,
loads manual noise exclusions from exclusions/, and produces a single
self-contained HTML report that aggregates all analyses across sessions and
grows with each new recording.

Sections currently implemented:
  - Device & acquisition setup
  - Dataset summary (sessions table + cumulative counts)
  - Morphology analysis (overlay, per-session medians, cross-session
    correlation matrix, hyperpolarization-depth distribution)

Designed to be extensible: add new sections (rhythm dynamics, respiratory
phase analysis, posture correlation, etc.) by appending to the `report_html`
sections.

Usage:
    python3 host/morphology_dashboard.py
        → writes reports/holter_dashboard.html
"""
import csv, json, os, glob, base64, io
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- classifier configuration ----------
PVC_MIN_AMP = 0.70   # V
REBOUND_PVC = 0.40
PVC_W_MS    = 95.0
PVC_W_MIN   = 40.0
PVC_W_MAX   = 220.0
PVC_MIN_REBOUND = 0.05

SR = 250    # sample rate, Hz
WIN = 0.6   # window for morphology overlay (±0.3 s)
N_SAMPLES = int(WIN * SR)
TG = np.linspace(-WIN/2, WIN/2, N_SAMPLES)

# Minimum thresholds for including a session
MIN_FILE_SIZE_MB = 5
MIN_PVC_COUNT    = 50

# Sessions to exclude from the dataset (e.g. inverted electrodes, test runs)
SKIP_SESSIONS = {
    "20260605_131136",  # electrodes inverted (polarity reversed)
}

DARK_BG = "#0d0f12"
GREEN, RED, BLUE, ORANGE = "#33ff66", "#ff4d6d", "#7ad9ff", "#ffa64d"

# ---------- layout helpers ----------
# Posizioni esplicite per garantire allineamento perfetto tra figure 2x2
# (l, b, w, h) in figure-relative coords (0-1). Stesso layout per le 3 figure.
PANEL_W = 0.32
PANEL_H = 0.36
LEFT_COL  = 0.07
RIGHT_COL = 0.48   # gap orizzontale ampio (0.48 - 0.39 = 0.09)
BOTTOM_ROW = 0.08
TOP_ROW    = 0.58  # gap verticale ampio (0.58 - 0.44 = 0.14)
PANEL_POS = {
    "tl": (LEFT_COL,  TOP_ROW,    PANEL_W, PANEL_H),
    "tr": (RIGHT_COL, TOP_ROW,    PANEL_W, PANEL_H),
    "bl": (LEFT_COL,  BOTTOM_ROW, PANEL_W, PANEL_H),
    "br": (RIGHT_COL, BOTTOM_ROW, PANEL_W, PANEL_H),
}

# ---------- I/O helpers ----------
def label_from_path(ecg_path):
    base = os.path.basename(ecg_path).replace("ecg_","").replace(".csv","")
    if len(base) >= 13:
        d = f"{base[0:4]}-{base[4:6]}-{base[6:8]} {base[9:11]}:{base[11:13]}"
    else:
        d = base
    return d, base

def short_label(label):
    # da "2026-06-05 14:59" → "06-05 14:59"
    return label[5:] if len(label) >= 16 else label

def load_session(ecg_path):
    pk_path = ecg_path.replace("ecg_","peaks_")
    if not os.path.exists(pk_path): return None
    _, base = label_from_path(ecg_path)
    excl_path = f"exclusions/exclusions_{base}.json"
    EXCL = []
    if os.path.exists(excl_path):
        try:
            with open(excl_path) as f:
                EXCL = [(d["start"], d["end"]) for d in json.load(f).get("intervals", [])]
        except Exception:
            pass

    peaks = []
    with open(pk_path) as f:
        for r in csv.DictReader(f):
            try:
                p = {"t":float(r["t_s"]), "amp":float(r["amp_V"]),
                     "w":float(r["width_ms"]), "reb":float(r["rebound_ratio"]),
                     "cls":r["class"]}
                shape  = (p["reb"] >= REBOUND_PVC or p["w"] >= PVC_W_MS)
                plaus  = PVC_W_MIN <= p["w"] <= PVC_W_MAX
                reb_ok = p["reb"] >= PVC_MIN_REBOUND
                p["cls"] = "pvc" if (shape and p["amp"] >= PVC_MIN_AMP
                                      and plaus and reb_ok) else "normal"
                peaks.append(p)
            except (KeyError, ValueError):
                continue
    peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP)]
    peaks = [p for p in peaks if not any(s <= p["t"] <= e for s, e in EXCL)]

    ts, vf = [], []
    with open(ecg_path) as f:
        for r in csv.DictReader(f):
            try:
                ts.append(float(r["t_s"])); vf.append(float(r["filt"]))
            except (KeyError, ValueError):
                continue
    return np.array(ts), np.array(vf), peaks, EXCL

def collect_traces(t_ecg, vf_arr, peaks, kind="pvc"):
    out = []
    for p in peaks:
        if p["cls"] != kind: continue
        pt = p["t"]
        mask = (t_ecg >= pt-WIN/2) & (t_ecg <= pt+WIN/2)
        if mask.sum() < N_SAMPLES * 0.9: continue
        v = np.interp(TG, t_ecg[mask]-pt, vf_arr[mask])
        out.append(v)
    return np.array(out) if out else np.zeros((0, N_SAMPLES))

def fig_to_b64(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

# ---------- main ----------
def main():
    print("Scanning logs/...")
    ecg_files = sorted(glob.glob("logs/ecg_*.csv"))
    ecg_files = [f for f in ecg_files
                 if os.path.getsize(f) > MIN_FILE_SIZE_MB * 1_000_000]
    ecg_files = [f for f in ecg_files
                 if label_from_path(f)[1] not in SKIP_SESSIONS]
    print(f"  {len(ecg_files)} candidate sessions ≥ {MIN_FILE_SIZE_MB} MB")

    sessions = []
    for ecg_path in ecg_files:
        label, base = label_from_path(ecg_path)
        print(f"  loading {label}...")
        data = load_session(ecg_path)
        if data is None: continue
        t_ecg, vf_arr, peaks, excl = data
        if not peaks: continue
        traces_raw = collect_traces(t_ecg, vf_arr, peaks, kind="pvc")
        if traces_raw.shape[0] < MIN_PVC_COUNT:
            print(f"    skip: only {traces_raw.shape[0]} PVCs")
            continue
        # coupling intervals (RR_pre per ogni PVC, in ms)
        coupling_ms = []
        for i, p in enumerate(peaks):
            if p["cls"] != "pvc" or i == 0: continue
            rr = (p["t"] - peaks[i-1]["t"]) * 1000
            if 200 < rr < 800:   # range fisiologico coupling
                coupling_ms.append(rr)
        # raccolgo anche N (max 500/sessione per non saturare la memoria)
        traces_n_raw = collect_traces(t_ecg, vf_arr, peaks, kind="normal")
        if traces_n_raw.shape[0] > 500:
            idx = np.linspace(0, traces_n_raw.shape[0]-1, 500, dtype=int)
            traces_n_raw = traces_n_raw[idx]

        n_pvc  = sum(1 for p in peaks if p["cls"] == "pvc")
        n_norm = sum(1 for p in peaks if p["cls"] == "normal")
        peaks_max = traces_raw.max(axis=1, keepdims=True)
        peaks_max = np.where(peaks_max > 0.1, peaks_max, 1.0)
        traces_norm = traces_raw / peaks_max
        n_max = traces_n_raw.max(axis=1, keepdims=True) if len(traces_n_raw) else np.array([])
        if len(n_max):
            n_max = np.where(n_max > 0.05, n_max, 1.0)
            traces_n_norm = traces_n_raw / n_max
        else:
            traces_n_norm = traces_n_raw
        sessions.append({
            "label": label, "base": base, "ecg_path": ecg_path,
            "n_pvc": n_pvc, "n_norm": n_norm,
            "duration_min": float(t_ecg[-1]/60) if len(t_ecg) else 0,
            "n_excluded_intervals": len(excl),
            "excluded_seconds": float(sum(e-s for s, e in excl)),
            "traces_raw": traces_raw,
            "traces_norm": traces_norm,
            "traces_n_raw": traces_n_raw,
            "traces_n_norm": traces_n_norm,
            "coupling_ms": np.array(coupling_ms),
        })
    if not sessions:
        print("No valid session found."); return

    print(f"Sessions kept: {len(sessions)}")
    all_traces_norm = np.concatenate([s["traces_norm"] for s in sessions], axis=0)
    all_traces_raw  = np.concatenate([s["traces_raw"]  for s in sessions], axis=0)
    all_n_norm = np.concatenate([s["traces_n_norm"] for s in sessions
                                 if len(s["traces_n_norm"])], axis=0)
    all_coupling = np.concatenate([s["coupling_ms"] for s in sessions
                                   if len(s["coupling_ms"])])
    print(f"Total PVC traces: {len(all_traces_norm)}, N traces: {len(all_n_norm)}, "
          f"couplings: {len(all_coupling)}")

    # ============ MORPHOLOGY: single 4-panel figure with shared axes ============
    med_per_sess = [np.median(s["traces_norm"], axis=0) for s in sessions]
    corr_matrix = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            corr_matrix[i, j] = np.corrcoef(med_per_sess[i], med_per_sess[j])[0, 1]
    mask_t = (TG > 0.05) & (TG < 0.25)
    trough_depth = -all_traces_norm[:, mask_t].min(axis=1)

    # range Y comune per pannelli 1 e 2 (forma)
    med_all = np.median(all_traces_norm, axis=0)
    p25 = np.percentile(all_traces_norm, 25, axis=0)
    p75 = np.percentile(all_traces_norm, 75, axis=0)
    y_min = min(p25.min(), min(m.min() for m in med_per_sess)) - 0.05
    y_max = 1.10

    # font sizes for matplotlib plots (più piccoli del CSS body per gerarchia)
    FS_TITLE  = 11
    FS_LABEL  = 9.5
    FS_TICK   = 8.5
    FS_LEGEND = 8.5
    FS_TEXT   = 8

    # Figure PVC morphology: posizioni esplicite per allineamento perfetto.
    fig = plt.figure(figsize=(12, 9), facecolor=DARK_BG)

    # (1) — overlay tutte PVC (palette rossa)
    ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
    step = max(1, len(all_traces_norm) // 500)
    for tr in all_traces_norm[::step]:
        ax.plot(TG, tr, color="#ff8a8a", lw=0.4, alpha=0.06)
    ax.fill_between(TG, p25, p75, color="#ff6b6b", alpha=0.25, label="IQR")
    ax.plot(TG, med_all, color="#ff6b6b", lw=2.5, label="Median")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2); ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Amplitude (peak-normalized)", color="white", fontsize=FS_LABEL)
    ax.set_xlabel("Time relative to ectopic peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_title(f"All PVCs overlaid — median ± IQR  (n={len(all_traces_norm):,})",
                 color="#cccccc", fontsize=FS_TITLE)
    ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    # (2) — mediane per sessione, STESSI assi, legenda VERTICALE FUORI A DESTRA
    ax = fig.add_axes(PANEL_POS["tr"], sharex=fig.axes[0], sharey=fig.axes[0])
    ax.set_facecolor(DARK_BG)
    palette = ["#5fb1ff","#ff8a4d","#ff6b8a","#7fd693","#b598ff",
               "#ffe169","#5fcc9e","#ff9ec7","#7ac8ff","#ffb37d","#cc9966",
               "#8fb3c8","#d49fcc","#a8d6a3","#e0b8a0","#9aa6c4"]
    for i, s in enumerate(sessions):
        col = palette[i % len(palette)]
        m = np.median(s["traces_norm"], axis=0)
        ax.plot(TG, m, color=col, lw=1.1,
                label=f"{short_label(s['label'])} (n={len(s['traces_norm'])})")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlabel("Time relative to ectopic peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_title("Median morphology by session", color="#cccccc", fontsize=FS_TITLE)
    # legenda verticale FUORI dal box, sul lato destro
    leg = ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
                    fontsize=FS_TEXT-1, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), ncol=1,
                    handlelength=1.4, handletextpad=0.5, borderpad=0.6,
                    labelspacing=0.5)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    # (3) — correlation matrix nel pannello (1, 0)
    labs = [short_label(s["label"]) for s in sessions]
    ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0.95, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sessions))); ax.set_yticks(range(len(sessions)))
    ax.set_xticklabels(labs, color="#bbb", rotation=45, ha="right", fontsize=FS_TEXT)
    ax.set_yticklabels(labs, color="#bbb", fontsize=FS_TEXT)
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            ax.text(j, i, f"{corr_matrix[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=FS_TEXT-1)
    cbar = plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    cbar.ax.yaxis.label.set_color("#bbb")
    cbar.ax.yaxis.label.set_fontsize(FS_LABEL)
    cbar.ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    ax.set_title("Cross-session correlation matrix", color="#cccccc", fontsize=FS_TITLE)
    for sp in ax.spines.values(): sp.set_color("#333")

    # (4) — coupling interval distribution (RR_pre tutti PVC)
    ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
    if len(all_coupling) > 0:
        ax.hist(all_coupling, bins=60, color="#ff8a8a", edgecolor="#0d0f12",
                linewidth=0.3, density=True, alpha=0.85)
        med_c = float(np.median(all_coupling))
        ax.axvline(med_c, color="yellow", ls="--", lw=1.2, alpha=0.8,
                   label=f"median {med_c:.0f} ms")
        ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
                  fontsize=FS_LEGEND, loc="upper right")
    ax.set_xlabel("Coupling interval RR_pre (ms)", color="#bbb", fontsize=FS_LABEL)
    ax.set_ylabel("Density", color="#bbb", fontsize=FS_LABEL)
    ax.set_title(f"Coupling interval distribution  (n={len(all_coupling):,})",
                 color="#cccccc", fontsize=FS_TITLE)
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    img_morphology_4panel = fig_to_b64(fig, dpi=220)

    # ============ CONTINUUM CHECK (data-driven, check if bimodal subtypes exist) ============
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        # centro le tracce normalizzate (shape-focus)
        X = all_traces_norm - all_traces_norm.mean(axis=1, keepdims=True)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        clusters_2 = km2.labels_
        # inertia per elbow
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
                    for k in range(1, 7)]
        # heatmap PVC ordinate per profondità trough
        order = np.argsort(trough_depth)
        step_hm = max(1, len(all_traces_norm) // 600)
        heatmap_data = all_traces_norm[order][::step_hm]

        fig = plt.figure(figsize=(12, 9), facecolor=DARK_BG)

        # (1) PCA scatter + KMeans k=2
        ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
        for c, col, name in [(0, "#7ad9ff", "A"), (1, "#ff8a8a", "B")]:
            mask = clusters_2 == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, s=6, alpha=0.45,
                       label=f"Cluster {name} (n={mask.sum():,})")
        ax.set_xlabel("PC1", color="white", fontsize=FS_LABEL)
        ax.set_ylabel("PC2", color="white", fontsize=FS_LABEL)
        ax.set_title("PCA + K-means k=2 — artificial bipartition?",
                     color="#cccccc", fontsize=FS_TITLE)
        ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
                  fontsize=FS_LEGEND, loc="upper right")
        ax.tick_params(colors="#bbb", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.grid(alpha=0.18, color="#444")

        # (2) heatmap PVC ordinate per profondità trough
        ax = fig.add_axes(PANEL_POS["tr"]); ax.set_facecolor(DARK_BG)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                       extent=[TG[0], TG[-1], 0, len(heatmap_data)],
                       vmin=-0.6, vmax=1.0, origin="lower")
        ax.set_xlabel("Time relative to ectopic peak (s)",
                      color="white", fontsize=FS_LABEL)
        ax.set_ylabel("PVCs sorted by trough depth", color="white", fontsize=FS_LABEL)
        ax.set_title("PVCs sorted by hyperpolarization depth",
                     color="#cccccc", fontsize=FS_TITLE)
        ax.tick_params(colors="#bbb", labelsize=FS_TICK)
        cbar = plt.colorbar(im, ax=ax, label="Amplitude (norm.)",
                            fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color("#bbb")
        cbar.ax.yaxis.label.set_fontsize(FS_LABEL)
        cbar.ax.tick_params(colors="#bbb", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#333")

        # (3) elbow plot
        ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
        ax.plot(range(1, 7), inertias, marker="o", color="#ffe169", lw=2, ms=8)
        ax.set_xlabel("k (number of clusters)", color="white", fontsize=FS_LABEL)
        ax.set_ylabel("Within-cluster sum of squares", color="white", fontsize=FS_LABEL)
        ax.set_title("Elbow plot — discrete clusters?",
                     color="#cccccc", fontsize=FS_TITLE)
        ax.tick_params(colors="#bbb", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.grid(alpha=0.18, color="#444")

        # (4) trough depth distribution (spostata qui dal pannello morfologia)
        ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
        ax.hist(trough_depth, bins=60, color="#ff8a8a", edgecolor="#0d0f12",
                linewidth=0.3, density=True, alpha=0.85)
        ax.set_xlabel("Post-QRS trough depth (peak-normalized)",
                      color="#bbb", fontsize=FS_LABEL)
        ax.set_ylabel("Density", color="#bbb", fontsize=FS_LABEL)
        ax.set_title(f"Hyperpolarization depth distribution  (n={len(trough_depth):,})",
                     color="#cccccc", fontsize=FS_TITLE)
        ax.tick_params(colors="#bbb", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.grid(alpha=0.18, color="#444")

        img_pvc_continuum = fig_to_b64(fig, dpi=220)
    except Exception as e:
        print(f"  warning: continuum check failed: {e}")
        img_pvc_continuum = None

    # ============ NORMAL BEATS MORPHOLOGY (4-panel, same layout) ============
    med_per_sess_n = [np.median(s["traces_n_norm"], axis=0)
                      for s in sessions if len(s["traces_n_norm"])]
    corr_matrix_n = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            corr_matrix_n[i, j] = np.corrcoef(med_per_sess_n[i], med_per_sess_n[j])[0, 1]
    med_all_n = np.median(all_n_norm, axis=0)
    p25_n = np.percentile(all_n_norm, 25, axis=0)
    p75_n = np.percentile(all_n_norm, 75, axis=0)
    y_min_n = min(p25_n.min(), min(m.min() for m in med_per_sess_n)) - 0.05
    y_max_n = 1.10

    fig = plt.figure(figsize=(12, 9), facecolor=DARK_BG)

    # (1,1) — overlay tutti N
    ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
    step_n = max(1, len(all_n_norm) // 500)
    for tr in all_n_norm[::step_n]:
        ax.plot(TG, tr, color="#5fcc9e", lw=0.4, alpha=0.06)
    ax.fill_between(TG, p25_n, p75_n, color="#7fd693", alpha=0.25, label="IQR")
    ax.plot(TG, med_all_n, color="#7fd693", lw=2.5, label="Median")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2); ax.set_ylim(y_min_n, y_max_n)
    ax.set_ylabel("Amplitude (peak-normalized)", color="white", fontsize=FS_LABEL)
    ax.set_xlabel("Time relative to sinus peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_title(f"All N beats overlaid (sampled, n={len(all_n_norm):,}) — median ± IQR",
                 color="#cccccc", fontsize=FS_TITLE)
    ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    # (1,2) — mediane N per sessione
    ax = fig.add_axes(PANEL_POS["tr"], sharex=fig.axes[-1], sharey=fig.axes[-1])
    ax.set_facecolor(DARK_BG)
    for i, s in enumerate(sessions):
        if not len(s["traces_n_norm"]): continue
        col = palette[i % len(palette)]
        m = np.median(s["traces_n_norm"], axis=0)
        ax.plot(TG, m, color=col, lw=1.1,
                label=f"{short_label(s['label'])} (n={len(s['traces_n_norm'])})")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlabel("Time relative to sinus peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_title("Median N morphology by session", color="#cccccc", fontsize=FS_TITLE)
    leg = ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
                    fontsize=FS_TEXT-1, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), ncol=1,
                    handlelength=1.4, handletextpad=0.5, borderpad=0.6,
                    labelspacing=0.5)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    # (2,1) — correlation matrix N
    ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
    im = ax.imshow(corr_matrix_n, cmap="RdYlGn", vmin=0.95, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sessions))); ax.set_yticks(range(len(sessions)))
    ax.set_xticklabels(labs, color="#bbb", rotation=45, ha="right", fontsize=FS_TEXT)
    ax.set_yticklabels(labs, color="#bbb", fontsize=FS_TEXT)
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            ax.text(j, i, f"{corr_matrix_n[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=FS_TEXT-1)
    cbar = plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    cbar.ax.yaxis.label.set_color("#bbb")
    cbar.ax.yaxis.label.set_fontsize(FS_LABEL)
    cbar.ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    ax.set_title("Cross-session correlation matrix (N beats)",
                 color="#cccccc", fontsize=FS_TITLE)
    for sp in ax.spines.values(): sp.set_color("#333")

    # (2,2) — direct comparison: median PVC vs median N (overlap)
    ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
    ax.plot(TG, med_all_n,  color="#7fd693", lw=2.5, label=f"N (median, n={len(all_n_norm):,})")
    ax.plot(TG, med_all,    color="#ff6b6b", lw=2.5, label=f"PVC (median, n={len(all_traces_norm):,})")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2)
    ax.set_xlabel("Time relative to peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_ylabel("Amplitude (peak-normalized)", color="white", fontsize=FS_LABEL)
    ax.set_title("N vs PVC — median morphology comparison",
                 color="#cccccc", fontsize=FS_TITLE)
    ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")

    img_n_morphology_4panel = fig_to_b64(fig, dpi=220)

    # ============ OUTLIER ANALYSIS — N beats ============
    # Calcola mean correlation di ogni sessione vs tutte le altre
    n_sessions = len(sessions)
    mean_corr_per_session = []
    for i in range(n_sessions):
        rs = [corr_matrix_n[i, j] for j in range(n_sessions) if j != i]
        mean_corr_per_session.append((sessions[i]["label"], float(np.mean(rs)), i))
    mean_corr_per_session.sort(key=lambda x: x[1])
    outlier_label, outlier_r, outlier_idx = mean_corr_per_session[0]

    # plot: mediana sessione outlier vs mediana di tutte le ALTRE
    median_outlier = med_per_sess_n[outlier_idx]
    others = [m for i, m in enumerate(med_per_sess_n) if i != outlier_idx]
    median_others = np.median(np.array(others), axis=0)
    p25_others = np.percentile(np.array(others), 25, axis=0)
    p75_others = np.percentile(np.array(others), 75, axis=0)

    # stessa dimensione del pannello correlation matrix nella 4-grid
    # (figsize=(7, 5.5)) + extra spazio a destra per legenda esterna
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.set_box_aspect(1)
    ax.fill_between(TG, p25_others, p75_others, color="#7fd693", alpha=0.20,
                    label=f"Other {n_sessions-1} sessions (IQR)")
    ax.plot(TG, median_others, color="#7fd693", lw=2,
            label=f"Median of other {n_sessions-1}")
    ax.plot(TG, median_outlier, color="#ff8a8a", lw=2.5,
            label=f"Outlier {outlier_label}\n(mean r = {outlier_r:.3f})")
    ax.axvline(0, color="#888", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2)
    ax.set_xlabel("Time relative to sinus peak (s)", color="white", fontsize=FS_LABEL)
    ax.set_ylabel("Amplitude (peak-normalized)", color="white", fontsize=FS_LABEL)
    ax.set_title("Outlier vs other sessions (median N)",
                 color="#cccccc", fontsize=FS_TITLE)
    # legenda esterna a destra
    leg = ax.legend(facecolor="#1a1d22", labelcolor="white", edgecolor="#333",
                    fontsize=FS_LEGEND, loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    handlelength=1.4, handletextpad=0.5,
                    borderpad=0.6, labelspacing=0.6)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#bbb", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.18, color="#444")
    img_n_outlier = fig_to_b64(fig, dpi=220)

    # tabella mean corr per sessione (HTML)
    outlier_rows = "\n".join(
        f"<tr><td>{lab}</td><td class='num'>{mc:.3f}</td></tr>"
        for lab, mc, _ in mean_corr_per_session)

    # ============ AGGREGATE STATS ============
    cum_total_pvc  = sum(s["n_pvc"] for s in sessions)
    cum_total_norm = sum(s["n_norm"] for s in sessions)
    cum_duration   = sum(s["duration_min"] for s in sessions)
    cum_excluded   = sum(s["excluded_seconds"] for s in sessions)

    # Sessions table
    sessions_rows = []
    for s in sessions:
        burden = 100*s["n_pvc"]/(s["n_norm"]+s["n_pvc"]) if (s["n_norm"]+s["n_pvc"]) else 0
        sessions_rows.append(
            f"<tr><td>{s['label']}</td>"
            f"<td>{s['duration_min']:.1f} min</td>"
            f"<td>{s['n_norm']:,}</td>"
            f"<td>{s['n_pvc']:,}</td>"
            f"<td>{burden:.1f}%</td>"
            f"<td>{s['n_excluded_intervals']} ({s['excluded_seconds']:.0f}s)</td>"
            "</tr>"
        )
    sessions_table = "\n".join(sessions_rows)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Holter ECG DIY — Personal Cardiac Monitoring Dashboard</title>
<style>
  body {{ background:#0d0f12; color:#cfd2d6; font-family:-apple-system,BlinkMacSystemFont,sans-serif;
         margin: 24px auto; max-width: 1400px; line-height: 1.5; padding: 0 18px;
         font-size: 15.5px; }}
  h1 {{ color:#e6e8ea; font-size: 1.6em; margin-bottom: 4px; font-weight: 600; }}
  .subtitle {{ color:#888; font-size:0.92em; margin-bottom: 4px; }}
  .updated  {{ color:#666; font-size:0.82em; }}
  h2 {{ color:#c8ccd0; border-bottom:1px solid #2a2d33; padding-bottom:5px;
        margin-top:30px; font-size: 1.22em; font-weight: 600; }}
  h3 {{ color:#a8acb2; font-size:1em; margin-top: 14px; margin-bottom: 6px;
        font-weight: 600; }}
  details {{ background:#15171b; border:1px solid #25282d; border-radius: 6px;
             margin: 10px 0; }}
  details > summary {{ cursor: pointer; padding: 10px 16px; color:#a8acb2;
                       font-weight: 600; font-size: 0.95em; list-style: none;
                       user-select: none; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details > summary::before {{ content: "▸  "; color:#666; }}
  details[open] > summary::before {{ content: "▾  "; color:#666; }}
  details[open] > summary {{ border-bottom: 1px solid #25282d; }}
  details > .content {{ padding: 12px 18px; }}
  .stat-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
                gap: 10px; margin: 14px 0; }}
  .stat {{ background:#15171b; padding: 10px 14px; border-radius: 6px;
           border-left: 3px solid #4a90a4; }}
  .stat .v {{ display:block; font-size:1.4em; color:#e6e8ea; font-weight:600; }}
  .stat .l {{ display:block; color:#888; font-size:0.78em; margin-top:2px; }}
  .stat.pvc {{ border-left-color: #ff6b6b; }}
  .stat.pvc .v {{ color: #ff8a8a; }}
  .stat.burden {{ border-left-color: #ff9c45; }}
  .stat.burden .v {{ color: #ffb070; }}
  .stat.heartbeats {{ border-left-color: #5fb1ff; }}
  .stat.heartbeats .v {{ color: #7ac8ff; }}
  table {{ border-collapse:collapse; width:100%; margin: 10px 0; font-size:0.85em; }}
  th, td {{ padding:5px 10px; border-bottom:1px solid #2a2d33; text-align:left; }}
  th {{ background:#23272d; color:#cfd2d6; font-weight: 600; }}
  tr:nth-child(even) td {{ background:#15171b; }}
  td.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  img {{ display:block; max-width:100%; margin: 0;
         border:1px solid #2a2d33; border-radius:4px; background:#0d0f12; }}
  .commentary {{ background:#15171b; padding:8px 14px;
                 border-left:3px solid #4a90a4; margin:8px 0;
                 border-radius:0 5px 5px 0; font-size: 0.85em; }}
  .commentary b {{ color:#e6e8ea; }}
  .device-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px 26px;
                  padding: 0; font-size: 0.88em; }}
  .device-grid dt {{ color:#a8acb2; font-weight: 600; }}
  .device-grid dd {{ color:#cfd2d6; margin: 0 0 3px 0; }}
  @media (max-width: 900px) {{
    .device-grid {{ grid-template-columns: 1fr; }}
  }}
  code {{ background:#23272d; color:#cfd2d6; padding:1px 5px; border-radius:3px;
          font-size:0.82em; }}
  footer {{ margin-top:32px; padding-top:10px; border-top:1px solid #25282d;
            color:#666; font-size:0.78em; line-height:1.55; }}
</style>
</head>
<body>

<h1>Holter ECG DIY — Personal Cardiac Monitoring Dashboard</h1>
<div class="subtitle">
  Longitudinal aggregation of all self-recorded ECG sessions
</div>
<div class="updated">
  Last update <code>{now}</code> · refresh by running
  <code>python3 host/dashboard.py</code>
</div>

<h2>Dataset overview</h2>
<div class="stat-grid">
  <div class="stat"><span class="v">{len(sessions)}</span>
    <span class="l">sessions analyzed</span></div>
  <div class="stat"><span class="v">{cum_duration:.0f}</span>
    <span class="l">total minutes ({cum_duration/60:.1f} h)</span></div>
  <div class="stat heartbeats"><span class="v">{cum_total_pvc + cum_total_norm:,}</span>
    <span class="l">heartbeats classified (N + PVC)</span></div>
  <div class="stat pvc"><span class="v">{cum_total_pvc:,}</span>
    <span class="l">PVCs detected</span></div>
  <div class="stat burden"><span class="v">{100*cum_total_pvc/max(1,(cum_total_pvc+cum_total_norm)):.1f}%</span>
    <span class="l">cumulative PVC burden</span></div>
  <div class="stat"><span class="v">{cum_excluded/60:.1f} min</span>
    <span class="l">manually excluded as noise</span></div>
</div>

<details>
  <summary>Acquisition setup &amp; processing pipeline</summary>
  <div class="content">
    <dl class="device-grid">
      <dt>Front-end</dt>
      <dd>AD8232 single-lead bio-potential amplifier</dd>
      <dt>Lead configuration</dt>
      <dd>3 snap electrodes, single precordial derivation</dd>
      <dt>Microcontroller</dt>
      <dd>Raspberry Pi Pico 2 W (MicroPython)</dd>
      <dt>ADC</dt>
      <dd>Pico internal 12-bit, channel <code>ADC0 (GP26)</code></dd>
      <dt>Sample rate</dt>
      <dd>{SR} Hz</dd>
      <dt>Transport</dt>
      <dd>WiFi (TCP, port 5005) or USB serial (mpremote)</dd>
      <dt>Power</dt>
      <dd>1S LiPo + TP4056 charger + SPDT switch, ~24 h autonomy</dd>
      <dt>Server / analysis</dt>
      <dd>Python (Flask + SSE dashboard, batch report tools) on MacBook / Pi 5</dd>
      <dt>Bandpass filter</dt>
      <dd>1st-order IIR: HP 0.3 Hz + LP 25 Hz (online, on server)</dd>
      <dt>QRS detector</dt>
      <dd>4-state FSM (IDLE → WIDTH → DETECT → POST), refractory 300 ms</dd>
      <dt>PVC classifier</dt>
      <dd>(rebound ≥ {REBOUND_PVC} <em>or</em> width ≥ {PVC_W_MS:.0f} ms)
          AND amplitude ≥ {PVC_MIN_AMP} V
          AND {PVC_W_MIN:.0f} ≤ width ≤ {PVC_W_MAX:.0f} ms
          AND rebound ≥ {PVC_MIN_REBOUND}</dd>
      <dt>Manual noise exclusion</dt>
      <dd><code>host/mark_exclusions.py</code> — paginated matplotlib editor,
          per-session JSON under <code>exclusions/</code></dd>
    </dl>
    <div class="commentary" style="margin-top:12px">
      Setup is hobbyist / educational. <b>This is not a medical device</b> and
      the analyses below are not diagnostic. All clinical decisions are made
      under cardiologist supervision; data here serves to characterize
      long-term patterns of the personal ectopic focus and to track effects of
      postural/breathing interventions.
    </div>
  </div>
</details>

<details>
  <summary>Sessions table ({len(sessions)} sessions)</summary>
  <div class="content">
    <table>
      <tr><th>Date / time</th><th>Duration</th><th>N beats</th><th>PVC</th>
          <th>Burden</th><th>Excluded</th></tr>
      {sessions_table}
    </table>
  </div>
</details>

<h2>PVC morphology analysis</h2>
<div class="commentary">
  Each PVC is centered on its ectopic peak, normalized to its own amplitude
  (peak = 1.0), and overlaid in a common time window of ±{WIN/2*1000:.0f} ms.
  Conservation of this shape across sessions may suggest a single stable
  ectopic focus; visible divergences could indicate multifocality or
  substrate changes. Panels 1-2 share the same Y-axis scale for direct
  visual comparison.
</div>
<img src="data:image/png;base64,{img_morphology_4panel}" alt="PVC morphology summary"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1000px; display:block; margin: 0 auto;"/>

<h3>Are PVCs a single continuous population or do they form discrete subtypes?</h3>
<div class="commentary">
  Visual inspection of the "Median morphology by session" panel above suggests
  that one session (2026-06-06 13:41, n ≈ 924) shows a slightly different
  hyperpolarization profile compared to the others. This raises the question
  of whether the variability across PVCs reflects a single ectopic focus
  with continuous intrinsic variability, or whether two (or more) discrete
  morphological subtypes coexist — which would suggest mild bifocality.
  Three data-driven checks address this:
  <ul>
    <li><b>PCA + K-means (k=2)</b>: forces a bipartition of all PVCs in
        principal-component space. If the two clusters appear as two
        well-separated clouds, that is evidence for two real subtypes; if
        they look like one connected cloud arbitrarily cut down the middle,
        the partition is an artefact of the algorithm.</li>
    <li><b>PVCs sorted by hyperpolarization depth</b>: a 2D heatmap where
        each row is one PVC, ordered by the depth of its post-QRS trough.
        Two discrete bands (sharp transition) would indicate two subtypes;
        a smooth color gradient indicates a continuum.</li>
    <li><b>Elbow plot</b>: K-means inertia as a function of k. A sharp
        "knee" at some k* would suggest k* natural clusters; a smoothly
        decaying curve indicates no preferred number of clusters.</li>
    <li><b>Hyperpolarization depth distribution</b>: histogram of post-QRS
        trough depth across all PVCs. A single unimodal peak is consistent
        with a single focus and continuous variability; two well-separated
        peaks would suggest two morphological subtypes.</li>
  </ul>
</div>
<img src="data:image/png;base64,{img_pvc_continuum}" alt="PVC continuum check"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1000px; display:block; margin: 0 auto;"/>

<h2>Normal beats morphology</h2>
<div class="commentary">
  Same superimposition analysis applied to normal sinus beats (N), with up to 500
  evenly-spaced N samples per session to control memory footprint. Together with
  the PVC analysis above it serves as a baseline reference. Inspection of N
  beats may help identify electrode-placement or posture effects (in the
  observed amplitude modulation pattern, R-amplitude appears to vary
  sinusoidally with breathing). The fourth panel directly overlays the median
  N and the median PVC for visual comparison.
</div>
<img src="data:image/png;base64,{img_n_morphology_4panel}" alt="Normal beat morphology — 4-panel summary"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1000px; display:block; margin: 0 auto;"/>

<h3>Cross-session N correlation — outlier inspection</h3>
<div class="commentary">
  The N correlation matrix shows some cells below the upper bound of the
  color scale (r &lt; 0.97). All values remain ≥ 0.93, which in absolute
  terms is still a high concordance. The colormap is configured with
  vmin=0.95 to maximize contrast and surface any session that drifts
  slightly from the others.
  <p style="margin: 8px 0 4px 0"><b>Observed difference in this dataset</b>:
  the outlier session (09 13:23) is the only one in which the subject was
  seated at a desk with anterior trunk flexion, while all other sessions
  were recorded in lying / extended postures. This posture difference may
  explain the morphology drift, since a change in trunk position can rotate
  the heart axis relative to the precordial electrode and modify the
  projection of the depolarization vector on the single-lead signal.</p>
  <p style="margin: 4px 0 0 0">A plausible reason why the effect surfaces
  in N and not in PVC: PVCs have a large dominant deflection (~1.5 V) whose
  shape, once amplitude-normalized, tends to be robust to small perturbations
  of the projection; N beats are smaller (~0.5 V) and roughly symmetric
  around baseline, so normalization amplifies the relative weight of P-wave
  and T-wave regions, which may be more sensitive to posture changes. This
  is an interpretation of the observed data, not a confirmed mechanism.</p>
</div>

<div style="display:grid; grid-template-columns: 280px 680px;
            gap: 22px; align-items: center; justify-content: center;
            margin: 14px auto; max-width: 1020px;">
  <div>
    <table style="margin: 0;">
      <tr><th>Session</th><th>Mean r</th></tr>
      {outlier_rows}
    </table>
    <p style="color:#888; font-size:0.82em; margin-top:10px; margin-bottom:0;">
      Outlier: <code>{outlier_label}</code> (mean r = <b>{outlier_r:.3f}</b>).
      Plot on the right: outlier median N morphology vs median (and IQR) of
      the other {len(sessions)-1} sessions.
    </p>
  </div>
  <div>
    <img src="data:image/png;base64,{img_n_outlier}" alt="N outlier comparison"
         style="border:1px solid #25282d; border-radius:6px;
                width:100%; display:block;"/>
  </div>
</div>

<footer>
  Sample rate {SR} Hz · single precordial lead AD8232 · 1st-order IIR bandpass
  0.3-25 Hz · QRS detector and PVC classifier as described in the
  &laquo;Acquisition setup&raquo; box above. Sessions filtered:
  size ≥ {MIN_FILE_SIZE_MB} MB and ≥ {MIN_PVC_COUNT} PVCs.<br/>
  This dashboard is intended for personal use and as a discussion document with
  the supervising cardiologist. It does <b>not</b> constitute a diagnostic
  report. To export a printable copy, open in a browser and use
  <em>Print → Save as PDF</em>.
</footer>

</body>
</html>"""

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/holter_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n✓ Dashboard written: {out_path}")
    print(f"  Open in browser. Use Cmd+P → Save as PDF for export.")
    return out_path

if __name__ == "__main__":
    main()
